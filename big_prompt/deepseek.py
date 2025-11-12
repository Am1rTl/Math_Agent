import os
import json
import time
import uuid
import requests
import sympy
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import io
import sys
import math
import itertools
import multiprocessing as mp
import tempfile
try:
	import resource
except ImportError:  # pragma: no cover - Windows fallback
	resource = None
from contextlib import redirect_stdout, redirect_stderr
from openai import OpenAI

from settings import get_settings
from telemetry import init_tracer, trace_context, start_span
from agent_ops import (
	SchemaValidator,
	FewShotRetriever,
	RoutingManager,
	LMJudge,
	AgentMetrics,
	HumanEscalationQueue,
)

# ========== ИЗМЕНЕНИЕ 1: ГЛОБАЛЬНЫЕ НАСТРОЙКИ ДЛЯ ОТЛАДКИ ==========
DEBUG_MODE = False
LOG_FILE = "math_agent.log"
SETTINGS = get_settings()

# ========== КОНФИГУРАЦИЯ OPENROUTER ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
#DEFAULT_MODEL = "microsoft/phi-4-multimodal-instruct"
#DEFAULT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"
#DEFAULT_MODEL = "qwen/qwen3-32b"
DEFAULT_MODEL = "google/gemini-2.0-flash-lite-001"

# Инициализация клиента OpenRouter
client = OpenAI(
	base_url="https://openrouter.ai/api/v1",
	api_key=OPENROUTER_API_KEY
)

init_tracer(SETTINGS)
schema_validator = SchemaValidator(SETTINGS)
few_shot_retriever = FewShotRetriever()
routing_manager = RoutingManager(SETTINGS)
agent_metrics = AgentMetrics()
human_queue = HumanEscalationQueue(SETTINGS)

# ========== ИЗМЕНЕНИЕ 2: УЛУЧШЕННАЯ ФУНКЦИЯ ИЗВЛЕЧЕНИЯ JSON ==========
def _robust_extract_json(text: str) -> str:
	"""
	Более надежная функция для извлечения JSON из ответа AI.
	Ищет либо ```json ... ```, либо первый попавшийся блок {...} или [...].
	"""
	text = text.strip()
	max_window = SETTINGS.thresholds.max_schema_bytes

	def _balanced(candidate: str) -> bool:
		stack = []
		pairs = {'}': '{', ']': '['}
		for ch in candidate:
			if ch in '{[':
				stack.append(ch)
			elif ch in pairs:
				if not stack or stack.pop() != pairs[ch]:
					return False
		return not stack

	# 1. Сначала ищем блоки ```json
	json_match = re.search(r'```json\s*([\s\S]+?)\s*```', text)
	if json_match:
		return json_match.group(1).strip()
	
	# 2. Если не нашли, ищем первый '{' или '['
	start_index_curly = text.find('{')
	start_index_square = text.find('[')
	
	start_index = -1
	start_char = ''
	end_char = ''
	
	if start_index_curly == -1 and start_index_square == -1:
		return text # Ничего похожего на JSON не найдено

	if start_index_curly != -1 and (start_index_curly < start_index_square or start_index_square == -1):
		start_index = start_index_curly
		start_char = '{'
		end_char = '}'
	else:
		start_index = start_index_square
		start_char = '['
		end_char = ']'

	depth = 0
	for idx in range(start_index, min(len(text), start_index + max_window)):
		char = text[idx]
		if char == start_char:
			depth += 1
		elif char == end_char:
			depth -= 1
			if depth == 0:
				candidate = text[start_index: idx + 1]
				if _balanced(candidate):
					return candidate
				break

	# Если не нашли сбалансированный блок, возвращаем исходный текст
	return text[start_index: min(len(text), start_index + max_window)]


ACTION_REQUIRED_FIELDS = {
	"CALL_PYTHON": ["code"],
	"CALL_SYMPY": ["expression"],
	"CALL_WOLFRAM": ["query"],
	"FIX_CODE": ["error", "new_code"],
	"ADD_STEP": ["step_description"],
	"MODIFY_PLAN": ["new_steps", "reason"],
	"FINISH": ["answer", "values"]
}


def _validate_action_schema(action: Dict[str, Any]) -> Tuple[bool, str]:
	action_type = action.get("type")
	if action_type not in ACTION_REQUIRED_FIELDS:
		return False, f"Неизвестный тип действия: {action_type}"
	payload = action.get("payload")
	if action_type == "FINISH":
		# FINISH валидируется отдельным валидатором
		return True, ""
	if not isinstance(payload, dict):
		return False, "payload обязан быть объектом"
	missing = [field for field in ACTION_REQUIRED_FIELDS[action_type] if field not in payload]
	if missing:
		return False, f"Не хватает полей в payload: {missing}"
	return True, ""


def _sandbox_python_worker(code: str, sandbox_conf, queue: mp.Queue, workdir: str):
	stdout_capture = io.StringIO()
	stderr_capture = io.StringIO()
	result_value = None
	start_time = time.time()

	try:
		os.chdir(workdir)
		sys.path = [workdir]
		if resource:
			try:
				resource.setrlimit(resource.RLIMIT_CPU, (sandbox_conf.cpu_time_seconds, sandbox_conf.cpu_time_seconds))
				mem_bytes = sandbox_conf.memory_limit_mb * 1024 * 1024
				resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
			except Exception:
				pass

		blocked_modules = set(sandbox_conf.blocked_modules)
		import builtins
		original_import = builtins.__import__

		def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
			if name.split(".")[0] in blocked_modules:
				raise ImportError(f"Модуль {name} запрещен политикой sandbox")
			return original_import(name, globals, locals, fromlist, level)

		builtins.__import__ = guarded_import

		safe_globals = {
			"math": math,
			"numpy": np,
			"np": np,
			"sympy": sympy,
			"json": json,
			"re": re,
			"itertools": itertools,
			"time": time,
			"result": None,
			"__builtins__": builtins.__dict__
		}

		with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
			exec(code, safe_globals)
			result_value = safe_globals.get("result")

	except Exception as exc:
		stderr_capture.write(f"{type(exc).__name__}: {exc}")
	finally:
		queue.put({
			"stdout": stdout_capture.getvalue(),
			"stderr": stderr_capture.getvalue(),
			"result_value": result_value,
			"duration": time.time() - start_time
		})


def generate_with_ai(system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: Optional[float] = None, role: str = "generic") -> str:
	"""
	Универсальная функция для обращения к нейросети через OpenRouter
	
	Args:
		system_prompt: Системный промпт
		user_prompt: Пользовательский промпт  
		model: Модель для использования (по умолчанию phi-4)
	
	Returns:
		Текст ответа от нейросети
	"""
	# ========== ИЗМЕНЕНИЕ 3: ЛОГИРОВАНИЕ В РЕЖИМЕ ОТЛАДКИ ==========
	global DEBUG_MODE 
	decision = None
	if model is None:
		decision = routing_manager.route(user_prompt, role=role)
		model = decision.model
		if temperature is None:
			temperature = decision.temperature
	span_attributes = {
		"gen_ai.request.model": model,
		"gen_ai.request.role": role,
		"gen_ai.router.reason": decision.reason if decision else "override",
		"gen_ai.router.system": decision.system if decision else "override"
	}
	with start_span("llm.call", span_attributes) as span:
		try:
			start_time = time.time()
			completion = client.chat.completions.create(
				model=model,
				messages=[
				{
					"role": "system",
					"content": system_prompt
				},
				{
					"role": "user", 
					"content": user_prompt
				}
			],
			#max_tokens=4000,
			temperature=temperature if temperature is not None else SETTINGS.models.system2_temperature
			)

			response_content = completion.choices[0].message.content.strip()
			elapsed = time.time() - start_time
			usage = getattr(completion, "usage", None)
			total_tokens = 0
			if usage:
				total_tokens = usage.get("total_tokens") or usage.get("totalTokens") or 0
				agent_metrics.token_usage += total_tokens
			span.set_attribute("gen_ai.response.duration_ms", elapsed * 1000)
			span.set_attribute("gen_ai.response.tokens", total_tokens)
			span.set_attribute("gen_ai.router.temperature", temperature)

			if DEBUG_MODE:
				with open(LOG_FILE, "a", encoding="utf-8") as f:
					f.write(f"===== AI CALL ({model}) - {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
					f.write(f"--- SYSTEM PROMPT ---\n{system_prompt}\n")
					f.write(f"--- USER PROMPT ---\n{user_prompt}\n")
					f.write(f"--- RAW AI RESPONSE ---\n{response_content}\n")
					f.write("=" * 50 + "\n\n")

			return response_content

		except Exception as e:
			print(f"Ошибка при обращении к нейросети: {e}")
			if DEBUG_MODE:
				with open(LOG_FILE, "a", encoding="utf-8") as f:
					f.write(f"===== AI CALL FAILED ({model}) - {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
					f.write(f"--- ERROR ---\n{str(e)}\n")
					f.write("=" * 50 + "\n\n")
			span.set_attribute("gen_ai.response.error", str(e))
			# Возвращаем пустую строку, чтобы вызвать ошибку JSON, а не падение
			return ""


def _judge_call(system_prompt: str, user_prompt: str, role: str = "judge") -> str:
	return generate_with_ai(
		system_prompt,
		user_prompt,
		model=SETTINGS.models.system2_model,
		temperature=SETTINGS.models.verifier_temperature,
		role=role
	)


lm_judge = LMJudge(_judge_call, SETTINGS)

# ========== МОДЕЛИ ДАННЫХ ==========
class ActionType(Enum):
	CALL_PYTHON = "CALL_PYTHON"
	CALL_SYMPY = "CALL_SYMPY" 
	CALL_WOLFRAM = "CALL_WOLFRAM"
	CALL_PAL_GENERATOR = "CALL_PAL_GENERATOR"
	FINISH = "FINISH"
	ADD_STEP = "ADD_STEP"
	MODIFY_PLAN = "MODIFY_PLAN"
	FIX_CODE = "FIX_CODE"

@dataclass
class ProblemObject:
	id: str
	statement: str
	entities: List[Dict]
	constraints: List[str]
	goals: List[Dict]
	hypotheses: List[Dict]
	irrelevant: List[str]
	notes: str

@dataclass
class Plan:
	id: str
	summary: str
	steps: List[str]
	estimated_complexity: str
	estimated_tooling: List[str]
	heuristic_score: float
	rationale: str

@dataclass
class ThoughtAction:
	thought: str
	action: Optional[Dict] = None

@dataclass
class ExecutionTrace:
	step_id: str
	thought: str
	action: Optional[Dict]
	observation: Optional[Dict]
	timestamp: float

@dataclass
class FormalTrace:
	step_id: str
	formal_statement: str
	verification_status: str
	evidence: List[Dict]
	confidence: float

# ========== ПРОМПТЫ СИСТЕМЫ ==========
# ========== ИЗМЕНЕНИЕ 12: Промпт формализатора сделан более строгим ==========
INPUT_FORMALIZER_SYSTEM_PROMPT = """Ты - семантический парсер. Твой ответ ДОЛЖЕН быть *ТОЛЬКО* JSON-объектом.
Не добавляй никакого сопроводительного текста, '```json' или '```'.

{
	"id": "уникальный_идентификатор",
	"statement": "оригинальный_текст",
	"entities": [{"name": "имя", "value": "число/выражение", "unit": "единица"}],
	"constraints": ["ограничения"],
	"goals": [{"type": "value/prove/simplify", "target": "цель"}],
	"hypotheses": [],
	"irrelevant": [],
	"notes": "заметки"
}"""

# ========== ИЗМЕНЕНИЕ 13: Промпт планировщика сделан более стратегическим ==========
TOT_PLANNER_SYSTEM_PROMPT = """Ты - ВЫСОКОУРОВНЕВЫЙ стратегический планировщик.
Твоя задача - предложить 3-5 РАЗНЫХ *стратегий* (планов) для решения задачи.
Не нужно излишне детализировать шаги; фокусируйся на общей идее и необходимых инструментах.
Отвечай ТОЛЬКО в формате JSON-массива. Используй примеры ниже как ориентир при необходимости:

[
	{
		"id": "plan_touching_curves",
		"summary": "Через касание окружности и гиперболы",
		"steps": [
			"Выписать границы множества (окружность/эллипс)",
			"Найти условие касания между кривыми",
			"Решить уравнение касания относительно параметра a",
			"Проверить количество решений системы на найденных a"
		],
		"estimated_complexity": "high",
		"estimated_tooling": ["Python", "SymPy"],
		"heuristic_score": 0.82,
		"rationale": "Касание гарантирует переход между числами решений."
	}
]
"""

# ========== ИЗМЕНЕНИЕ 14: Промпт исполнителя (ReAct) обновлен новыми действиями ==========
# ========== ИЗМЕНЕНИЕ 16: Правила 5 и 6 изменены, чтобы форсировать вычисления ==========
# ========== ИЗМЕНЕНИЕ 16: Правила 5 и 6 изменены, чтобы форсировать вычисления ==========
REACT_EXECUTOR_SYSTEM_PROMPT = """Ты - тактический исполнитель ReAct.
Твой ответ ДОЛЖЕН СТРОГО следовать формату:
Thought: [Твой анализ и рассуждения]
Action: [ОДИН JSON-объект]

ИЛИ (если действие не нужно):
Thought: [Твой анализ и рассуждения]

ЗАПРЕЩЕНО добавлять что-либо до "Thought:" или после "Action: {json_действие}".

ВАЖНЫЕ ПРАВИЛА ДЕЙСТВИЙ:
1. КОД: Для `CALL_PYTHON` ВСЕГДА генерируй Python код, который ВЫВОДИТ результаты через `print()`.
2. КОНТЕКСТ: Внимательно читай `ОБЩАЯ ЗАДАЧА`, `КЛЮЧЕВЫЕ СУЩНОСТИ` и `few-shot` примеры.
3. ОШИБКИ: Если код содержит ошибку, используй `FIX_CODE`.
4. ЗАВЕРШЕНИЕ: Когда задача решена и ты вычислил значения параметра `a`, вызывай `FINISH` строго по схеме.
5. ГЛАВНОЕ ПРАВИЛО: Не делай финальные вычисления в Thought — только через действия или FINISH.
6. ПРОМЕЖУТОЧНЫЕ ШАГИ: Используй `CALL_PYTHON` для расчетов и `ADD_STEP`/`MODIFY_PLAN`, если нужно уточнить стратегию.
7. FINISH ДОЛЖЕН содержать trace_id из контекста и список чисел `values.a`.

ФОРМАТ ДЕЙСТВИЙ (СТРОГИЙ JSON):
CALL_PYTHON:
{
	"type": "CALL_PYTHON",
	"payload": {
		"code": "import sympy as sp\\n..."
	}
}

FINISH (обязательно соблюдай схему):
{
	"type": "FINISH",
	"payload": {
		"answer": "строка",
		"values": {"a": [число, число]},
		"confidence": 0.0-1.0,
		"trace_id": "<trace_id из контекста>",
		"units": "опционально",
		"reasoning": "краткое обоснование"
	}
}"""

# ========== ИЗМЕНЕНИЕ 15: Промпт верификатора сделан более строгим ==========
VERIFIER_SYSTEM_PROMPT = """Ты - формальный верификатор. Проверь трассировку.
Отвечай ТОЛЬКО в формате JSON-**массива** (списка), даже если в нем всего один элемент.
Всегда проверяй, что финальный ответ содержит числовые значения `a` и соответствует цели задачи.
Не добавляй никакого сопроводительного текста, '```json' или '```'.

[
	{
		"step_id": "идентификатор_шага",
		"formal_statement": "формальная_запись (например, 'math.factorial(8) == 40320')", 
		"verification_status": "OK/FAIL",
		"evidence": [{"tool": "Python", "result": "40320"}],
		"confidence": 1.0
	}
]"""

# ========== РЕАЛИЗАЦИЯ МОДУЛЕЙ ==========
class InputFormalizer:
	def __init__(self):
		self.system_prompt = INPUT_FORMALIZER_SYSTEM_PROMPT
		
	def formalize(self, problem_text: str) -> ProblemObject:
		user_prompt = f"ЗАДАЧА: {problem_text}"
		
		try:
			response_text = generate_with_ai(
				self.system_prompt,
				user_prompt,
				temperature=SETTINGS.models.formalizer_temperature,
				role="formalizer"
			)
			
			# ========== ИЗМЕНЕНИЕ 4: Используем робастную функцию ==========
			json_str = _robust_extract_json(response_text)
			data = json.loads(json_str)
			return ProblemObject(**data)
		except json.JSONDecodeError as e:
			# Эта ошибка (Extra data) теперь должна происходить реже
			print(f"Ошибка формализации (JSONDecodeError): {e}")
			print(f"   Не удалось распарсить: {json_str[:200]}...")
			# Продолжаем с "пустым" объектом, чтобы не падать
			return self._create_fallback_problem(problem_text)
		except Exception as e:
			print(f"Ошибка формализации (Exception): {e}")
			return self._create_fallback_problem(problem_text)

	def _create_fallback_problem(self, problem_text: str) -> ProblemObject:
		"""Вспомогательный метод для создания ProblemObject при ошибке"""
		return ProblemObject(
			id=f"task_{int(time.time())}",
			statement=problem_text,
			entities=[],
			constraints=[],
			goals=[{"type": "value", "target": "unknown"}],
			hypotheses=[],
			irrelevant=[],
			notes="Автоматическая формализация не удалась"
		)
	
	# Метод _extract_json() удален, т.к. используется _robust_extract_json()


class ToTPlanner:
	def __init__(self):
		self.system_prompt = TOT_PLANNER_SYSTEM_PROMPT
		self.pruned_branches = set()
		
	def propose_plans(self, problem: ProblemObject) -> List[Plan]:
		examples = few_shot_retriever.most_relevant(
			problem.statement,
			limit=SETTINGS.thresholds.max_planner_examples,
			topic_hint="geometry"
		)
		user_prompt = (
			f"PROBLEM_OBJECT: {json.dumps(problem.__dict__, ensure_ascii=False, indent=2)}\n"
			f"FEW_SHOT_EXAMPLES: {json.dumps(examples, ensure_ascii=False, indent=2)}"
		)
		
		try:
			response_text = generate_with_ai(
				self.system_prompt,
				user_prompt,
				temperature=SETTINGS.models.planner_temperature,
				role="planner"
			)
			
			# ========== ИЗМЕНЕНИЕ 5: Используем робастную функцию ==========
			json_str = _robust_extract_json(response_text)
			plans_data = json.loads(json_str)
			
			plans = []
			for i, plan_data in enumerate(plans_data):
				if isinstance(plan_data, dict):
					plan_data.setdefault('id', f'plan_{i+1}')
					plan_data.setdefault('summary', 'Не указано')
					plan_data.setdefault('steps', [])
					plan_data.setdefault('estimated_complexity', 'medium')
					plan_data.setdefault('estimated_tooling', [])
					plan_data.setdefault('heuristic_score', 0.5)
					plan_data.setdefault('rationale', 'Не указано')
					plans.append(Plan(**plan_data))
			
			return plans
		except Exception as e:
			print(f"Ошибка планирования: {e}")
			return [Plan(
				id="basic_plan",
				summary="Прямое вычисление через Python",
				steps=[
					"Извлечь числовые значения из задачи",
					"Вычислить промежуточные результаты", 
					"Вычислить финальный результат",
					"Представить ответ"
				],
				estimated_complexity="low",
				estimated_tooling=["Python"],
				heuristic_score=0.7,
				rationale="Прямой вычислительный подход"
			)]
	
	# Метод _extract_json() удален, т.к. используется _robust_extract_json()
	
	def evaluate_plan(self, plan: Plan, feedback: Optional[str] = None) -> float:
		if feedback and "error" in feedback.lower():
			return 0.1
		return plan.heuristic_score
	
	def prune_branch(self, plan_id: str, reason: str):
		self.pruned_branches.add(plan_id)
		print(f"Ветка {plan_id} отсечена: {reason}")
	
	def modify_plan(self, plan: Plan, new_steps: List[str], reason: str) -> Plan:
		"""Модифицирует план с новыми шагами"""
		return Plan(
			id=plan.id + "_modified",
			summary=f"{plan.summary} (модифицирован: {reason})",
			steps=new_steps,
			estimated_complexity=plan.estimated_complexity,
			estimated_tooling=plan.estimated_tooling,
			heuristic_score=plan.heuristic_score * 0.9,
			rationale=f"{plan.rationale}. Модифицирован: {reason}"
		)

class ReActExecutor:
	def __init__(self):
		self.system_prompt = REACT_EXECUTOR_SYSTEM_PROMPT
		
	def execute_step(self, step_description: str, problem: ProblemObject, context: Dict, previous_outputs: List[str], last_error: str = None) -> ThoughtAction:
		outputs_context = "\n".join([f"Вывод шага {i}: {output}" for i, output in enumerate(previous_outputs)])
		error_context = f"\nПОСЛЕДНЯЯ ОШИБКА: {last_error}" if last_error else ""
		trace_id = trace_context.trace_id
		few_shot = few_shot_retriever.most_relevant(
			problem.statement,
			limit=2
		)
		
		entities_summary_list = []
		for entity in problem.entities:
			name = entity.get('name', 'сущность')
			value = entity.get('value', 'N/A')
			unit = entity.get('unit', '')
			entities_summary_list.append(f"- {name}: {value} {unit}".strip())
		entities_summary = "\n".join(entities_summary_list)
		
		goals_summary = "\n".join([
			f"- {goal.get('type', 'тип')}: {goal.get('target', 'цель')}"
			for goal in problem.goals
		])

		user_prompt = f"""
===== ОБЩАЯ ЗАДАЧА (ГЛАВНЫЙ КОНТЕКСТ) =====
ЗАДАЧА: {problem.statement}
КЛЮЧЕВЫЕ СУЩНОСТИ:
{entities_summary}
ЦЕЛИ (GOALS):
{goals_summary}
===========================================

ТЕКУЩИЙ ШАГ ПЛАНА: {step_description}

КОНТЕКСТ ВЫПОЛНЕНИЯ (предыдущие шаги и переменные): 
{json.dumps(context, ensure_ascii=False, indent=2)}

ПРЕДЫДУЩИЕ ВЫВОДЫ (stdout/stderr): 
{outputs_context}
{error_context}

TRACE_ID (обязательно используй в FINISH): {trace_id}

FEW-SHOT ПРИМЕРЫ:
{json.dumps(few_shot, ensure_ascii=False, indent=2)}

ВАЖНЫЕ ПРАВИЛА (ПОМНИ ИХ):
1. Твоя задача - выполнить ТОЛЬКО 'ТЕКУЩИЙ ШАГ ПЛАНА'.
2. ВСЕГДА сверяйся с 'ОБЩАЯ ЗАДАЧА', 'КЛЮЧЕВЫЕ СУЩНОСТИ' и 'ЦЕЛИ'.
3. Если шаг теоретический, просто верни 'Thought:'.
4. Если шаг - финальное вычисление, соответствующее 'ЦЕЛИ', ОБЯЗАТЕЛЬНО вызови 'FINISH' (Правило 5).
5. Используй `ADD_STEP` для детализации или `MODIFY_PLAN` для смены стратегии, если текущий план неоптимален.

Сгенерируй Thought и Action в СТРОГОМ формате.
"""
		
		try:
			response_text = generate_with_ai(
				self.system_prompt,
				user_prompt,
				temperature=SETTINGS.models.executor_temperature,
				role="executor"
			)
			return self._parse_response(response_text)
		except Exception as e:
			print(f"Ошибка выполнения шага: {e}")
			return ThoughtAction(thought=f"Ошибка: {str(e)}", action=None)
	
	def _parse_response(self, response_text: str) -> ThoughtAction:
		thought = ""
		action = None
		
		response_text = response_text.strip()
		
		if "Action:" in response_text:
			parts = response_text.split("Action:", 1)
			thought = parts[0].replace("Thought:", "").strip()
			action_text = parts[1].strip()
			
			# ========== ИЗМЕНЕНИЕ 6: Используем робастную функцию и здесь ==========
			action_text = _robust_extract_json(action_text)
			
			if not action_text:
				print("Ошибка парсинга: 'Action' блок найден, но он пустой или не содержит JSON.")
				return ThoughtAction(thought=thought, action=None)

			try:
				action_text = action_text.replace("None", "null").replace("True", "true").replace("False", "false")
				action = json.loads(action_text)
				is_valid, validation_error = _validate_action_schema(action)
				if not is_valid:
					raise ValueError(f"Схема действия отклонена: {validation_error}")
				if (action.get('type') == 'CALL_PYTHON' and 
					action.get('payload', {}).get('code')):
					code = action['payload']['code']
					# (Логика добавления print() оставлена без изменений)
					if 'print(' not in code and 'print ' not in code:
						lines = code.split('\n')
						last_line = None
						for line in reversed(lines):
							if line.strip() and not line.strip().startswith('#'):
								last_line = line.strip()
								break
						
						if last_line:
							# Проверяем, является ли последняя строка выражением или присваиванием
							try:
								# Если это можно "скомпилировать" как выражение, то его можно и "print"
								compile(last_line, '<string>', 'eval')
								code += f'\nprint({last_line})'
							except SyntaxError:
								# Вероятно, это присваивание, например 'x = 1'
								if '=' in last_line:
									var_name = last_line.split('=')[0].strip()
									if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
										code += f'\nprint("Результат {var_name}:", {var_name})'
						
						action['payload']['code'] = code
			except json.JSONDecodeError as e:
				print(f"Ошибка парсинга JSON в ReAct: {e}")
				print(f"   Не удалось распарсить: {action_text[:200]}...")
				action = self._parse_fallback_action(action_text)
		else:
			thought = response_text.replace("Thought:", "").strip()
			action = None 
		
		return ThoughtAction(thought=thought, action=action)
	
	# _clean_action_text() больше не нужен, его заменяет _robust_extract_json()
	
	def _parse_fallback_action(self, action_text: str) -> Dict:
		# (Этот метод оставлен без изменений, он полезен)
		action_text = action_text.strip().lower()
		
		if "call_python" in action_text or "python" in action_text:
			code = self._extract_python_code_from_text(action_text)
			return {
				"type": "CALL_PYTHON",
				"payload": {"code": code}
			}
		elif "fix_code" in action_text or "исправ" in action_text:
			error_match = re.search(r'error[\s:]*["\']([^"\']+)["\']', action_text, re.IGNORECASE)
			error = error_match.group(1) if error_match else "Неизвестная ошибка"
			
			code = self._extract_python_code_from_text(action_text)
			if not code:
				code = "# Исправленный код\nimport math\nprint('Код исправлен')"
			
			return {
				"type": "FIX_CODE",
				"payload": {
					"error": error,
					"new_code": code
				}
			}
		elif "add_step" in action_text or "добавь шаг" in action_text:
			step_match = re.search(r'step_description[\s:]*["\']([^"\']+)["\']', action_text, re.IGNORECASE)
			step_desc = step_match.group(1) if step_match else "Новый шаг"
			
			return {
				"type": "ADD_STEP",
				"payload": {
					"step_description": step_desc,
					"insert_after": "current"
				}
			}
		elif "finish" in action_text:
			answer_match = re.search(r'answer[\s:]*["\']([^"\']+)["\']', action_text, re.IGNORECASE)
			answer = answer_match.group(1) if answer_match else "Ответ вычислен"
			
			value_match = re.search(r'value[\s:]*([0-9.]+)', action_text, re.IGNORECASE)
			value = float(value_match.group(1)) if value_match else None
			
			return {
				"type": "FINISH",
				"payload": {
					"answer": answer,
					"value": value,
					"units": "unknown"
				}
			}
		
		return None
	
	def _extract_python_code_from_text(self, text: str) -> str:
		# (Этот метод оставлен без изменений)
		if "```python" in text:
			code = text.split("```python")[1].split("```")[0].strip()
		elif "```" in text:
			code = text.split("```")[1].split("```")[0].strip()
		else:
			lines = []
			for line in text.split('\n'):
				clean_line = line.strip()
				if (any(op in clean_line for op in ['=', '+', '-', '*', '/', 'def ', 'import ', 'print', 'from ']) and 
					not clean_line.startswith('{') and 
					not clean_line.startswith('"') and
					'```' not in clean_line):
					lines.append(clean_line)
			code = '\n'.join(lines) if lines else ""
		
		return code

class ToolExecutor:
	def __init__(self):
		self.execution_history = []
		self.sandbox_conf = SETTINGS.sandbox
		
	def execute_python(self, code: str) -> Dict:
		"""Выполняет Python код в безопасном sandbox"""
		with start_span("tool.call_python", {"sandbox": True}) as span:
			try:
				queue: mp.Queue = mp.Queue()
				with tempfile.TemporaryDirectory() as tmp_dir:
					process = mp.Process(
						target=_sandbox_python_worker,
						args=(code, self.sandbox_conf, queue, tmp_dir)
					)
					process.start()
					process.join(SETTINGS.thresholds.max_tool_runtime)
					if process.is_alive():
						process.terminate()
						process.join()
						return {
							"status": "error",
							"error": "Sandbox timeout",
							"stdout": "",
							"stderr": "timeout",
							"result_value": None
						}
					if queue.empty():
						return {
							"status": "error",
							"error": "Sandbox produced no output",
							"stdout": "",
							"stderr": "",
							"result_value": None
						}
					payload = queue.get()
					span.set_attribute("tool.duration_ms", payload.get("duration", 0) * 1000)
					status = "ok" if not payload.get("stderr") else "error"
					output = (payload.get("stdout", "") + payload.get("stderr", "")).strip()
					return {
						"status": status,
						"output": output,
						"stdout": payload.get("stdout", ""),
						"stderr": payload.get("stderr", ""),
						"result_value": payload.get("result_value")
					}
			except Exception as e:
				return {
					"status": "error",
					"error": f"Execution failed: {str(e)}",
					"output": str(e),
					"stdout": "",
					"stderr": str(e),
					"result_value": None
				}
	
	def execute_sympy(self, expression: str) -> Dict:
		"""Выполняет символьные вычисления через SymPy"""
		try:
			result = sympy.sympify(expression)
			simplified = sympy.simplify(result)
			
			output = f"Выражение: {expression}\n"
			output += f"Упрощенное: {simplified}\n"
			if simplified.is_number:
				output += f"Численное значение: {float(simplified)}"
			
			return {
				"status": "ok",
				"output": output,
				"result": str(simplified),
				"latex": sympy.latex(simplified),
				"evaluated": float(simplified) if simplified.is_number else None
			}
		except Exception as e:
			return {
				"status": "error",
				"error": str(e),
				"output": f"Ошибка SymPy: {str(e)}",
				"result": None
			}
	
	def execute_wolfram(self, query: str, api_key: str, timeout_ms: int = 8000) -> Dict:
		"""Выполняет запрос к WolframAlpha API"""
		print(f"--- ВЫЗОВ WOLFRAMALPHA С ЗАПРОСОМ: {query} ---")
		with start_span("tool.call_wolfram", {"query": query}) as span:
			try:
				if not api_key:
					raise ValueError("WolframAlpha API key отсутствует")
				params = {
					"input": query,
					"appid": api_key,
					"output": "json",
					"format": "plaintext"
				}
				response = requests.get(
					"https://api.wolframalpha.com/v2/query",
					params=params,
					timeout=timeout_ms / 1000
				)
				response.raise_for_status()
				data = response.json()
				query_result = data.get("queryresult", {})
				pods = query_result.get("pods", [])
				span.set_attribute("wolfram.success", query_result.get("success"))
				result_value = None
				output_lines = []
				for pod in pods:
					title = pod.get("title", "")
					subpods = pod.get("subpods", [])
					for subpod in subpods:
						text = subpod.get("plaintext", "").strip()
						if text:
							output_lines.append(f"{title}: {text}")
							try:
								result_value = float(text)
							except ValueError:
								continue
				output_text = "\n".join(output_lines) if output_lines else "WolframAlpha не вернул текстовый результат"
				return {
					"status": "ok" if query_result.get("success") else "error",
					"output": output_text,
					"pods": pods,
					"success": query_result.get("success"),
					"result_value": result_value
				}
					
			except Exception as e:
				return {
					"status": "error",
					"error": str(e),
					"output": f"Ошибка WolframAlpha: {str(e)}",
					"result": None,
					"result_value": None
				}

class Verifier:
	def __init__(self):
		self.system_prompt = VERIFIER_SYSTEM_PROMPT
		
	def verify_trace(self, execution_trace: List[ExecutionTrace], problem: ProblemObject) -> List[FormalTrace]:
		examples = few_shot_retriever.most_relevant(
			problem.statement,
			limit=SETTINGS.thresholds.max_verifier_examples
		)
		user_prompt = f"""
ПРОБЛЕМА: {problem.statement}
ТРАССИРОВКА_ВЫПОЛНЕНИЯ: {json.dumps([trace.__dict__ for trace in execution_trace], ensure_ascii=False, indent=2)}
REL_EXAMPLES: {json.dumps(examples, ensure_ascii=False, indent=2)}

Сгенерируй FormalTrace.
"""
		try:
			response_text = generate_with_ai(
				self.system_prompt,
				user_prompt,
				temperature=SETTINGS.models.verifier_temperature,
				role="verifier"
			)
			
			# ========== ИЗМЕНЕНИЕ 7: Используем робастную функцию ==========
			json_str = _robust_extract_json(response_text)
			
			# ========== ИЗМЕНЕНИЕ 11: (ИСПРАВЛЕНИЕ ОШИБКИ ВЕРИФИКАЦИИ) ==========
			# AI иногда возвращает последовательность объектов {...}, {...} 
			# вместо списка [{...}, {...}].
			# Если строка начинается с '{' и заканчивается на '}', 
			# мы оборачиваем ее в скобки, чтобы сделать ее валидным JSON-списком.
			# (Этот патч остается как страховка, даже если промпт исправлен)
			if json_str.startswith('{') and json_str.endswith('}'):
				json_str = f"[{json_str}]"
			# ======================================================================
			
			trace_data = json.loads(json_str)
			
			# ========== ИЗМЕНЕНИЕ 8: РОБАСТНЫЙ ПАРСИНГ ВЕРИФИКАТОРА ==========
			# Это исправит ошибку 'must be a mapping, not str'
			
			if not isinstance(trace_data, list):
				print(f"Ошибка верификации: AI вернула не список, а {type(trace_data)}")
				return []

			traces = []
			for item in trace_data:
				if isinstance(item, dict):
					# Убедимся, что все поля на месте, чтобы FormalTrace() не упал
					item.setdefault('step_id', 'unknown_step')
					item.setdefault('formal_statement', 'N/A')
					item.setdefault('verification_status', 'FAIL')
					item.setdefault('evidence', [])
					item.setdefault('confidence', 0.0)
					traces.append(FormalTrace(**item))
				else:
					# AI вернула список, но в нем не словари
					print(f"Ошибка верификации: Элемент в списке не словарь (dict), а {type(item)}")
					
			return traces
			
		except json.JSONDecodeError as e:
			print(f"Ошибка верификации (JSONDecodeError): {e}")
			print(f"   Не удалось распарсить: {json_str[:200]}...")
			return []
		except TypeError as e:
			# Это та самая ошибка, которую ты видел
			print(f"Ошибка верификации (TypeError): {e}") 
			print(f"   Вероятно, 'trace_data' не является списком словарей. 'item' был: {item}")
			return []
		except Exception as e:
			print(f"Неизвестная ошибка верификации: {e}")
			return []
	
	# Метод _extract_json() удален
	
	def produce_feedback(self, failed_step: FormalTrace) -> str:
		return f"Ошибка на шаге {failed_step.step_id}: {failed_step.formal_statement}"

# ========== ГЛАВНЫЙ ОРКЕСТРАТОР ==========
class MATHAGENTVL:
	def __init__(self):
		self.input_formalizer = InputFormalizer()
		self.tot_planner = ToTPlanner()
		self.react_executor = ReActExecutor()
		self.tool_executor = ToolExecutor()
		self.verifier = Verifier()
		self.schema_validator = schema_validator
		self.judge = lm_judge
		self.finish_retry_limit = SETTINGS.thresholds.finish_retry_limit
		self.metrics = agent_metrics
		
	def solve_problem(self, problem_text: str, wolfram_api_key: str = None) -> Dict:
		run_trace_id = trace_context.new_run()
		with start_span("agent.solve_problem", {"trace_id": run_trace_id}) as run_span:
			start_time = time.time()
			print("=== НАЧАЛО РЕШЕНИЯ ЗАДАЧИ ===")
			print(f"Задача: {problem_text}")
			
			# Шаг 1: Формализация входа
			print("\n1. ФОРМАЛИЗАЦИЯ ВХОДА...")
			problem_object = self.input_formalizer.formalize(problem_text)
			print(f"ProblemObject создан: {problem_object.id}")
			
			# Шаг 2: Генерация планов
			print("\n2. ГЕНЕРАЦИЯ ПЛАНОВ...")
			plans = self.tot_planner.propose_plans(problem_object)
			print(f"Сгенерировано планов: {len(plans)}")
			
			for i, plan in enumerate(plans):
				print(f"  План {i+1}: {plan.summary} (оценка: {plan.heuristic_score})")
			
			plans.sort(key=lambda p: p.heuristic_score, reverse=True)
			
			execution_trace: List[ExecutionTrace] = []
			formal_trace: List[FormalTrace] = []
			final_answer = None
			best_plan = None
			schema_failures = 0
			judge_failures = 0
			
			# Перебираем планы пока не найдем работающий
			for plan_idx, current_plan in enumerate(plans):
				if current_plan.id in self.tot_planner.pruned_branches:
					continue
					
				print(f"\n3. ВЫПОЛНЕНИЕ ПЛАНА {plan_idx + 1}: {current_plan.id}")
				print(f"   Стратегия: {current_plan.summary}")
				
				context = {
					"problem": problem_object.__dict__,
					"plan": current_plan.__dict__,
					"trace_id": run_trace_id,
					"finish_schema": self.schema_validator.schema
				}
				
				previous_outputs: List[str] = []
				execution_trace = []
				plan_success = True
				current_steps = current_plan.steps.copy()
				
				step_idx = 0
				while step_idx < len(current_steps):
					step_desc = current_steps[step_idx]
					print(f"\n   Шаг {step_idx + 1}: {step_desc}")
					
					attempts = 0
					max_attempts = 3
					step_completed = False
					last_error = None
					
					while attempts < max_attempts and not step_completed:
						attempts += 1
						
						if attempts > 1:
							print(f"   Попытка {attempts} из {max_attempts}")
						
						thought_action = self.react_executor.execute_step(
							step_desc, problem_object, context, previous_outputs, last_error
						)
						
						current_step_id = f"step_{step_idx}_attempt_{attempts}"
						
						trace_entry = ExecutionTrace(
							step_id=current_step_id,
							thought=thought_action.thought,
							action=thought_action.action,
							observation=None,
							timestamp=time.time()
						)
						
						print(f"   Мысль: {thought_action.thought}")
						
						if thought_action.action:
							action_type = thought_action.action.get('type', 'UNKNOWN')
							print(f"   Действие: {action_type}")
							
							if action_type == 'FINISH':
								is_valid, errors = self.schema_validator.validate_finish(thought_action.action)
								if not is_valid:
									schema_failures += 1
									last_error = "; ".join(errors)
									print(f"   FINISH отклонен: {last_error}")
									trace_entry.observation = {"status": "finish_rejected", "errors": errors}
									execution_trace.append(trace_entry)
									if schema_failures > self.finish_retry_limit:
										human_queue.enqueue("finish_schema_failure", {
											"trace_id": run_trace_id,
											"errors": errors,
											"step": step_desc
										})
										plan_success = False
										break
									continue
								
								payload = thought_action.action.get('payload', {})
								payload.setdefault("trace_id", run_trace_id)
								judge_report = self.judge.evaluate(problem_object.statement, payload)
								payload["judge"] = judge_report
								
								if judge_report.get("verdict") != "PASS":
									judge_failures += 1
									last_error = f"LM_JUDGE: {judge_report.get('score', 0):.2f}"
									print(f"   Судья отклонил ответ: {judge_report}")
									trace_entry.observation = {"status": "judge_fail", "report": judge_report}
									execution_trace.append(trace_entry)
									if judge_failures > self.finish_retry_limit:
										human_queue.enqueue("judge_failure", {
											"trace_id": run_trace_id,
											"report": judge_report
										})
										plan_success = False
										break
									continue
								
								final_answer = payload
								trace_entry.observation = {"status": "completed", "action": "FINISH"}
								execution_trace.append(trace_entry)
								step_completed = True
								plan_success = True
								break
								
							elif action_type == 'FIX_CODE':
								last_error = thought_action.action.get('payload', {}).get('error', 'Unknown error')
								trace_entry.observation = {"status": "fix_attempt", "error": last_error}
								execution_trace.append(trace_entry)
								
							elif action_type == 'ADD_STEP':
								payload = thought_action.action.get('payload', {})
								new_step = payload.get('step_description', 'Новый неописанный шаг')
								insert_after = payload.get('insert_after', 'current')
								
								if insert_after == 'current':
									current_steps.insert(step_idx + 1, new_step)
								else:
									current_steps.append(new_step)
								
								trace_entry.observation = {"status": "step_added", "new_step": new_step}
								execution_trace.append(trace_entry)
								step_completed = True
								
							elif action_type == 'MODIFY_PLAN':
								payload = thought_action.action.get('payload', {})
								new_steps = payload.get('new_steps', [])
								reason = payload.get('reason', 'N/A')
								
								modified_plan = self.tot_planner.modify_plan(current_plan, new_steps, reason)
								current_steps = modified_plan.steps
								current_plan = modified_plan
								
								trace_entry.observation = {"status": "plan_modified", "new_steps": new_steps}
								execution_trace.append(trace_entry)
								step_completed = True
								step_idx = -1
								break
								
							elif action_type in ['CALL_PYTHON', 'CALL_SYMPY', 'CALL_WOLFRAM']:
								action_result = self._execute_action(thought_action.action, wolfram_api_key)
								trace_entry.observation = action_result
								
								print(f"   Результат: {action_result.get('status', 'unknown')}")
								
								if action_result.get('status') == 'ok':
									output = action_result.get('output', '')
									print(f"   ВЫВОД КОДА/ИНСТРУМЕНТА:\n{output}")
									
									previous_outputs.append(output)
									context[f"step_{step_idx}_output"] = output
									context["last_output"] = output
									
									if action_result.get('result_value') is not None:
										context[f"step_{step_idx}_result"] = action_result['result_value']
										context["last_result"] = action_result['result_value']
									
									step_completed = True
									plan_success = True
								else:
									error_msg = action_result.get('error', 'Неизвестная ошибка')
									print(f"   Ошибка: {error_msg}")
									last_error = error_msg
									previous_outputs.append(f"Ошибка: {error_msg}")
								
								execution_trace.append(trace_entry)
							else:
								trace_entry.observation = {"status": "unknown_action"}
						execution_trace.append(trace_entry)
						step_completed = True
				else:
					print("   Действие: None (Теоретический шаг)")
					if thought_action.thought.lower().startswith("ошибка"):
						last_error = thought_action.thought
					execution_trace.append(trace_entry)
					step_completed = True
					
					if not step_completed:
						print(f"   Шаг не выполнен после {max_attempts} попыток")
						plan_success = False
						break
					
					step_idx += 1
					
					if final_answer:
						break
				
				if plan_success and final_answer:
					best_plan = current_plan
					break
				else:
					self.tot_planner.prune_branch(current_plan.id, "План не выполнен успешно или не привел к FINISH")
					final_answer = None
			
			print("\n4. ВЕРИФИКАЦИЯ...")
			formal_trace = self.verifier.verify_trace(execution_trace, problem_object)
			
			verification_ok = False
			if formal_trace:
				verification_ok = all(step.verification_status == "OK" for step in formal_trace)
			
			confidence_scores = [s.confidence for s in formal_trace] if formal_trace else []
			avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
			
			latency = time.time() - start_time
			success = final_answer is not None and verification_ok
			self.metrics.record_task(success, latency)
			run_span.set_attribute("agent.success", success)
			run_span.set_attribute("agent.latency_ms", latency * 1000)
			
			if not success:
				human_queue.enqueue("unresolved_task", {
					"trace_id": run_trace_id,
					"reason": "final_answer_missing_or_verifier_fail",
					"problem": problem_object.statement
				})
			
			if latency > SETTINGS.thresholds.max_tool_runtime:
				human_queue.enqueue("latency_budget_exceeded", {
					"trace_id": run_trace_id,
					"latency": latency,
					"budget_seconds": SETTINGS.thresholds.max_tool_runtime
				})
			
			result = {
				"problem_id": problem_object.id,
				"trace_id": run_trace_id,
				"problem_object": problem_object.__dict__,
				"selected_plan": best_plan.__dict__ if best_plan else None,
				"execution_trace": [trace.__dict__ for trace in execution_trace],
				"formal_trace": [trace.__dict__ for trace in formal_trace],
				"final_answer": final_answer,
				"verifier_status": "OK" if verification_ok else "FAIL",
				"verifier_report": {
					"passed_steps": len([s for s in formal_trace if s.verification_status == "OK"]) if formal_trace else 0,
					"total_steps": len(formal_trace) if formal_trace else 0,
					"confidence": avg_confidence
				},
				"metrics": {
					"success_rate": self.metrics.success_rate,
					"avg_latency": self.metrics.avg_latency
				}
			}
			
			print("\n=== РЕШЕНИЕ ЗАВЕРШЕНО ===")
			return result
	
	def _execute_action(self, action: Dict, wolfram_api_key: str) -> Dict:
		action_type = action.get('type')
		payload = action.get('payload', {})
		
		if action_type == ActionType.CALL_PYTHON.value:
			return self.tool_executor.execute_python(payload.get('code', ''))
		elif action_type == ActionType.CALL_SYMPY.value:
			return self.tool_executor.execute_sympy(payload.get('expression', ''))
		elif action_type == ActionType.CALL_WOLFRAM.value:
			payload['api_key'] = wolfram_api_key or payload.get('api_key', '')
			return self.tool_executor.execute_wolfram(
				payload.get('query', ''),
				payload['api_key'],
				payload.get('timeout_ms', 8000)
			)
		elif action_type == ActionType.FINISH.value:
			return {"status": "completed", "answer": payload}
		else:
			return {"status": "error", "error": f"Неизвестный тип действия: {action_type}"}
	
# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
def main():
	# ========== ИЗМЕНЕНИЕ 9: Активация DEBUG_MODE ==========
	global DEBUG_MODE
	if "-d" in sys.argv:
		DEBUG_MODE = True
		print("--- DEBUG MODE ENABLED: Logging raw AI output to math_agent.log ---")
		# Очищаем лог-файл при старте
		with open(LOG_FILE, "w") as f:
			f.write(f"--- Log started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

	agent = MATHAGENTVL()
	
	problem = "Найдите все значения параметра a, при каждом из которых имеет ровно два решения система неравенств, 9y^2 >= 16x^2, (x  - 7a + 4x)^2 + y^2 <= 16a^2"
	
	result = agent.solve_problem(
		problem_text=problem,
		wolfram_api_key=os.getenv("WOLFRAM_API_KEY", "ERP3K8L5L3")
	)
	
	with open("math_agent_result.json", "w", encoding="utf-8") as f:
		json.dump(result, f, ensure_ascii=False, indent=2, default=str)
	
	print("Результат сохранен в math_agent_result.json")
	
	if result['final_answer']:
		answer = result['final_answer']
		print(f"\nФИНАЛЬНЫЙ ОТВЕТ: {answer.get('answer', 'N/A')}")
		if answer.get('value') is not None:
			print(f"ЧИСЛОВОЕ ЗНАЧЕНИЕ: {answer['value']}")
	else:
		print("\nФИНАЛЬНЫЙ ОТВET: Не найден")
	
	print(f"СТАТУС ВЕРИФИКАЦИИ: {result['verifier_status']}")

if __name__ == "__main__":
	main()
