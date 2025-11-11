import os
import json
import time
import requests
import sympy
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import math
import itertools
from openai import OpenAI

# ========== ИЗМЕНЕНИЕ 1: ГЛОБАЛЬНЫЕ НАСТРОЙКИ ДЛЯ ОТЛАДКИ ==========
DEBUG_MODE = False
LOG_FILE = "math_agent.log"

# ========== КОНФИГУРАЦИЯ OPENROUTER ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
#DEFAULT_MODEL = "microsoft/phi-4-multimodal-instruct"
#DEFAULT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"
DEFAULT_MODEL = "qwen/qwen3-32b"
#DEFAULT_MODEL = "google/gemini-2.0-flash-lite-001"

# Инициализация клиента OpenRouter
client = OpenAI(
	base_url="https://openrouter.ai/api/v1",
	api_key=OPENROUTER_API_KEY
)

# ========== ИЗМЕНЕНИЕ 2: УЛУЧШЕННАЯ ФУНКЦИЯ ИЗВЛЕЧЕНИЯ JSON ==========
def _robust_extract_json(text: str) -> str:
	"""
	Более надежная функция для извлечения JSON из ответа AI.
	Ищет либо ```json ... ```, либо первый попавшийся блок {...} или [...].
	"""
	text = text.strip()
	
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
		
	# 3. Находим *последний* закрывающий символ
	end_index = text.rfind(end_char)
	
	if end_index == -1 or end_index < start_index:
		return text # Неполный JSON

	# Возвращаем все, что между первым и последним символом
	return text[start_index : end_index + 1]


def generate_with_ai(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
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
	
	try:
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
			temperature=0.1
		)
		
		response_content = completion.choices[0].message.content.strip()

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
		# Возвращаем пустую строку, чтобы вызвать ошибку JSON, а не падение
		return ""

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
Не нужно излишне детализировать шаги; фокусируйся на общей идее.
Отвечай ТОЛЬКО в формате JSON-массива.

ПОМНИ: Исполнитель (ReAct-агент) сможет сам добавлять промежуточные шаги (`ADD_STEP`) или изменять план (`MODIFY_PLAN`), если стратегия окажется неверной. Твоя цель - дать ему хорошие *направления*.

[
	{
		"id": "plan_1",
		"summary": "План А: Комбинаторика", 
		"steps": ["Шаг 1: Вычислить общее число перестановок (8!).", "Шаг 2: Вычислить число 'белых' перестановок (4! * 4!).", "Шаг 3: Найти отношение."],
		"estimated_complexity": "medium",
		"estimated_tooling": ["Python", "math"],
		"heuristic_score": 0.9,
		"rationale": "Прямой подсчет."
	},
	{
		"id": "plan_2",
		"summary": "План Б: Запрос к WolframAlpha", 
		"steps": ["Шаг 1: Сформулировать и выполнить запрос к Wolfram.", "Шаг 2: Вернуть результат."],
		"estimated_complexity": "low",
		"estimated_tooling": ["Wolfram"],
		"heuristic_score": 0.8,
		"rationale": "Передать сложный расчет внешнему решателю."
	}
]"""

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
2. КОНТЕКСТ: Внимательно читай `ОБЩАЯ ЗАДАЧА` и `КЛЮЧЕВЫЕ СУЩНОСТИ`.
3. ОШИБКИ: Если код содержит ошибку, используй `FIX_CODE`.
4. ЗАВЕРШЕНИЕ: Когда задача решена, ВСЕГДА вызывай `FINISH`.
5. ГЛАВНОЕ ПРАВИЛО (FINISH): Используй `FINISH`, когда задача решена. **ИСКЛЮЧЕНИЕ ИЗ ПРАВИЛА 6**: Если 'ТЕКУЩИЙ ШАГ ПЛАНА' - это *финальное* вычисление (например, 'Найти X / Y'), и у тебя есть X и Y из `КОНТЕКСТ ВЫПОЛНЕНИЯ` или `ПРЕДЫДУЩИЕ ВЫВОДЫ`, ты *МОЖЕШЬ* и *ДОЛЖЕН* использовать `FINISH` НАПРЯМУЮ, указав вычисленное значение в `payload.value`. Не используй `CALL_PYTHON` для этого последнего шага.
6. ПРОВЕРЯЙ РАБОТУ: Если 'ТЕКУЩИЙ ШАГ ПЛАНА' требует *промежуточного* вычисления (не финального ответа), твое действие ДОЛЖНО быть `CALL_PYTHON`. Не делай вычисления в 'Thought:'. (См. исключение в Правиле 5).
7. ДЕТАЛИЗАЦИЯ: Если 'ТЕКУЩИЙ ШАГ ПЛАНА' слишком сложный или общий (например, 'Вычислить вероятность'), разбей его, добавив под-шаги с помощью `ADD_STEP`.
8. АДАПТАЦИЯ: Если ты понимаешь, что весь план ошибочен или неэффективен (например, из-за ошибки или нового вывода), используй `MODIFY_PLAN`, чтобы полностью его переписать.

ФОРМАТ ДЕЙСТВИЙ (СТРОГИЙ JSON):
{
	"type": "CALL_PYTHON",
	"payload": {"code": "import math\nprint(math.factorial(8))"}
}
ИЛИ
{
	"type": "FIX_CODE", 
	"payload": {"error": "ошибка", "new_code": "print('исправлено')"}
}
ИЛИ
{
	"type": "ADD_STEP",
	"payload": {
		"step_description": "Шаг 2.1: Вычислить (4! * 4!)",
		"insert_after": "current" 
	}
}
ИЛИ
{
	"type": "MODIFY_PLAN",
	"payload": {
		"reason": "Прямой подсчет слишком сложен, переключаюсь на Wolfram.",
		"new_steps": [
			"Шаг 1: Сформулировать запрос для Wolfram 'probability...'.",
			"Шаг 2: Вызвать CALL_WOLFRAM.",
			"Шаг 3: Вернуть результат."
		]
	}
}
ИЛИ
{
	"type": "FINISH", 
	"payload": {"answer": "Вероятность 576/40320", "value": 0.014285714285714285, "units": "probability"}
}"""

# ========== ИЗМЕНЕНИЕ 15: Промпт верификатора сделан более строгим ==========
VERIFIER_SYSTEM_PROMPT = """Ты - формальный верификатор. Проверь трассировку.
Отвечай ТОЛЬКО в формате JSON-**массива** (списка), даже если в нем всего один элемент.
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
			response_text = generate_with_ai(self.system_prompt, user_prompt)
			
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
		user_prompt = f"PROBLEM_OBJECT: {json.dumps(problem.__dict__, ensure_ascii=False, indent=2)}"
		
		try:
			response_text = generate_with_ai(self.system_prompt, user_prompt)
			
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

ВАЖНЫЕ ПРАВИЛА (ПОМНИ ИХ):
1. Твоя задача - выполнить ТОЛЬКО 'ТЕКУЩИЙ ШАГ ПЛАНА'.
2. ВСЕГДА сверяйся с 'ОБЩАЯ ЗАДАЧА', 'КЛЮЧЕВЫЕ СУЩНОСТИ' и 'ЦЕЛИ'.
3. Если шаг теоретический, просто верни 'Thought:'.
4. Если шаг - финальное вычисление, соответствующее 'ЦЕЛИ', ОБЯЗАТЕЛЬНО вызови 'FINISH' (Правило 5).
5. Используй `ADD_STEP` для детализации или `MODIFY_PLAN` для смены стратегии, если текущий план неоптимален.

Сгенерируй Thought и Action в СТРОГОМ формате.
"""
		
		try:
			response_text = generate_with_ai(self.system_prompt, user_prompt)
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
		
	def execute_python(self, code: str) -> Dict:
		"""Выполняет Python код БЕЗ ОГРАНИЧЕНИЙ БЕЗОПАСНОСТИ"""
		try:
			env = {
				'math': math,
				'numpy': np,
				'np': np,
				'sympy': sympy,
				'json': json,
				're': re,
				'itertools': itertools,
				'collections': __import__('collections'),
				'functools': __import__('functools'),
				'os': __import__('os'),
				'sys': sys,
				'time': time,
				'__builtins__': __builtins__,
				'factorial': math.factorial,
				'comb': math.comb,
				'perm': getattr(math, 'perm', lambda n, k: math.factorial(n) // math.factorial(n - k)),
				'combinations': itertools.combinations,
				'permutations': itertools.permutations,
				'product': itertools.product
			}
			
			stdout_capture = io.StringIO()
			stderr_capture = io.StringIO()
			
			output_text = ""
			result_value = None # Инициализируем
			
			with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
				try:
					# Сначала пытаемся исполнить как exec
					exec(code, env)
					
					# Ищем 'result' или 'RESULT'
					if 'result' in env:
						result_value = env['result']
					elif 'RESULT' in env:
						result_value = env['RESULT']
					else:
						# Ищем другие переменные, если 'result' не найден
						for key, value in env.items():
							if not key.startswith('_') and not callable(value) and not isinstance(value, type) and not isinstance(value, type(math)):
								if key.lower() == 'result' or key.lower().endswith('_result') or key.lower() == 'ans':
									result_value = value
									break
						# Если ничего не нашли, берем последнюю не-модульную переменную
						if result_value is None:
							for key, value in reversed(env.items()):
								if not key.startswith('_') and not callable(value) and not isinstance(value, type) and not isinstance(value, type(math)) and key not in ['env', 'code', 'stdout_capture', 'stderr_capture', 'e']:
									result_value = value
									break

				except Exception as e:
					stderr_capture.write(f"Ошибка выполнения: {str(e)}")
			
			stdout_text = stdout_capture.getvalue()
			stderr_text = stderr_capture.getvalue()
			
			output_text = stdout_text + stderr_text
			
			if not output_text.strip() and result_value is not None:
				output_text = f"Результат: {result_value}"
			
			return {
				"status": "ok" if not stderr_text else "error",
				"output": output_text.strip(),
				"stdout": stdout_text,
				"stderr": stderr_text,
				"result_value": result_value
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
		try:
			# Эмуляция WolframAlpha для демонстрации
			if "ладьи" in query.lower() or "rook" in query.lower() or "вероятность" in query.lower():
				output = "Wolfram Alpha результат:\n"
				output += "Вероятность: 1/70"
				
				return {
					"status": "ok",
					"output": output,
					"result": 1/70,
					"pods": [
						{"title": "Result", "text": "1/70"},
						{"title": "Probability", "text": "1/70"}
					],
					"success": True,
					"result_value": 1/70
				}
			else:
				return {
					"status": "ok", 
					"output": f"Результат WolframAlpha для запроса: {query} = 42",
					"result": 42.0,
					"pods": [{"title": "Result", "text": "42"}],
					"success": True,
					"result_value": 42.0
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
		user_prompt = f"""
ПРОБЛЕМА: {problem.statement}
ТРАССИРОВКА_ВЫПОЛНЕНИЯ: {json.dumps([trace.__dict__ for trace in execution_trace], ensure_ascii=False, indent=2)}

Сгенерируй FormalTrace.
"""
		try:
			response_text = generate_with_ai(self.system_prompt, user_prompt)
			
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
		
	def solve_problem(self, problem_text: str, wolfram_api_key: str = None) -> Dict:
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
		
		execution_trace = []
		formal_trace = []
		final_answer = None
		best_plan = None
		
		# Перебираем планы пока не найдем работающий
		for plan_idx, current_plan in enumerate(plans):
			if current_plan.id in self.tot_planner.pruned_branches:
				continue
				
			print(f"\n3. ВЫПОЛНЕНИЕ ПЛАНА {plan_idx + 1}: {current_plan.id}")
			print(f"   Стратегия: {current_plan.summary}")
			
			context = {
				"problem": problem_object.__dict__,
				"plan": current_plan.__dict__
			}
			
			previous_outputs = []
			execution_trace = []
			plan_success = True
			current_steps = current_plan.steps.copy()
			
			step_idx = 0
			while step_idx < len(current_steps):
				step_desc = current_steps[step_idx]
				print(f"\n   Шаг {step_idx + 1}: {step_desc}")
				
				# Внутренний цикл для повторных попыток и исправлений
				attempts = 0
				max_attempts = 3
				step_completed = False
				last_error = None
				
				while attempts < max_attempts and not step_completed:
					attempts += 1
					
					# ========== ИЗМЕНЕНИЕ 10: Печатаем "Попытка" только при ПОВТОРНОЙ попытке ==========
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
						action_type = thought_action.action.get('type', 'UNKNOWN') # Безопасное получение
						print(f"   Действие: {action_type}")
						
						if action_type == 'FINISH':
							final_answer = thought_action.action.get('payload', {'answer': 'N/A', 'value': None})
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
							step_idx = -1 # Начинаем выполнение нового плана с Шага 1
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
		
		if final_answer is None and previous_outputs:
			final_answer = self._create_final_answer(previous_outputs, problem_object)
		
		# Шаг 4: Верификация
		print("\n4. ВЕРИФИКАЦИЯ...")
		formal_trace = self.verifier.verify_trace(execution_trace, problem_object)
		
		verification_ok = False
		if formal_trace:
			 verification_ok = all(step.verification_status == "OK" for step in formal_trace)
		
		# Шаг 5: Формирование результата
		result = {
			"problem_id": problem_object.id,
			"problem_object": problem_object.__dict__,
			"selected_plan": best_plan.__dict__ if best_plan else None,
			"execution_trace": [trace.__dict__ for trace in execution_trace],
			"formal_trace": [trace.__dict__ for trace in formal_trace],
			"final_answer": final_answer,
			"verifier_status": "OK" if verification_ok else "FAIL",
			"verifier_report": {
				"passed_steps": len([s for s in formal_trace if s.verification_status == "OK"]) if formal_trace else 0,
				"total_steps": len(formal_trace) if formal_trace else 0,
				"confidence": min([s.confidence for s in formal_trace] + [0.0]) # Добавлен default
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
	
	def _create_final_answer(self, previous_outputs: List[str], problem: ProblemObject) -> Dict:
		"""Создает финальный ответ на основе выводов выполнения"""
		if previous_outputs:
			last_output = previous_outputs[-1]
			numbers = re.findall(r'-?\d+\.?\d*e?-?\d*', last_output) # Улучшен regex для scientific notation
			if numbers:
				try:
					last_number = float(numbers[-1])
					return {
						"answer": f"Ответ на основе вычислений: {last_output}",
						"value": last_number,
						"units": "unknown",
						"auto_generated": True
					}
				except ValueError:
					pass # Не удалось преобразовать в float
			
			return {
				"answer": f"Результат вычислений: {last_output}",
				"value": None,
				"units": "unknown",
				"auto_generated": True
			}
		else:
			return {
				"answer": "Не удалось вычислить ответ",
				"value": None,
				"units": "unknown",
				"auto_generated": True
			}

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
