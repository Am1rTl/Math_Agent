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
import wolframalpha # <-- ДОБАВЛЕНО

# ========== ИЗМЕНЕНИЕ 1: ГЛОБАЛЬНЫЕ НАСТРОЙКИ ДЛЯ ОТЛАДКИ ==========
DEBUG_MODE = False
LOG_FILE = "math_agent.log"

# ========== КОНФИГУРАЦИЯ OPENROUTER ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
#DEFAULT_MODEL = "microsoft/phi-4-multimodal-instruct"
#DEFAULT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"
#DEFAULT_MODEL = "qwen/qwen3-32b"
DEFAULT_MODEL = "google/gemini-2.0-flash-lite-001"
#DEFAULT_MODEL = "google/gemini-2.5-pro"
#DEFAULT_MODEL = "openai/gpt-5-mini"


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
	"constraints": ["*ТОЛЬКО* извлеченные уравнения/ограничения *КАК ОНИ ЕСТЬ*. Не упрощай и не преобразуй их."],
	"goals": [
		{
			"type": "find_parameter", 
			"target": "a",
			"condition": "система неравенств имеет *ровно два решения* (exactly two solutions)"
		}
	],
	"hypotheses": [],
	"irrelevant": [],
	"notes": "Твоя задача - *только* извлечь данные, не решать задачу."
}"""

# ========== ИЗМЕНЕНИЕ 13: Промпт планировщика (ВОССТАНОВЛЕН) ==========
TOT_PLANNER_SYSTEM_PROMPT = """Ты - ВЫСОКОУРОВНЕВЫЙ стратегический планировщик. 
Твоя задача - предложить 3-5 РАЗНЫХ *стратегий* (планов) для решения задачи. 
Не нужно излишне детализировать шаги; фокусируйся на общей идее.
Отвечай ТОЛЬКО в формате JSON-массива.

ПОМНИ: Исполнитель (ReAct-агент) сможет сам добавлять промежуточные шаги (`ADD_STEP`) или изменять план (`MODIFY_PLAN`), если стратегия окажется неверной. Твоя цель - дать ему хорошие *направления*.

[
	{
		"id": "plan_1",
		"summary": "План А: Геометрический анализ", 
		"steps": ["Шаг 1: Преобразовать неравенства к каноническому виду (эллипс и область).", "Шаг 2: Найти условия касания/пересечения эллипса и границ области.", "Шаг 3: Выразить 'a' через эти условия."],
		"estimated_complexity": "medium",
		"estimated_tooling": ["Python", "SymPy"],
		"heuristic_score": 0.9,
		"rationale": "Прямой аналитический подход."
	},
	{
		"id": "plan_2",
		"summary": "План Б: Запрос к WolframAlpha", 
		"steps": ["Шаг 1: Сформулировать и выполнить запрос к Wolfram 'solve {система} for a such that exactly 2 solutions'.", "Шаг 2: Вернуть результат."],
		"estimated_complexity": "low",
		"estimated_tooling": ["Wolfram"],
		"heuristic_score": 0.8,
		"rationale": "Передать сложный расчет внешнему решателю."
	}
]"""

# ========== ИЗМЕНЕНИЕ 14 и 16: Промпт исполнителя (ReAct) (ВОССТАНОВЛЕН) ==========
REACT_EXECUTOR_SYSTEM_PROMPT = """Ты - тактический исполнитель ReAct.
Твой ответ ДОЛЖЕН СТРОГО следовать формату:
Thought: [Твой анализ и рассуждения]
Action: [ОДИН JSON-объект]

ИЛИ (если действие не нужно):
Thought: [Твой анализ и рассуждения]

ЗАПРЕЩЕНО добавлять что-либо до "Thought:" или после "Action: {json_действие}".

ВАЖНЫЕ ПРАВИЛА ДЕЙСТВИЙ:
1. КОД: Для `CALL_PYTHON` ВСЕГДА генерируй Python код, который ВЫВОДИТ результаты через `print()`. *ОБЯЗАТЕЛЬНО* используй `sympy` для определения переменных (`a, x, y = sympy.symbols('a x y')`) и уравнений (`ineq1 = ...`) в *первый раз*, когда они встречаются.
2. КОНТЕКСТ: Внимательно читай `ОБЩАЯ ЗАДАЧА` и `КЛЮЧЕВЫЕ СУЩНОСТИ`.
3. ОШИБКИ: Если код содержит ошибку, используй `FIX_CODE`.
4. ЗАВЕРШЕНИЕ: Когда задача решена, ВСЕГДА вызывай `FINISH`.
5. ГЛАВНОЕ ПРАВИЛО (FINISH): Используй `FINISH` только тогда, и только тогда, когда ты получил **полный и исчерпывающий ответ на `ОБЩАЯ ЗАДАЧА`**, как она определена в `ЦЕЛИ (GOALS)`.
	- Твой `payload` в `FINISH` должен *в точности* соответствовать `ЦЕЛИ (GOALS)` (например, если цель "найти значение X", твой `payload` должен содержать "X = ...").
	- **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** использовать `FINISH` для завершения *промежуточных* шагов, таких как "упростить уравнение", "найти радиус", "вычислить 5!" или "определить параметры". Это НЕ является решением `ОБЩЕЙ ЗАДАЧИ`.
	- **ИСКЛЮЧЕНИЕ ИЗ ПРАВИЛА 6**: Ты *можешь* использовать `FINISH` без `CALL_PYTHON`, если `ТЕКУЩИЙ ШАГ ПЛАНА` - это *самый последний* арифметический расчет (например, 'Найти A/B'), и у тебя есть все компоненты (A и B) из `КОНТЕКСТ ВЫПОЛНЕНИЯ` для вычисления *финального ответа* на `ОБЩАЯ ЗАДАЧА`.
6. ПРОВЕРЯЙ РАБОТУ: Если 'ТЕКУЩИЙ ШАГ ПЛАНА' требует *промежуточного* вычисления (не финального ответа), твое действие ДОЛЖНО быть `CALL_PYTHON`. Не делай вычисления в 'Thought:'. (См. исключение в Правиле 5).
7. ДЕТАЛИЗАЦИЯ: Если 'ТЕКУЩИЙ ШАГ ПЛАНА' слишком сложный или общий (например, 'Вычислить вероятность'), разбей его, добавив под-шаги с помощью `ADD_STEP`.
8. АДАПТАЦИЯ: Если ты понимаешь, что весь план ошибочен или неэффективен (например, из-за ошибки или нового вывода), используй `MODIFY_PLAN`, чтобы полностью его переписать.

ФОРМАТ ДЕЙСТВИЙ (СТРОГИЙ JSON):
{
	"type": "CALL_PYTHON",
	"payload": {"code": "import sympy\na, x, y = sympy.symbols('a x y')\nineq1 = 9*y**2 >= 16*x**2\nprint(ineq1)"}
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
	"payload": {"answer": "a = 5 или a = -5", "value": ["a=5", "a=-5"], "units": "parameter_values"}
}"""

# ========== ДОБАВЛЕНИЕ: Промпт для нового Агента-Верификатора Финального Ответа ==========
FINAL_VERIFIER_SYSTEM_PROMPT = """Ты - главный верификатор. Твоя задача - оценить, полностью ли решена задача.
Тебе предоставят:
1.  ОБЩАЯ ЗАДАЧА (GOALS): Чего нужно было достичь.
2.  ПРЕДЛОЖЕННЫЙ ОТВЕТ (PROPOSED): Что ReAct-агент считает финальным ответом.
3.  ИСТОРИЯ РЕШЕНИЯ (TRACE): Полный лог мыслей и действий.

Твой ответ ДОЛЖЕН быть СТРОГО в формате JSON:

-   Если ПРЕДЛОЖЕННЫЙ ОТВЕТ *полностью* и *корректно* решает ОБЩУЮ ЗАДАЧУ:
    {
    	"decision": "accept",
    	"summary": "Краткое (1-2 предложения) резюме, подтверждающее решение."
    }

-   Если ПРЕДЛОЖЕННЫЙ ОТВЕТ *неверен*, *неполон* или *не решает* ОБЩУЮ ЗАДАЧУ:
    {
    	"decision": "reject",
    	"reason": "Объяснение, почему ответ не принят (например, 'Найден только радиус, но не финальная вероятность').",
    	"new_steps": [
    		"Шаг 1: [Новый шаг, который нужно добавить для решения]",
    		"Шаг 2: [Еще один шаг, если нужен]"
    	]
    }
"""
# ======================================================================================

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

		# ========== ИСПРАВЛЕНИЕ (TypeError: Mul is not JSON serializable) ==========
		# Эта функция будет конвертировать не-JSON типы (как SymPy) в строки
		# ПРИМЕЧАНИЕ: Это ИСПРАВЛЕНИЕ (1) - определить эту функцию
		def json_safe_default(obj):
			"""Конвертирует несериализуемые объекты (например, SymPy) в строки."""
			try:
				# Попытка стандартной сериализации
				return json.JSONEncoder().default(obj)
			except TypeError:
				# Если не вышло - просто превращаем в строку
				return str(obj)
		# =========================================================================

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
{json.dumps(context, ensure_ascii=False, indent=2, default=json_safe_default)}

ПРЕДЫДУЩИЕ ВЫВОДЫ (stdout/stderr): 
{outputs_context}
{error_context}

ВАЖНЫЕ ПРАВИЛА (ПОМНИ ИХ):
1. Твоя задача - выполнить ТОЛЬКО 'ТЕКУЩИЙ ШАГ ПЛАНА'.
2. ВСЕГДА сверяйся с 'ОБЩАЯ ЗАДАЧA', 'КЛЮЧЕВЫЕ СУЩНОСТИ' и 'ЦЕЛИ'.
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
			
			# ========== УЛУЧШЕНИЕ: Умная обработка SymPy объектов ==========
			# Вместо простой конвертации в str, извлечем больше пользы
			processed_result_value = None
			try:
				if isinstance(result_value, (list, tuple)):
					# Обрабатываем списки/кортежи
					processed_list = []
					for item in result_value:
						if hasattr(item, 'subs') and hasattr(item, 'atoms'): # Простой эвристический тест на "похожесть" на SymPy
							processed_list.append({
								"type": "sympy_object",
								"str": str(item),
								"latex": sympy.latex(item)
							})
						elif not isinstance(item, (str, int, float, bool, dict, type(None))):
							processed_list.append(str(item)) # Обычная конвертация в str
						else:
							processed_list.append(item) # Оставляем как есть
					processed_result_value = processed_list
				
				elif hasattr(result_value, 'subs') and hasattr(result_value, 'atoms'): # Проверка на SymPy
					# Это одиночный SymPy объект
					processed_result_value = {
						"type": "sympy_object",
						"str": str(result_value),
						"latex": sympy.latex(result_value)
					}
				elif not isinstance(result_value, (str, int, float, bool, dict, type(None))):
					# Обычная конвертация в str для других несериализуемых типов
					processed_result_value = str(result_value)
				else:
					# Это уже сериализуемый тип
					processed_result_value = result_value
			
			except Exception as e:
				# На случай, если str() или latex() упадет
				print(f"Warning: Could not convert result_value to string/latex: {e}")
				processed_result_value = f"Non-serializable object: {type(result_value)}"
			# ===================================================================================

			return {
				"status": "ok" if not stderr_text else "error",
				"output": output_text.strip(),
				"stdout": stdout_text,
				"stderr": stderr_text,
				"result_value": processed_result_value # Теперь это значение безопасно и информативно
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
		api_key = "ERP3K8L5L3"
		if not api_key: # Проверка на ключ-заглушку
			print("   Ошибка: API-ключ для WolframAlpha не предоставлен.")
			return {
				"status": "error",
				"error": "API-ключ для WolframAlpha не настроен.",
				"output": "Ошибка: API-ключ для WolframAlpha не настроен.",
				"result": None,
				"result_value": None
			}
			
		try:
			client = wolframalpha.Client(api_key)
			# API wolframalpha использует таймаут в секундах
			scantimeout_sec = timeout_ms / 1000.0
			result = client.query(query, scantimeout=scantimeout_sec)
			
			if result['@success'] == 'true':
				# Пытаемся найти основной результат
				try:
					text_result = next(result.results).text
				except StopIteration:
					# Если .results пуст, ищем поды
					text_result = "Результат не найден в 'results', см. 'pods'."
					for pod in result.pods:
						if pod.get('@title') in ['Result', 'Solution', 'Value']:
							text_result = pod.text
							break

				# Собираем все поды для контекста
				pods_data = []
				if hasattr(result, 'pods'):
					for pod in result.pods:
						pods_data.append({
							"title": pod.get('@title', 'N/A'),
							"text": pod.text
						})
				
				print(f"   Результат Wolfram: {text_result}")
				return {
					"status": "ok",
					"output": text_result,
					"result": text_result,
					"pods": pods_data,
					"success": True,
					"result_value": text_result # Возвращаем текст, т.к. результат может быть нечисловым
				}
			else:
				print("   Запрос к Wolfram не был успешным (success=false).")
				return {
					"status": "error",
					"error": "WolframAlpha query was not successful.",
					"output": "Запрос к WolframAlpha не был успешным.",
					"result": None,
					"result_value": None
				}
				
		except StopIteration:
			print("   Wolfram не вернул 'results'.")
			return {"status": "error", "error": "WolframAlpha returned no results.", "output": "WolframAlpha не вернул 'results'.", "result": None, "result_value": None}
		except Exception as e:
			print(f"   Ошибка при вызове WolframAlpha: {e}")
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
	
# ========== ДОБАВЛЕНИЕ: Класс для нового Агента-Верификатора Финального Ответа ==========
class FinalAnswerVerifier:
	def __init__(self):
		self.system_prompt = FINAL_VERIFIER_SYSTEM_PROMPT

	def verify_answer(self, problem: ProblemObject, trace: List[ExecutionTrace], proposed_answer: Dict) -> Dict:
		"""
		Вызывает LLM для проверки, решает ли предложенный ответ исходную задачу.
		"""
		# Создаем краткую сводку трассировки
		trace_summary_lines = []
		for i, t in enumerate(trace):
			action_str = str(t.action) if t.action else "None"
			obs_str = str(t.observation) if t.observation else "None"
			
			# Обрезаем длинные строки, чтобы не превысить лимит контекста
			action_str = (action_str[:150] + '...') if len(action_str) > 150 else action_str
			obs_str = (obs_str[:150] + '...') if len(obs_str) > 150 else obs_str
			
			trace_summary_lines.append(f"Step {i}: {t.thought[:100]}...\n  Action: {action_str}\n  Obs: {obs_str}")
		
		trace_summary = "\n".join(trace_summary_lines)
		
		user_prompt = f"""
===== ОБЩАЯ ЗАДАЧА (GOALS) =====
{json.dumps(problem.goals, ensure_ascii=False, indent=2)}
=================================

===== ПРЕДЛОЖЕННЫЙ ОТВЕТ (PROPOSED) =====
{json.dumps(proposed_answer, ensure_ascii=False, indent=2)}
=======================================

===== ПОЛНАЯ ИСТОРИЯ РЕШЕНИЯ (TRACE) =====
{trace_summary}
=========================================

Проверь, соответствует ли ПРЕДЛОЖЕННЫЙ ОТВЕТ полностью ОБЩЕЙ ЗАДАЧЕ (GOALS),
основываясь на ИСТОРИИ РЕШЕНИЯ.

-   Если ответ *полный* и *верный*: "decision": "accept".
-   Если ответ *неполный* или *не решает* главную цель: "decision": "reject"
    и ОБЯЗАТЕЛЬНО предложи "new_steps" для завершения.
    
Ответь СТРОГО в формате JSON.
"""
		
		# Вызов AI
		response_text = generate_with_ai(self.system_prompt, user_prompt)
		
		# Парсинг ответа
		try:
			json_str = _robust_extract_json(response_text)
			data = json.loads(json_str)
			
			if not isinstance(data, dict):
				raise json.JSONDecodeError("Ответ не является объектом JSON", response_text, 0)

			# Валидация
			if data.get('decision') == 'reject' and not data.get('new_steps'):
				data['new_steps'] = ["Шаг: Пересмотреть предложенный ответ и сгенерировать новые шаги для достижения цели."]
				data['reason'] = data.get('reason', 'Причина не указана, но шаги отсутствуют.')
			
			if data.get('decision') not in ['accept', 'reject']:
				raise ValueError("Decision 'accept' или 'reject' отсутствует.")

			return data
		except Exception as e:
			print(f"Ошибка парсинга FinalAnswerVerifier: {e}")
			print(f"   Не удалось распарсить: {json_str[:200]}...")
			# Безопасный откат - отклонить и попросить переделать
			return {
				"decision": "reject", 
				"reason": f"Ошибка парсинга верификатора: {e}", 
				"new_steps": ["Шаг: Повторить попытку финальной верификации из-за ошибки парсинга."]
			}
# ======================================================================================

# ========== ГЛАВНЫЙ ОРКЕСТРАТОР ==========
class MATHAGENTVL:
	def __init__(self):
		self.input_formalizer = InputFormalizer()
		self.tot_planner = ToTPlanner()
		self.react_executor = ReActExecutor()
		self.tool_executor = ToolExecutor()
		self.verifier = Verifier()
		self.final_verifier = FinalAnswerVerifier() # <-- ДОБАВЛЕНО
		
	def solve_problem(self, problem_text: str, wolfram_api_key: str = None, progress_callback=None, execution_callback=None) -> Dict:
		def emit(event_type: str, message: str):
			if progress_callback and message:
				event = {
					"type": event_type or "log",
					"message": message,
					"timestamp": time.time()
				}
				try:
					progress_callback(event)
				except Exception:
					pass

		def log(message: str, event_type: str = "log"):
			print(message)
			emit(event_type, message)

		def emit_trace(trace_entry: ExecutionTrace):
			if execution_callback and trace_entry:
				try:
					execution_callback(trace_entry.__dict__)
				except Exception as e:
					print(f"Ошибка в execution_callback: {e}")

		log("=== НАЧАЛО РЕШЕНИЯ ЗАДАЧИ ===", "start")
		log(f"Задача: {problem_text}", "start")
		
		# Шаг 1: Формализация входа
		log("\n1. ФОРМАЛИЗАЦИЯ ВХОДА...", "stage")
		problem_object = self.input_formalizer.formalize(problem_text)
		log(f"ProblemObject создан: {problem_object.id}", "stage")
		
		# Шаг 2: Генерация планов
		log("\n2. ГЕНЕРАЦИЯ ПЛАНОВ...", "stage")
		plans = self.tot_planner.propose_plans(problem_object)
		log(f"Сгенерировано планов: {len(plans)}")
		
		for i, plan in enumerate(plans):
			log(f"  План {i+1}: {plan.summary} (оценка: {plan.heuristic_score})", "plan_preview")
		
		plans.sort(key=lambda p: p.heuristic_score, reverse=True)
		
		execution_trace = []
		formal_trace = []
		final_answer = None
		best_plan = None
		
		# Перебираем планы пока не найдем работающий
		for plan_idx, current_plan in enumerate(plans):
			if current_plan.id in self.tot_planner.pruned_branches:
				continue
				
			log(f"\n3. ВЫПОЛНЕНИЕ ПЛАНА {plan_idx + 1}: {current_plan.id}", "plan_start")
			log(f"   Стратегия: {current_plan.summary}", "plan_strategy")
			
			context = {
				"problem": problem_object.__dict__,
				"plan": current_plan.__dict__
			}
			
			# --- ИЗМЕНЕНИЕ: Сброс трассировки/вывода ДЛЯ КАЖДОГО ПЛАНА ---
			# Это гарантирует, что при провале Плана 1, 
			# верификатор не будет смотреть на старые выводы от Плана 1, когда будет судить План 2
			previous_outputs = []
			execution_trace = [] # Важно: мы верифицируем только трассировку *текущего* плана
			# -----------------------------------------------------------

			plan_success = True
			current_steps = current_plan.steps.copy()
			
			step_idx = 0
			while step_idx < len(current_steps):
				step_desc = current_steps[step_idx]
				log(f"\n   Шаг {step_idx + 1}: {step_desc}", "step_start")
				
				# Внутренний цикл для повторных попыток и исправлений
				attempts = 0
				max_attempts = 3
				step_completed = False
				last_error = None
				
				while attempts < max_attempts and not step_completed:
					attempts += 1
					
					# ========== ИЗМЕНЕНИЕ 10: Печатаем "Попытка" только при ПОВТОРНОЙ попытке ==========
					if attempts > 1:
						log(f"   Попытка {attempts} из {max_attempts}", "retry")
					
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
					
					log(f"   Мысль: {thought_action.thought}", "thought")
					
					if thought_action.action:
						action_type = thought_action.action.get('type', 'UNKNOWN') # Безопасное получение
						log(f"   Действие: {action_type}", "action")
						
						if action_type == 'FINISH':
							# === ИЗМЕНЕНИЕ: ЛОГИКА ПЕРЕХВАТА FINISH ===
							# ReAct-агент *предлагает* закончить. Теперь FinalVerifier *проверяет* это.
							
							proposed_answer = thought_action.action.get('payload', {'answer': 'N/A', 'value': None})
							trace_entry.observation = {"status": "proposing_finish", "answer": proposed_answer}
							execution_trace.append(trace_entry)
							
							log("   Агент предложил финальный ответ. Запуск верификации...", "final_proposal")
							
							# Вызов нового верификатора
							verification_result = self.final_verifier.verify_answer(
								problem_object, 
								execution_trace, 
								proposed_answer
							)
							
							if verification_result.get('decision') == 'accept':
								# ВЕРИФИКАЦИЯ ПРОЙДЕНА
								log(f"   Финальный ответ ПОДТВЕРЖДЕН: {verification_result.get('summary', 'OK')}", "final_accept")
								final_answer = proposed_answer
								step_completed = True
								plan_success = True
								break # Выход из цикла 'while step_idx < len(current_steps)'
							
							else:
								# ВЕРИФИКАЦИЯ НЕ ПРОЙДЕНА
								log(f"   Финальный ответ ОТКЛОНЕН: {verification_result.get('reason', 'N/A')}", "final_reject")
								new_steps = verification_result.get('new_steps', [])
								if new_steps:
									log(f"   Добавление {len(new_steps)} новых шагов в план...", "plan_update")
									current_steps.extend(new_steps) # Добавляем новые шаги в конец плана
								
								# Сбрасываем last_error, т.к. это не ошибка, а отклонение
								last_error = f"Ответ был отклонен: {verification_result.get('reason', 'N/A')}"
								step_completed = True
								# Не выходим из цикла, продолжаем с новыми шагами
							
							# === КОНЕЦ ЛОГИКИ ПЕРЕХВATA ===
							
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
							
							log(f"   Результат: {action_result.get('status', 'unknown')}", "action_result")
							
							if action_result.get('status') == 'ok':
								output = action_result.get('output', '')
								log(f"   ВЫВОД КОДА/ИНСТРУМЕНТА:\n{output}", "tool_output")
								
								previous_outputs.append(output)
								context[f"step_{step_idx}_output"] = output
								context["last_output"] = output
								
								if action_result.get('result_value') is not None:
									context[f"step_{step_idx}_result"] = action_result['result_value']
									context["last_result"] = action_result['result_value']
								
								step_completed = True
								plan_success = True # Успех этого *шага*
							else:
								error_msg = action_result.get('error', 'Неизвестная ошибка')
								log(f"   Ошибка: {error_msg}")
								last_error = error_msg
								previous_outputs.append(f"Ошибка: {error_msg}")
								# plan_success остается True, но step_completed - False,
								# что приведет к повторной попытке
							
							execution_trace.append(trace_entry)
						else:
							trace_entry.observation = {"status": "unknown_action"}
							execution_trace.append(trace_entry)
							step_completed = True
					else:
						log("   Действие: None (Теоретический шаг)")
						execution_trace.append(trace_entry)
						step_completed = True
				
				if not step_completed:
					log(f"   Шаг не выполнен после {max_attempts} попыток")
					plan_success = False # Успех *всего плана* теперь False
					break # Выход из 'while step_idx...'
				
				step_idx += 1
				
				if final_answer: # Если 'FINISH' был вызван и одобрен, выходим
					break
			
			# --- ВОТ ГЛАВНОЕ ИЗМЕНЕНИЕ ---
			# Этот блок теперь выполняется ПОСЛЕ КАЖДОГО плана,
			# если он НЕ завершился успешным и одобренным 'FINISH'.
			
			if plan_success and final_answer:
				# Случай 1: План завершился одобренным FINISH
				best_plan = current_plan
				break # Выходим из 'for plan_idx...'
			
			else:
				# Случай 2: План ПРОВАЛИЛСЯ (plan_success=False)
				# Случай 3: План ВЫПОЛНИЛ все шаги, но не вызвал FINISH (plan_success=True, final_answer=None)
				
				log(f"\n   План {plan_idx + 1} ({current_plan.id}) завершился (без FINISH) или провалился. Запуск верификатора...")

				# У нас есть *хоть что-нибудь* для верификации?
				if execution_trace and previous_outputs:
					# Да, у нас есть выводы. Попытаемся их верифицировать.
					
					# 1. Генерируем "предлагаемый" ответ из последнего вывода
					proposed_answer = self._create_final_answer(previous_outputs, problem_object)
					log(f"   Предполагаемый ответ (на основе последнего вывода): {proposed_answer.get('answer', 'N/A')}")

					# 2. Вызываем верификатор
					try:
						verification_result = self.final_verifier.verify_answer(
							problem_object, 
							execution_trace, # Передаем трассировку *этого* плана
							proposed_answer
						)
						
						# 3. Принимаем решение
						if verification_result.get('decision') == 'accept':
							log("   Верификатор ПОДТВЕРДИЛ сгенерированный ответ.")
							final_answer = proposed_answer
							best_plan = current_plan # Засчитываем этот план как успешный
							break # ВЫХОДИМ из 'for plan_idx...', т.к. нашли ответ
						else:
							reason = verification_result.get('reason', 'Ответ отклонен без причины.')
							log(f"   Верификатор ОТКЛОНИЛ: {reason}")
							# НЕ выходим, 'for plan_idx...' продолжится
					
					except Exception as e:
						log(f"   Ошибка при вызове финального верификатора: {e}")
						# НЕ выходим, 'for plan_idx...' продолжится
				
				else:
					# У нас нет ни трассировки, ни выводов от этого плана
					log("   Нет выводов для верификации.")

				# Если верификатор не одобрил ответ (final_answer все еще None),
				# отсекаем эту ветку и переходим к следующему плану
				if final_answer is None: 
					self.tot_planner.prune_branch(current_plan.id, "План не выполнен, либо верификатор отклонил результат")
					final_answer = None # Явный сброс (на всякий случай)
		
		
		# === ИЗМЕНЕНИЕ: ЛОГИКА ВЫЗОВА ВЕРИФИКАТОРА ПРИ ОТСУТСТВИИ 'FINISH' ===
		# Этот блок теперь срабатывает ТОЛЬКО если 'for' цикл прошел до конца
		# И 'final_answer' НИ РАЗУ не был установлен (ни через FINISH, ни через верификацию).
		if final_answer is None:
			log("\n   Ни один план не дал подтвержденного ответа.")
			
			# Нам больше не нужно здесь вызывать верификатор, т.к. он уже вызывался
			# в конце каждого плана.
			# Просто создаем финальный ответ-заглушку.
			
			if not (execution_trace or previous_outputs):
				# Это редкий случай, когда ВООБЩЕ ничего не было выполнено
				log("   Нет успешных выводов для создания ответа.")
				
			final_answer = {
				"answer": "Не удалось вычислить ответ. Ни один план не дал результата, и ни одна верификация не прошла.",
				"value": None,
				"auto_generated": True,
				"verification_failed": True # Добавляем флаг, что верификация провалена
			}
		# === КОНЕЦ ИЗМЕНЕНИЯ ===
		
		# Шаг 4: Верификация (формальная)
		log("\n4. ВЕРИФИКАЦИЯ...")
		# Важно: execution_trace здесь будет содержать трассировку ПОСЛЕДНЕГО
		# выполненного плана (или того, который дал ответ)
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
		
		log("\n=== РЕШЕНИЕ ЗАВЕРШЕНО ===")
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
	
	problem = "Найдите все значения параметра a, при каждом из которых имеет ровно два решения система неравенств, 9y^2 >= 16x^2, (x  - 7a + 4)^2 + y^2 <= 16a^2"
	
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
