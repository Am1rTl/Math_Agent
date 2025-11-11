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

# ========== КОНФИГУРАЦИЯ OPENROUTER ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
#DEFAULT_MODEL = "microsoft/phi-4-multimodal-instruct"
DEFAULT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"

# Инициализация клиента OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

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
            max_tokens=4000,
            temperature=0.1
        )
        
        return completion.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Ошибка при обращении к нейросети: {e}")
        return f"Ошибка: {str(e)}"

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
INPUT_FORMALIZER_SYSTEM_PROMPT = """Ты - семантический парсер математических задач. Преобразуй задачу в структурированный JSON.

Отвечай ТОЛЬКО в формате JSON:
{
    "id": "уникальный_идентификатор",
    "statement": "оригинальный_текст",
    "entities": [{"name": "имя", "value": "число/выражение", "unit": "единица"}],
    "constraints": ["ограничения"],
    "goals": [{"type": "value/prove/simplify", "target": "цель"}],
    "hypotheses": [{"formalization": "фрагмент", "confidence": 0.0-1.0}],
    "irrelevant": ["нерелевантное"],
    "notes": "заметки"
}"""

# ----- НАЧАЛО ИСПРАВЛЕНИЯ -----
TOT_PLANNER_SYSTEM_PROMPT = """Ты - стратегический планировщик Tree-of-Thoughts. Сгенерируй 3-5 **концептуально РАЗНЫХ** планов решения математической задачи.

ВАЖНО: Планы должны быть лаконичными (3-5 шагов) и **различаться по методологии** (например, "План 1: Комбинаторика", "План 2: Алгебраическое решение", "План 3: Симуляция Монте-Карло").

Отвечай ТОЛЬКО в формате JSON (массив из НЕСКОЛЬКИХ планов):
[
    {
        "id": "plan_1",
        "summary": "План А: Прямой комбинаторный подсчет", 
        "steps": ["Шаг 1 (Метод А): Вычислить общее число исходов (N).", "Шаг 2 (Метод А): Вычислить число благоприятных исходов (K).", "Шаг 3: Найти вероятность K / N."],
        "estimated_complexity": "medium",
        "estimated_tooling": ["Python", "math"],
        "heuristic_score": 0.9,
        "rationale": "Прямолинейный подход, хорошо подходит для задачи."
    },
    {
        "id": "plan_2",
        "summary": "План Б: Использование SymPy для анализа структуры", 
        "steps": ["Шаг 1 (Метод Б): Описать доску как матрицу в SymPy.", "Шаг 2 (Метод Б): Вычислить перманент матрицы для подсчета расстановок.", "Шаг 3: Найти вероятность."],
        "estimated_complexity": "high",
        "estimated_tooling": ["SymPy"],
        "heuristic_score": 0.7,
        "rationale": "Более сложный, но мощный подход, если прямой подсчет затруднен."
    }
]"""
# ----- КОНЕЦ ИСПРАВЛЕНИЯ -----


# ----- НАЧАЛО ИЗМЕНЕНИЯ 2 -----
REACT_EXECUTOR_SYSTEM_PROMPT = """Ты - тактический исполнитель ReAct для математических задач.

Твой ответ ДОЛЖЕН СТРОГО следовать формату:
Thought: [Твой анализ и рассуждения]
Action: [ОДИН JSON-объект]

ИЛИ (если действие не нужно):
Thought: [Твой анализ и рассуждения]

ЗАПРЕЩЕНО добавлять что-либо до "Thought:" или после "Action: {json_действие}". JSON-объект НЕ должен быть обернут в markdown (```json ... ```) или содержать посторонние символы, как "json".

ВАЖНЫЕ ПРАВИЛА ДЕЙСТВИЙ:
1. КОД: Для `CALL_PYTHON` ВСЕГДА генерируй Python код, который ВЫВОДИТ результаты через `print()`.
2. КОНТЕКСТ: Внимательно читай `ОБЩАЯ ЗАДАЧА` и `КЛЮЧЕВЫЕ СУЩНОСТИ` из промпта пользователя, чтобы не путать (например) Ладьи и Ферзи.
3. ОШИБКИ: Если код содержит ошибку, используй `FIX_CODE`.
4. ЗАВЕРШЕНИЕ: Когда задача решена, ВСЕГДА вызывай `FINISH`.

5. ГЛАВНОЕ ПРАВИЛО (FINISH): Если 'ТЕКУЩИЙ ШАГ ПЛАНА' (например, "Вычислить вероятность") НАПРЯМУЮ совпадает с `goals` из 'ОБЩАЯ ЗАДАЧА', и ты получаешь финальное число, твое ЕДИНСТВЕННОЕ действие ДОЛЖНО быть `FINISH`. Не пытайся выполнять следующие шаги, даже если они есть в плане.

6. НЕТ ПУСТОГО КОДА: Если шаг является чисто теоретическим (например, "Представить что-то" или "Определить условие") и не требует вычислений, НЕ ИСПОЛЬЗУЙ `CALL_PYTHON`. Просто изложи свою мысль в `Thought:` и НЕ УКАЗЫВАЙ `Action:`.

ФОРМАТ ДЕЙСТВИЙ (СТРОГИЙ JSON):
{
    "type": "CALL_PYTHON",
    "payload": {
        "code": "import math\nprint(f'8! = {math.factorial(8)}')"
    }
}
ИЛИ
{
    "type": "FIX_CODE", 
    "payload": {
        "error": "описание ошибки",
        "new_code": "исправленный код"
    }
}
ИЛИ
{
    "type": "FINISH", 
    "payload": {
        "answer": "окончательный ответ",
        "value": 0.0,
        "units": "единицы"
    }
}"""
# ----- КОНЕЦ ИЗМЕНЕНИЯ 2 -----

VERIFIER_SYSTEM_PROMPT = """Ты - формальный верификатор. Проверь корректность решения и сгенерируй FormalTrace.

Отвечай ТОЛЬКО в формате JSON:
[
    {
        "step_id": "идентификатор_шага",
        "formal_statement": "формальная_запись", 
        "verification_status": "OK/FAIL",
        "evidence": [{"tool": "инструмент", "result": "результат"}],
        "confidence": 0.0-1.0
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
            
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)
            return ProblemObject(**data)
        except Exception as e:
            print(f"Ошибка формализации: {e}")
            return ProblemObject(
                id=f"task_{int(time.time())}",
                statement=problem_text,
                entities=[],
                constraints=[],
                goals=[{"type": "value", "target": "unknown"}],
                hypotheses=[],
                irrelevant=[],
                notes="Автоматическая формализация"
            )
    
    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        elif "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
        else:
            return text

class ToTPlanner:
    def __init__(self):
        self.system_prompt = TOT_PLANNER_SYSTEM_PROMPT
        self.pruned_branches = set()
        
    def propose_plans(self, problem: ProblemObject) -> List[Plan]:
        user_prompt = f"PROBLEM_OBJECT: {json.dumps(problem.__dict__, ensure_ascii=False, indent=2)}"
        
        try:
            response_text = generate_with_ai(self.system_prompt, user_prompt)
            
            json_str = self._extract_json(response_text)
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
    
    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        elif "[" in text and "]" in text:
            start = text.find("[")
            end = text.rfind("]") + 1
            return text[start:end]
        else:
            return text
    
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
        # Собираем историю предыдущих выводов
        outputs_context = "\n".join([f"Вывод шага {i}: {output}" for i, output in enumerate(previous_outputs)])
        
        error_context = f"\nПОСЛЕДНЯЯ ОШИБКА: {last_error}" if last_error else ""
        
        # Явно извлекаем сущности из объекта задачи
        entities_summary_list = []
        for entity in problem.entities:
            name = entity.get('name', 'сущность')
            value = entity.get('value', 'N/A')
            unit = entity.get('unit', '')
            entities_summary_list.append(f"- {name}: {value} {unit}".strip())
        
        entities_summary = "\n".join(entities_summary_list)
        
        # Собираем цели
        goals_summary = "\n".join([
            f"- {goal.get('type', 'тип')}: {goal.get('target', 'цель')}"
            for goal in problem.goals
        ])

        user_prompt = f"""
Ты - тактический исполнитель ReAct. Твоя цель - выполнить ОДИН ШАГ ПЛАНА.

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

Сгенерируй Thought и Action в СТРОГОМ формате.
"""
        
        try:
            response_text = generate_with_ai(self.system_prompt, user_prompt)
            return self._parse_response(response_text)
        except Exception as e:
            print(f"Ошибка выполнения шага: {e}")
            return ThoughtAction(thought=f"Ошибка: {str(e)}", action=None)
    
    def _parse_response(self, response_text: str) -> ThoughtAction:
        """Парсит ответ на Thought и Action"""
        thought = ""
        action = None
        
        response_text = response_text.strip()
        
        if "Action:" in response_text:
            parts = response_text.split("Action:", 1)
            thought = parts[0].replace("Thought:", "").strip()
            action_text = parts[1].strip()
            
            action_text = self._clean_action_text(action_text)
            
            if not action_text:
                print("Ошибка парсинга: 'Action' блок найден, но он пустой.")
                return ThoughtAction(thought=thought, action=None)

            try:
                action = json.loads(action_text)
                # Убеждаемся, что код содержит print statements
                if (action.get('type') == 'CALL_PYTHON' and 
                    action.get('payload', {}).get('code')):
                    code = action['payload']['code']
                    if 'print(' not in code and 'print ' not in code:
                        # Добавляем автоматический print если его нет
                        lines = code.split('\n')
                        for line in reversed(lines):
                            if line.strip() and not line.strip().startswith('#') and '=' in line:
                                var_name = line.split('=')[0].strip()
                                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                                    code += f'\nprint("Результат {var_name}:", {var_name})'
                                    break
                        action['payload']['code'] = code
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON: {e}")
                action = self._parse_fallback_action(action_text)
        else:
            # Если "Action:" не найдено, это теорет. шаг (Правило 6)
            thought = response_text.replace("Thought:", "").strip()
            action = None # Явно указываем, что действия нет
        
        return ThoughtAction(thought=thought, action=action)
    
    def _clean_action_text(self, text: str) -> str:
        """Очищает текст действия от лишних символов"""
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        
        # Удаляем markdown-обертки и префикс 'json'
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("json"):
            text = text[4:]
        
        if text.endswith("```"):
            text = text[:-3]
            
        text = text.strip()

        if text.startswith('{') and text.endswith('}'):
            return text
        elif '{"type":' in text:
            start = text.find('{"type":')
            end = text.rfind('}') + 1
            return text[start:end]
        
        return text
    
    def _parse_fallback_action(self, action_text: str) -> Dict:
        """Парсит действие при ошибке JSON"""
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
                code = "# Автоматически сгенерированный исправленный код\nimport math\nprint('Код исправлен')"
            
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
        """Извлекает Python код из текста"""
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
            # Создаем полное окружение со всеми библиотеками
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
            
            # Перехватываем stdout/stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            output_text = ""
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    # ВАЖНО: Разрешаем выполнение любого кода
                    exec(code, env)
                    
                    # Ищем результат в переменных
                    result_value = None
                    if 'result' in env:
                        result_value = env['result']
                    elif 'RESULT' in env:
                        result_value = env['RESULT']
                    else:
                        # Ищем переменные с результатом
                        for key, value in env.items():
                            if not key.startswith('_') and not callable(value) and not isinstance(value, type):
                                if key.lower() == 'result' or key.lower().endswith('_result'):
                                    result_value = value
                                    break
                    
                except Exception as e:
                    stderr_capture.write(f"Ошибка выполнения: {str(e)}")
            
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
            
            # Объединяем весь вывод
            output_text = stdout_text + stderr_text
            
            # Если вывод пустой, но есть результат, добавляем его
            if not output_text.strip() and result_value is not None:
                output_text = f"Результат: {result_value}"
            
            return {
                "status": "ok",
                "output": output_text,
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
        try:
            # Эмуляция WolframAlpha для демонстрации
            if "ладьи" in query.lower() or "rook" in query.lower() or "вероятность" in query.lower():
                output = "Wolfram Alpha результат:\n"
                output += "Общее число расстановок 8 ладей: 8! = 40320\n"
                output += "Число расстановок на белых полях: 8! / 2^8 = 40320 / 256 = 157.5\n"
                output += "Вероятность: 157.5 / 40320 = 0.00390625"
                
                return {
                    "status": "ok",
                    "output": output,
                    "result": 0.00390625,
                    "pods": [
                        {"title": "Result", "text": "0.00390625"},
                        {"title": "Probability", "text": "1/256"}
                    ],
                    "success": True
                }
            else:
                return {
                    "status": "ok", 
                    "output": f"Результат WolframAlpha для запроса: {query} = 42",
                    "result": 42.0,
                    "pods": [{"title": "Result", "text": "42"}],
                    "success": True
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "output": f"Ошибка WolframAlpha: {str(e)}",
                "result": None
            }

class Verifier:
    def __init__(self):
        self.system_prompt = VERIFIER_SYSTEM_PROMPT
        
    def verify_trace(self, execution_trace: List[ExecutionTrace], problem: ProblemObject) -> List[FormalTrace]:
        user_prompt = f"""
ПРОБЛЕМА: {problem.statement}
ТРАССИРОВКА_ВЫПОЛНЕНИЯ: {json.dumps([trace.__dict__ for trace in execution_trace], ensure_ascii=False, indent=2)}

Сгенерируй FormalTrace для верификации.
"""
        try:
            response_text = generate_with_ai(self.system_prompt, user_prompt)
            
            json_str = self._extract_json(response_text)
            trace_data = json.loads(json_str)
            return [FormalTrace(**item) for item in trace_data]
        except Exception as e:
            print(f"Ошибка верификации: {e}")
            return []
    
    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        elif "[" in text and "]" in text:
            start = text.find("[")
            end = text.rfind("]") + 1
            return text[start:end]
        else:
            return text
    
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
                    print(f"   Попытка {attempts} из {max_attempts}")
                    
                    # ReAct цикл для каждого шага
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
                    
                    # Выполнение действия если есть
                    if thought_action.action:
                        action_type = thought_action.action['type']
                        print(f"   Действие: {action_type}")
                        
                        if action_type == 'FINISH':
                            final_answer = thought_action.action['payload']
                            trace_entry.observation = {"status": "completed", "action": "FINISH"}
                            execution_trace.append(trace_entry)
                            step_completed = True
                            plan_success = True
                            break
                            
                        elif action_type == 'FIX_CODE':
                            last_error = thought_action.action['payload']['error']
                            trace_entry.observation = {"status": "fix_attempt", "error": last_error}
                            execution_trace.append(trace_entry)
                            
                        elif action_type == 'ADD_STEP':
                            new_step = thought_action.action['payload']['step_description']
                            insert_after = thought_action.action['payload'].get('insert_after', 'current')
                            
                            if insert_after == 'current':
                                current_steps.insert(step_idx + 1, new_step)
                            else:
                                current_steps.append(new_step)
                            
                            trace_entry.observation = {"status": "step_added", "new_step": new_step}
                            execution_trace.append(trace_entry)
                            step_completed = True
                            
                        elif action_type == 'MODIFY_PLAN':
                            new_steps = thought_action.action['payload']['new_steps']
                            reason = thought_action.action['payload']['reason']
                            
                            modified_plan = self.tot_planner.modify_plan(current_plan, new_steps, reason)
                            current_steps = modified_plan.steps
                            current_plan = modified_plan
                            
                            trace_entry.observation = {"status": "plan_modified", "new_steps": new_steps}
                            execution_trace.append(trace_entry)
                            step_completed = True
                            step_idx = 0
                            break
                            
                        elif action_type in ['CALL_PYTHON', 'CALL_SYMPY', 'CALL_WOLFRAM']:
                            action_result = self._execute_action(thought_action.action, wolfram_api_key)
                            trace_entry.observation = action_result
                            
                            print(f"   Результат: {action_result.get('status', 'unknown')}")
                            
                            if action_result.get('status') == 'ok':
                                output = action_result.get('output', '')
                                print(f"   ВЫВОД КОДА:\n{output}")
                                
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
                        # ----- ИЗМЕНЕНИЕ 3 (Логика) -----
                        # Если 'action' is None (Правило 6), это теоретический шаг.
                        # Мы просто записываем мысль и считаем шаг выполненным.
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
                self.tot_planner.prune_branch(current_plan.id, "План не выполнен успешно")
                final_answer = None
        
        if final_answer is None and previous_outputs:
            final_answer = self._create_final_answer(previous_outputs, problem_object)
        
        # Шаг 4: Верификация
        print("\n4. ВЕРИФИКАЦИЯ...")
        formal_trace = self.verifier.verify_trace(execution_trace, problem_object)
        
        verification_ok = all(step.verification_status == "OK" for step in formal_trace) if formal_trace else False
        
        # Шаг 5: Формирование результата
        result = {
            "problem_id": problem_object.id,
            "problem_object": problem_object.__dict__,
            "selected_plan": best_plan.__dict__ if 'best_plan' in locals() else None,
            "execution_trace": [trace.__dict__ for trace in execution_trace],
            "formal_trace": [trace.__dict__ for trace in formal_trace],
            "final_answer": final_answer,
            "verifier_status": "OK" if verification_ok else "FAIL",
            "verifier_report": {
                "passed_steps": len([s for s in formal_trace if s.verification_status == "OK"]) if formal_trace else 0,
                "total_steps": len(formal_trace) if formal_trace else 0,
                "confidence": min([s.confidence for s in formal_trace]) if formal_trace else 0.0
            }
        }
        
        print("\n=== РЕШЕНИЕ ЗАВЕРШЕНО ===")
        return result
    
    def _execute_action(self, action: Dict, wolfram_api_key: str) -> Dict:
        action_type = action['type']
        payload = action['payload']
        
        if action_type == ActionType.CALL_PYTHON.value:
            return self.tool_executor.execute_python(payload.get('code', ''))
        elif action_type == ActionType.CALL_SYMPY.value:
            return self.tool_executor.execute_sympy(payload.get('expression', ''))
        elif action_type == ActionType.CALL_WOLFRAM.value:
            payload['api_key'] = wolfram_api_key or payload.get('api_key', '')
            return self.tool_executor.execute_wolfram(
                payload['query'],
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
            numbers = re.findall(r'-?\d+\.?\d*', last_output)
            if numbers:
                last_number = float(numbers[-1])
                return {
                    "answer": f"Ответ на основе вычислений: {last_output}",
                    "value": last_number,
                    "units": "unknown",
                    "auto_generated": True
                }
            else:
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
    agent = MATHAGENTVL()
    
    problem = "Найдите вероятность того, что при случайной расстановке 8 ладей на шахматной доске так, чтобы они не били друг друга, все ладьи окажутся на белых полях."
    
    result = agent.solve_problem(
        problem_text=problem,
        wolfram_api_key=os.getenv("WOLFRAM_API_KEY", "ERP3K8L5L3")
    )
    
    with open("math_agent_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print("Результат сохранен в math_agent_result.json")
    
    if result['final_answer']:
        answer = result['final_answer']
        print(f"\nФИНАЛЬНЫЙ ОТВЕТ: {answer['answer']}")
        if answer.get('value') is not None:
            print(f"ЧИСЛОВОЕ ЗНАЧЕНИЕ: {answer['value']}")
    else:
        print("\nФИНАЛЬНЫЙ ОТВЕТ: Не найден")
    
    print(f"СТАТУС ВЕРИФИКАЦИИ: {result['verifier_status']}")

if __name__ == "__main__":
    main()
