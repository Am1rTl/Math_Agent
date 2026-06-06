import os
import json
import uuid
import subprocess  # Для выполнения Python-кода (см. ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ)
import requests    # Для WolframAlpha
import google.generativeai as genai
from typing import Dict, Any, List, Optional

# --- 1. Конфигурация API и Модели ---

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WOLFRAM_API_KEY = os.environ.get("WOLFRAM_API_KEY")  # App ID: ERP3K8L5L3

MODEL_NAME = "gemini-1.5-flash-latest" 
GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 1.0,
    "top_k": 32,
    "max_output_tokens": 8192,
}
# ИСПРАВЛЕНИЕ: Добавлено ''
SAFETY_SETTINGS = 

# --- 2. НОВЫЙ КЛАСС: StatusLogger (Для красивого отображения) ---

class StatusLogger:
    """
    Управляет "красивым" структурированным выводом 
    в консоль, используя отступы и Unicode-символы.
    """
    def __init__(self, indent_char="  "):
        self.indent_char = indent_char
        self.current_indent = 0

    def _log(self, symbol: str, message: str):
        indent = self.indent_char * self.current_indent
        print(f"{indent}{symbol} {message}")

    def start_session(self, title: str):
        print("\n" + "="*20 + f" {title} " + "="*20)
        self.current_indent = 0

    def start_block(self, title: str):
        print("") # Пустая строка для разделения
        self._log("▶️ ", title)
        self.current_indent += 1

    def end_block(self):
        self.current_indent -= 1
        if self.current_indent < 0:
            self.current_indent = 0

    def success(self, message: str):
        self._log("✅", message)

    def fail(self, message: str):
        self._log("❌", message)
    
    def info(self, message: str):
        self._log("ℹ️ ", message)

    def react_thought(self, message: str):
        # Очистка 'Thought:' из сообщения для краткости
        clean_message = message.replace("Thought:", "", 1).strip()
        self._log("💡", f"Thought: {clean_message}")

    def react_action(self, action_name: str, payload: Dict):
        self._log("⚡", f"Action: {action_name}")
        self.log_json(payload, indent_level=self.current_indent + 1)

    def react_observation(self, observation: Dict):
        self._log("🔬", "Observation:")
        self.log_json(observation, indent_level=self.current_indent + 1)
        
    def step(self, message: str):
        self._log("➡️ ", message)

    def backtrack(self, message: str):
        self._log("🔄", f"ОТКАТ (Backtracking): {message}")
        
    def log_json(self, data: Any, indent_level: Optional[int] = None):
        if indent_level is None:
            indent_level = self.current_indent + 1
            
        indent = self.indent_char * indent_level
        try:
            # Пытаемся распарсить, если 'data' - это строка
            if isinstance(data, str):
                data = json.loads(data)

            pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
            for line in pretty_json.split('\n'):
                print(f"{indent}{line}")
        except Exception:
            # Фоллбэк, если 'data' - не JSON
            print(f"{indent}{str(data)}")

    def final_result(self, title: str, data: dict):
        print("\n" + "="*20 + f" {title} " + "="*20)
        self.log_json(data, indent_level=0)
        print("="* (42 + len(title)))

# --- 3. Клиент Gemini ---

class GeminiClient:
    def __init__(self, model_name=MODEL_NAME):
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS
            )
        else:
            self.model = None # Режим мок-ответов

    # ИСПРАВЛЕНИЕ: Добавлен 'List]'
    def generate_json_response(self, system_prompt: str, user_prompt: str, history: Optional]] = None):
        """
        Выполняет вызов LLM с ожиданием JSON-ответа.
        """
        if not self.model:
            # Это условие теперь будет обрабатываться в модулях
            return {"error": "API Key not configured", "raw_text": ""}

        try:
            model_with_system_prompt = genai.GenerativeModel(
                model_name=self.model.model_name,
                generation_config=self.model.generation_config,
                safety_settings=self.model.safety_settings,
                system_instruction=system_prompt
            )
            
            # ИСПРАВЛЕНИЕ: Добавлено ''
            chat_history =
            if history:
                 for item in history:
                    gemini_role = "user" if item["role"] == "user" else "model"
                    chat_history.append({"role": gemini_role, "parts": [item["content"]]})

            chat = model_with_system_prompt.start_chat(history=chat_history)
            response = chat.send_message(user_prompt)
            
            raw_text = response.text.strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            
            raw_text = raw_text.strip()

            return json.loads(raw_text)
        
        except json.JSONDecodeError as e:
            return {"error": "Invalid JSON response from LLM", "raw_text": raw_text, "details": str(e)}
        except Exception as e:
            return {"error": f"API call failed", "details": str(e)}

# --- 4. Модуль 4: Tiered Toolkit (Инструментарий) ---
# (Убраны все 'print' операторы. Логгирование выполняется Оркестратором.)

class TieredToolkit:
    def __init__(self, wolfram_api_key: str):
        self.wolfram_api_key = wolfram_api_key

    def execute_tool(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get("action")
        payload = action.get("payload", {})
        
        try:
            if action_name == "CALL_PYTHON":
                return self.call_python(code=payload.get("code", ""), timeout_ms=payload.get("timeout_ms", 5000))
            elif action_name == "CALL_SYMPY":
                return self.call_python(code=self.wrap_sympy(payload.get("expression", "")), timeout_ms=payload.get("timeout_ms", 3000))
            elif action_name == "CALL_WOLFRAM":
                payload["api_key"] = self.wolfram_api_key
                return self.call_wolframalpha(payload=payload)
            else:
                return {"status": "error", "error": f"Неизвестный action: {action_name}"}
        except Exception as e:
            return {"status": "error", "stderr": f"Критическая ошибка выполнения инструмента: {e}", "return_value": None}

    def wrap_sympy(self, expression: str) -> str:
        return f"""
import sympy as sp
from sympy import Eq, solve, simplify, diff, integrate, symbols
x, y, z = symbols('x y z')
try:
    result = {expression}
    print(f"SymPy Result: {{result}}")
except Exception as e:
    print(f"SymPy Error: {{e}}")
"""

    def call_python(self, code: str, timeout_ms: int) -> Dict[str, Any]:
        # ВАЖНО: Это НЕБЕЗОПАСНАЯ реализация.
        timeout_sec = timeout_ms / 1000.0
        
        try:
            if "return {" in code:
                code_to_run = code.replace("return {", "import json; print(json.dumps({", 1)
                # ИСПРАВЛЕНИЕ: Исправлена ошибка 'TypeError' при конкатенации list и str
                code_to_run = code_to_run.rsplit('}', 1) + '}))'
            else:
                code_to_run = code

            completed_process = subprocess.run(
                ['python', '-c', code_to_run],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=True
            )
            
            stdout = completed_process.stdout
            stderr = completed_process.stderr
            
            last_line = stdout.strip().split('\n')[-1]
            try:
                return_value = json.loads(last_line)
            except json.JSONDecodeError:
                return_value = {"ok": True, "result": stdout}

            return {"status": "ok", "stdout": stdout, "stderr": stderr, "return_value": return_value}

        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timeout", "return_value": None}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "stdout": e.stdout, "stderr": e.stderr, "return_value": None}
        except Exception as e:
            return {"status": "error", "stderr": f"Python sandbox error: {e}", "return_value": None}


    def call_wolframalpha(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query")
        api_key = payload.get("api_key")
        timeout_sec = payload.get("timeout_ms", 8000) / 1000.0
        
        if not api_key:
            return {"status": "error", "error": "WOLFRAM_API_KEY не предоставлен."}
        
        base_url = "http://api.wolframalpha.com/v2/query"
        params = { "input": query, "appid": api_key, "output": "JSON", "format": "plaintext", "scanner": "Numeric" }

        try:
            response = requests.get(base_url, params=params, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()
            
            if data.get("queryresult", {}).get("success"):
                pods = data.get("queryresult", {}).get("pods",)
                primary_pod = next((p for p in pods if p.get("primary")), None)
                
                if primary_pod and primary_pod.get("subpods"):
                    # ИСПРАВЛЕНИЕ: 'subpods' - это список, нужен 
                    result = primary_pod["subpods"].get("plaintext")
                    return {"status": "ok", "return_value": {"result": result, "raw_pods": pods}}
                else:
                    return {"status": "error", "error": "Ответ WA получен, но не найден Primary Pod."}
            else:
                return {"status": "error", "error": f"WA query failed: {data.get('queryresult', {}).get('error')}"}

        except requests.Timeout:
            return {"status": "error", "error": "WolframAlpha API timeout"}
        except requests.RequestException as e:
            return {"status": "error", "error": f"WolframAlpha API request failed: {e}"}


# --- 5. Когнитивные Модули (Оболочки для Gemini API) ---
# (Убраны все 'print' операторы.)

class Module1_InputFormalizer:
    SYSTEM_PROMPT = """
Вы — 'Input Formalizer', специализированный модуль семантического парсинга.
Ваша единственная задача — преобразовать предоставленный пользователем неструктурированный запрос (текст или описание изображения) в СТРОГИЙ JSON-объект 'ProblemObject'.
Вы НЕ решаете задачу. Вы НЕ пишете код. Вы НЕ выполняете вычисления.
Вы только АНАЛИЗИРУЕТЕ и ФОРМАЛИЗУЕТЕ.
Ваш вывод должен быть ИСКЛЮЧИТЕЛЬНО валидным JSON-объектом, соответствующим схеме 'ProblemObject'.

ПРАВИЛА:
1.  **Извлечение Сущностей (Entities)**: Идентифицируйте все числовые значения, переменные и их единицы (units).
2.  **Извлечение Ограничений (Constraints)**: Формализуйте отношения между сущностями (например, 'total = cost * items').
3.  **Извлечение Целей (Goals)**: Определите, что требуется найти, доказать или упростить.
4.  **Обработка Неоднозначности (Hypotheses)**: При неоднозначности, сгенерируйте 1-3 объекта 'hypotheses' с 'confidence' и 'assumption'.
5.  **Изоляция Нерелевантного (Irrelevant)**: Поместите в 'irrelevant' любую информацию, не влияющую на математическую суть.
6.  **Schema (Обязательна):**
    {
      "id": "string",
      "statement": "string",
      "entities": [{"name": "string", "value": "number|string", "unit": "string|null"}],
      "constraints": ["string"],
      "goals": [{"type": "value|prove", "target": "string", "unit": "string|null"}],
      "hypotheses":, 
      "irrelevant":
    }
"""
    def __init__(self, client: GeminiClient):
        self.client = client
        self.use_mock = not self.client.model

    def run(self, user_query: str) -> Dict[str, Any]:
        if self.use_mock:
            return self.get_mock_response(user_query)
        
        user_prompt = f"Формализуй следующий запрос:\n\n{user_query}"
        response_json = self.client.generate_json_response(self.SYSTEM_PROMPT, user_prompt)
        
        if "error" not in response_json:
            response_json["id"] = f"problem-{uuid.uuid4()}"
        
        return response_json

    def get_mock_response(self, user_query: str) -> Dict[str, Any]:
        return {
            "id": "pie-task-mock-001", "statement": user_query,
            "entities": [
                {"name": "apple_pie_count", "value": 15, "unit": "pies"},
                {"name": "apple_pie_cost", "value": 12.75, "unit": "dollars_per_pie"},
                {"name": "cherry_pie_count", "value": 7, "unit": "pies"},
                {"name": "cherry_pie_cost", "value": 14.50, "unit": "dollars_per_pie"}
            ],
            "constraints": [
                "apple_revenue = apple_pie_count * apple_pie_cost",
                "cherry_revenue = cherry_pie_count * cherry_pie_cost",
                "total_revenue = apple_revenue + cherry_revenue"
            ],
            "goals": [{"type": "value", "target": "total_revenue", "unit": "dollars"}],
            "hypotheses":, "irrelevant":
        }


class Module2_ToTPlanner:
    SYSTEM_PROMPT = """
Вы — 'ToT Planner', стратегический планировщик для математического агента.
Ваша задача — принять 'ProblemObject' и сгенерировать дерево из 2-5 НЕЗАВИСИМЫХ, высокоуровневых планов (Plans) для его решения.
Вывод - это JSON-объект: { "plan_tree": [... ] }

ПРАВИЛА:
1.  **Генерация Планов (Propose)**:
    *   Создайте 2-5 различных стратегических подходов. (например, План А: "Использовать символьное решение SymPy", План Б: "Использовать пошаговый численный расчет PAL", План В: "Использовать WolframAlpha для прямого ответа").
    *   Каждый план — это УПОРЯДОЧНЕННЫЙ список высокоуровневых шагов (Steps). Шаг — это ЧЕТКАЯ цель (например, "Решить систему уравнений для 'x' и 'y'"), а НЕ вызов инструмента.
2.  **Оценка Планов (Evaluate)**:
    *   Для каждого плана вы ДОЛЖНЫ предоставить 'Plan.metadata'.
    *   'estimated_complexity': Оцените (low/medium/high).
    *   'estimated_tooling': Предскажите, какие инструменты (PAL, SymPy, Wolfram) ПОТРЕБУЮТСЯ.
    *   'heuristic_score': Ваша лучшая оценка (0.0-1.0), насколько этот план ВЕРИФИЦИРУЕМ (предпочитайте "белые ящики" - PAL/SymPy) и насколько он ВЕРОЯТНО приведет к успеху.
    *   'rationale': Краткое обоснование.
"""
    def __init__(self, client: GeminiClient):
        self.client = client
        self.use_mock = not self.client.model
        # ИСПРАВЛЕНИЕ: Это была строка с ошибкой.
        self.plan_tree: List] =
        
    # ИСПРАВЛЕНИЕ: Исправлен тип возврата
    def propose_plans(self, problem_object: Dict[str, Any]) -> List]:
        if self.use_mock:
            self.plan_tree = self.get_mock_response(problem_object).get("plan_tree",)
            return self.plan_tree
            
        user_prompt = f"Сгенерируй дерево планов для следующего ProblemObject:\n\n{json.dumps(problem_object, indent=2)}"
        response_json = self.client.generate_json_response(self.SYSTEM_PROMPT, user_prompt)
        
        self.plan_tree = response_json.get("plan_tree",)
        return self.plan_tree

    # ИСПРАВЛЕНИЕ: Исправлен тип возврата
    def get_next_best_plan(self) -> Optional]:
        """
        Возвращает лучший план, который еще не PENDING или EXECUTING.
        """
        # ИСПРАВЛЕНИЕ: Добавлено ''
        pending_plans =
        if not pending_plans:
            return None
        
        pending_plans.sort(key=lambda p: p.get("metadata", {}).get("heuristic_score", 0), reverse=True)
        
        plan = pending_plans
        plan["metadata"]["status"] = "EXECUTING"
        return plan
        
    def prune_branch(self, plan_id: str, reason: str):
        for plan in self.plan_tree:
            if plan.get("plan_id") == plan_id:
                plan["metadata"]["status"] = "PRUNED"
                plan["metadata"]["rationale"] += f""
                break

    def get_mock_response(self, problem_object: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "plan_tree": [
                {
                    "plan_id": "plan-pal-001", "summary": "План А: Пошаговый расчет через PAL-код",
                    "steps": [
                        {"step_id": "step-a1", "goal": "Рассчитать выручку от яблочных пирогов (apple_revenue)"},
                        {"step_id": "step-a2", "goal": "Рассчитать выручку от вишневых пирогов (cherry_revenue)"},
                        {"step_id": "step-a3", "goal": "Сложить apple_revenue и cherry_revenue для total_revenue"}
                    ],
                    "metadata": {
                        "status": "PENDING", "estimated_complexity": "low",
                        "estimated_tooling": ["PAL", "Python"], "heuristic_score": 0.95,
                        "rationale": "Высокая верифицируемость (белый ящик), низкая сложность."
                    }
                },
                {
                    "plan_id": "plan-wolfram-002", "summary": "План Б: Прямой запрос к WolframAlpha",
                    # ИСПРАВЛЕНИЕ: Заполнены 'steps' и 'estimated_tooling'
                    "steps":,
                    "metadata": {
                        "status": "PENDING", "estimated_complexity": "low",
                        "estimated_tooling":, "heuristic_score": 0.70,
                        "rationale": "Быстро, но 'черный ящик', низкая верифицируемость."
                    }
                }
            ]
        }


class Module3_ReActExecutor:
    #... (SYSTEM_PROMPT не изменился)
    SYSTEM_PROMPT = """
Вы — 'ReAct Executor', тактический исполнитель.
Ваша задача — достичь ОДНОЙ КОНКРЕТНОЙ 'current_goal' (цели шага), предоставленной вам.
Вы ОБЯЗАНЫ работать в строгом цикле "Thought -> Action -> Observation".
Ваш ответ ВСЕГДА должен быть JSON-объектом, содержащим "thought" (строка) и "action" (JSON-объект).

ПРАВИЛА ЦИКЛА:
1.  USER предоставит вам 'problem_object', 'current_goal' и 'history' (предыдущие шаги) и (опционально) 'observation'.
2.  ASSISTANT (Вы): Вы генериете JSON: { "thought": "...", "action": {... } }.
3.  Когда цель достигнута, вы ОБЯЗАНЫ использовать 'action': { "action": "FINISH", "payload": { "status": "success", "result":... } }.
4.  Если вы не можете достичь цели, используйте: { "action": "FINISH", "payload": { "status": "failure", "error_info": "..." } }.

ПРАВИЛА ИНСТРУМЕНТОВ (ACTIONS):
*   Доступные action: "CALL_PYTHON", "CALL_SYMPY", "CALL_WOLFRAM", "FINISH".
*   `CALL_PYTHON`: Для арифметики, логики или выполнения PAL-кода.
*   `CALL_SYMPY`: Для символьных операций (solve, simplify, diff, integrate).
*   `CALL_WOLFRAM`: ИСПОЛЬЗУЙТЕ ТОЛЬКО для: 1. Запросов, требующих внешних знаний. 2. Сложных вычислений "черного ящика".
*   **ЗАПРЕТ**: Вы НИКОГДА не должны включать API ключи (например, 'api_key') в 'action'. Оркестратор добавит их сам.

ПРАВИЛА PAL-КОДА (для `CALL_PYTHON`):
*   Если вы пишете код "в середине мысли" (Mid-thought PAL), он ДОЛЖЕН следовать шаблону `def solve_step_{id}(...) -> dict:`.
*   Код должен включать `assert`'ы для предусловий и постусловий.
*   Код должен возвращать `dict` (например, `{"ok": True, "result":...}`).
*   Сразу после `thought` с кодом вы должны сгенерировать `action: CALL_PYTHON` с этим кодом.
"""
    def __init__(self, client: GeminiClient):
        self.client = client
        self.use_mock = not self.client.model
        # ИСПРАВЛЕНИЕ: Исправлен тип
        self.history: List] =

    # ИСПРАВЛЕНИЕ: Исправлен тип возврата
    def run_step(self, problem_object: Dict[str, Any], step: Dict[str, Any], history_from_orchestrator: Dict) -> List]:
        """
        Выполняет полный цикл ReAct для одного шага.
        В этом прототипе мы ИСПОЛЬЗУЕМ MOCK.
        """
        if self.use_mock:
            return self.get_mock_step_trace(step['step_id'])

        # --- Логика для реального ReAct-цикла (в данном прототипе не используется) ---
        # ИСПРАВЛЕНИЕ: Добавлено ''
        prompt_history =
        user_prompt_parts = [
            f"ProblemObject: {json.dumps(problem_object)}",
            f"History (предыдущие шаги): {json.dumps(history_from_orchestrator)}",
            f"Current Goal: {step['goal']}"
        ]
        # ИСПРАВЛЕНИЕ: Добавлено ''
        step_execution_trace =
        
        for _ in range(10): 
            user_prompt = "\n".join(user_prompt_parts)
            
            # 1. Вызов LLM (Thought + Action)
            response_json = self.client.generate_json_response(self.SYSTEM_PROMPT, user_prompt, history=prompt_history)
            
            if "error" in response_json:
                break
            
            action = response_json.get("action", {})
            step_execution_trace.append({"role": "assistant", "content": json.dumps(response_json)})
            prompt_history.append({"role": "model", "content": json.dumps(response_json)})

            if action.get("action") == "FINISH":
                break 

            # 2. Вызов Инструмента (здесь нужен yield или callback)
            # observation = toolkit.execute_tool(action)
            observation = {"status": "error", "error": "Live ReAct loop not implemented in this prototype"}
            
            step_execution_trace.append({"role": "observation", "content": json.dumps(observation)})
            user_prompt_parts.append(f"Observation: {json.dumps(observation)}")
            
        return step_execution_trace

    def get_mock_step_trace(self, step_id: str) -> List]:
        """
        Возвращает жестко закодированную трассировку ReAct для задачи о пирогах.
        """
        
        # ИСПРАВЛЕНИЕ: Полностью определены все PAL-коды
        pal_code_a1 = """
def solve_step_a1(problem: dict, history: dict) -> dict:
    \"\"\"
    intent: Рассчитать выручку от 15 яблочных пирогов по 12.75
    \"\"\"
    try:
        count = 15
        cost = 12.75
        assert count > 0 and cost > 0, "Input values must be positive"
        revenue = count * cost
        assert revenue == 191.25, f"Expected 191.25, got {revenue}"
        return {"ok": True, "result": revenue, "unit": "dollars"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
"""
        
        pal_code_a2 = """
def solve_step_a2(problem: dict, history: dict) -> dict:
    \"\"\"
    intent: Рассчитать выручку от 7 вишневых пирогов по 14.50
    \"\"\"
    try:
        count = 7
        cost = 14.50
        assert count > 0 and cost > 0
        revenue = count * cost
        assert revenue == 101.50, f"Expected 101.50, got {revenue}"
        return {"ok": True, "result": revenue, "unit": "dollars"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
"""

        pal_code_a3 = """
def solve_step_a3(problem: dict, history: dict) -> dict:
    \"\"\"
    intent: Суммировать выручку от яблочных (191.25) и вишневых (101.50) пирогов
    \"\"\"
    try:
        # В реальном коде history - это dict.
        # apple_revenue = history['step-a1']['result']
        apple_revenue = 191.25 
        cherry_revenue = 101.50
        
        assert apple_revenue > 0
        assert cherry_revenue > 0
        
        total_revenue = apple_revenue + cherry_revenue
        
        assert total_revenue == 292.75, f"Expected 292.75, got {total_revenue}"
        return {"ok": True, "result": total_revenue, "unit": "dollars"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
"""
        
        # ИСПРАВЛЕНИЕ: Полностью заполнен словарь traces
        traces = {
            "step-a1":,
            "step-a2":,
            "step-a3":
        }
        return traces.get(step_id,) # Возвращаем трассировку для конкретного шага


class Module5_Verifier:
    SYSTEM_PROMPT = """
Вы — 'Verifier / Critic', модуль формальной верификации.
Ваша задача — получить 'Execution_Trace' (трассировку выполнения от 'ReAct Executor') и пошагово проверить ее на корректность, сгенерировав 'FormalTrace'.
Вы НЕ доверяете 'Executor'. Вы ПРОВЕРЯЕТЕ его работу.
Ваш вывод - это JSON-объект: { "verification_status": "OK|FAIL", "formal_trace_entry": {... }, "feedback_message": "..." }
(Далее правила верификации...)
"""
    def __init__(self, client: GeminiClient, toolkit: TieredToolkit):
        self.client = client
        self.toolkit = toolkit
        self.use_mock = not self.client.model

    def verify_step(self, step_id: str, step_trace: List) -> (str, Dict, str):
        # --- MOCK LOGIC (для прототипа) ---
        # В реальной системе мы бы вызвали LLM (self.client.generate_json_response)
        # Но для прототипа мы реализуем логику верификации прямо здесь.
        
        action_item = None
        observation_item = None
        
        for item in step_trace:
            if item["role"] == "assistant":
                content = json.loads(item["content"])
                action = content.get("action", {})
                if action.get("action") not in:
                    action_item = action
            elif item["role"] == "observation":
                observation_item = json.loads(item["content"])
            
            if action_item and observation_item:
                break
        
        if not action_item:
            return "OK", {"step_id": step_id, "verification_method": "No Action"}, ""

        action_type = action_item.get("action")
        
        # ИСПРАВЛЕНИЕ: Добавлено ''
        formal_entry = {
            "step_id": step_id,
            "formal_statement": f"Execute {action_type} for {step_id}",
            "confidence": 1.0, "evidence":
        }
        
        if action_type == "CALL_PYTHON":
            formal_entry["verification_method"] = "Re-run PAL code in clean sandbox"
            
            code = action_item.get("payload", {}).get("code")
            original_observation = observation_item
            
            if not code or not original_observation:
                 return "FAIL", formal_entry, "Трассировка 'CALL_PYTHON' неполная."

            verification_result = self.toolkit.execute_tool(action_item)
            
            original_rv = original_observation.get("return_value")
            verified_rv = verification_result.get("return_value")
            
            if (verification_result.get("status") == "ok" and 
                original_rv and verified_rv and
                original_rv.get("ok") and verified_rv.get("ok") and
                original_rv.get("result") == verified_rv.get("result")):
                
                formal_entry["verification_status"] = "OK"
                return "OK", formal_entry, "Шаг верифицирован (PAL Re-run)."
            else:
                formal_entry["verification_status"] = "FAIL"
                return "FAIL", formal_entry, f"Результат re-run не совпал. Ожидалось: {original_rv}, получено: {verified_rv}"

        elif action_type == "CALL_WOLFRAM":
            formal_entry["verification_method"] = "Trusted Black Box (WolframAlpha)"
            formal_entry["verification_status"] = "TRUSTED_BLACK_BOX"
            formal_entry["confidence"] = 0.5
            return "OK", formal_entry, "Черный ящик, принято."

        else:
            formal_entry["verification_method"] = f"Skipped ({action_type})"
            formal_entry["verification_status"] = "SKIPPED"
            return "OK", formal_entry, "Метод верификации не реализован."


# --- 6. Модуль 6: Orchestrator (Главный цикл с логгированием) ---

class Orchestrator:
    def __init__(self, logger: StatusLogger):
        self.logger = logger
        
        # Инициализация всех модулей
        self.gemini_client = GeminiClient()
        self.toolkit = TieredToolkit(wolfram_api_key=WOLFRAM_API_KEY)
        self.formalizer = Module1_InputFormalizer(self.gemini_client)
        self.planner = Module2_ToTPlanner(self.gemini_client)
        self.executor_mock = Module3_ReActExecutor(self.gemini_client) # Используем мок-трассировки
        self.verifier = Module5_Verifier(self.gemini_client, self.toolkit)
        
    def get_final_answer_from_history(self, history: Dict) -> Dict:
        if not history:
            return {"value": None, "unit": None, "confidence": 0.0}
            
        last_step_id = sorted(history.keys())[-1]
        last_result = history[last_step_id]
        
        return {
            "value": last_result.get("result"), # Изменил 'value' на 'result'
            "unit": last_result.get("unit"),
            "confidence": 1.0
        }

    def main_orchestration_loop(self, user_query: str) -> Dict:
        """
        Главный цикл, использующий StatusLogger для отображения этапов.
        """
        self.logger.start_session(f"Запуск MATH-AGENT-VL: \"{user_query[:50]}...\"")
        
        # 1. Формализация Ввода
        self.logger.start_block("1. Формализация Ввода (Модуль 1)")
        problem_object = self.formalizer.run(user_query)
        if "error" in problem_object:
            self.logger.fail(f"Ошибка формализации: {problem_object.get('details')}")
            self.logger.end_block()
            return {"status": "FAIL", "error": "Ошибка формализации", "details": problem_object}
        self.logger.success("Ввод успешно формализован.")
        self.logger.log_json(problem_object)
        self.logger.end_block()
        
        # 2. Генерация Планов (ToT)
        self.logger.start_block("2. Планирование (ToT) (Модуль 2)")
        plan_tree = self.planner.propose_plans(problem_object)
        if not plan_tree:
             self.logger.fail("Планировщик (ToT) не сгенерировал планов.")
             self.logger.end_block()
             return {"status": "FAIL", "error": "Планировщик (ToT) не сгенерировал планов."}
        self.logger.success(f"Сгенерировано {len(plan_tree)} планов.")
        for p in plan_tree:
             self.logger.info(f"План '{p['plan_id']}' (Score: {p['metadata']['heuristic_score']}): {p['summary']}")
        self.logger.end_block()

        final_result = None
        full_execution_trace = {}
        full_formal_trace = {}
        
        # 3. Цикл по Планам (Backtracking Loop)
        while plan := self.planner.get_next_best_plan():
            
            plan_id = plan["plan_id"]
            self.logger.start_block(f"3. Выполнение Плана: {plan_id} ({plan['summary']})")
            
            # ИСПРАВЛЕНИЕ: Добавлено ''
            plan_execution_trace =
            plan_formal_trace =
            plan_failed = False
            current_state_history = {} 

            # 4. Цикл по Шагам Плана
            for step in plan["steps"]:
                step_id = step["step_id"]
                self.logger.start_block(f"Шаг: {step_id} ('{step['goal']}')")

                # 5. Выполнение Шага (ReAct)
                step_trace = self.executor_mock.get_mock_step_trace(step_id)
                if not step_trace:
                    self.logger.fail(f"Критическая ошибка: Мок-трассировка для шага '{step_id}' не найдена.")
                    plan_failed = True
                    self.planner.prune_branch(plan_id, f"Missing mock trace for {step_id}")
                    self.logger.end_block() # End Step
                    break # Прерываем план

                plan_execution_trace.append({step_id: step_trace})
                
                # --- Отображение ReAct-цикла ---
                for item in step_trace:
                    content = json.loads(item["content"])
                    if item["role"] == "assistant":
                        self.logger.react_thought(content.get("thought", "Нет 'thought'"))
                        action = content.get("action", {})
                        self.logger.react_action(action.get("action"), action.get("payload"))
                    elif item["role"] == "observation":
                        self.logger.react_observation(content)
                # --- Конец отображения ReAct-цикла ---
                
                # 6. (Симуляция) Извлечение результата из 'FINISH'
                finish_action = json.loads(step_trace[-1]["content"])["action"]
                if finish_action["payload"]["status"] == "failure":
                    plan_failed = True
                    feedback = f"Executor failure in {step_id}: {finish_action['payload']['error_info']}"
                    self.logger.fail(feedback)
                    
                    # 8. ОБРАТНАЯ СВЯЗЬ И ОТКАТ
                    self.planner.prune_branch(plan_id, feedback)
                    self.logger.backtrack(f"План {plan_id} отсечен.")
                    self.logger.end_block() # End Step
                    break # Прерываем этот план

                # 7. Верификация Шага (Verifier)
                (status, formal_entry, feedback) = self.verifier.verify_step(step_id, step_trace)
                plan_formal_trace.append(formal_entry)
                
                if status == "FAIL":
                    plan_failed = True
                    feedback_msg = f"Verifier failure in {step_id}: {feedback}"
                    self.logger.fail(feedback_msg)

                    # 8. ОБРАТНАЯ СВЯЗЬ И ОТКАТ
                    self.planner.prune_branch(plan_id, feedback_msg)
                    self.logger.backtrack(f"План {plan_id} отсечен.")
                    self.logger.end_block() # End Step
                    break # Прерываем этот план
                
                self.logger.success(f"Верификация УСПЕШНА: {feedback}")
                
                # Шаг УСПЕШЕН и ВЕРИФИЦИРОВАН
                step_result = finish_action["payload"]["result"]
                current_state_history[step_id] = step_result
                
                self.logger.end_block() # End Step

            # 9. Успешное Завершение Плана
            full_execution_trace[plan_id] = plan_execution_trace
            full_formal_trace[plan_id] = plan_formal_trace
            self.logger.end_block() # End Plan
            
            if not plan_failed:
                self.logger.success(f"===== План {plan_id} УСПЕШНО ВЫПОЛНЕН И ВЕРИФИЦИРОВАН =====")
                
                final_answer = self.get_final_answer_from_history(current_state_history)
                
                final_result = {
                    "problem_id": problem_object["id"],
                    "problem_object": problem_object,
                    "selected_plan": plan,
                    "execution_trace": plan_execution_trace,
                    "formal_trace": plan_formal_trace,
                    "final_answer": final_answer,
                    "verifier_status": "OK",
                    "verifier_report": {
                        "status": "OK",
                        "message": f"План '{plan_id}' успешно верифицирован.",
                        "steps_verified": len(plan_formal_trace)
                    }
                }
                break # Успех, выходим из цикла 'while plan'

        # 10. Завершение
        if final_result:
            self.logger.final_result("Финальный Ответ", final_result)
            return final_result
        else:
            self.logger.fail("===== ВСЕ ПЛАНЫ ПРОВАЛЕНЫ =====")
            final_result = {
                "problem_id": problem_object["id"],
                "problem_object": problem_object,
                "verifier_status": "FAIL",
                "verifier_report": "Все планы были отсечены (PRUNED).",
                "execution_trace_all_plans": full_execution_trace,
                "formal_trace_all_plans": full_formal_trace,
            }
            self.logger.final_result("Финальный Результат (Провал)", final_result)
            return final_result

# --- 7. Точка Входа (Запуск примера) ---

if __name__ == "__main__":
    
    if not GEMINI_API_KEY or not WOLFRAM_API_KEY:
        print("="*50)
        print("ВНИМАНИЕ: Ключи API не установлены.")
        print("Скрипт будет работать в MOCK-РЕЖИМЕ (используя статические ответы).")
        print("="*50)
    
    # Инициализация Оркестратора с новым логгером
    main_logger = StatusLogger(indent_char="  ")
    orchestrator = Orchestrator(logger=main_logger)
    
    # Задача (из отчета)
    query = "15 яблочных пирогов по $12.75 и 7 вишневых пирогов по $14.50 — какова общая выручка?"
    
    # Запуск
    result = orchestrator.main_orchestration_loop(query)
    
    # --- Пример 2: Демонстрация CALL_WOLFRAM (через Toolkit) ---
    main_logger.start_session("Прямой вызов WolframAlpha (Тест Инструмента)")
    wa_action = {
        "action": "CALL_WOLFRAM",
        "payload": {
            "query": "integrate x^2 * sin(x) dx",
            "timeout_ms": 8000
        }
    }
    # Мы вызываем toolkit напрямую, поэтому логгируем вручную
    main_logger.react_action(wa_action["action"], wa_action["payload"])
    wa_result = orchestrator.toolkit.execute_tool(wa_action)
    main_logger.react_observation(wa_result)
    main_logger.end_block()
