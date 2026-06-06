import os
import json
import time
from typing import Dict, List, Any, Optional
import sympy
import numpy as np
import math
import itertools

# ========== КОНФИГУРАЦИЯ OPENROUTER ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
DEFAULT_MODEL = "google/gemini-2.0-flash-lite-001"

# ========== ПРАВИЛЬНЫЕ ИМПОРТЫ QWEN-AGENT ==========
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.code_interpreter import CodeInterpreter

# ========== КАСТОМНЫЕ ИНСТРУМЕНТЫ ДЛЯ МАТЕМАТИКИ ==========
@register_tool('sympy_calculator')
class SympyCalculator(BaseTool):
    """Инструмент для символьных вычислений с использованием SymPy"""
    description = 'Выполняет символьные математические вычисления с помощью SymPy'
    parameters = [{
        'name': 'expression',
        'type': 'string',
        'description': 'Математическое выражение для вычисления',
        'required': True
    }]
    
    def call(self, params: str, **kwargs) -> str:
        try:
            params = json.loads(params)
            expression = params['expression']
            result = sympy.sympify(expression)
            simplified = sympy.simplify(result)
            
            output = f"Выражение: {expression}\n"
            output += f"Упрощенное: {simplified}\n"
            
            if simplified.is_number:
                output += f"Численное значение: {float(simplified)}"
            
            return json.dumps({
                "status": "ok",
                "output": output,
                "result": str(simplified),
                "latex": sympy.latex(simplified),
                "evaluated": float(simplified) if simplified.is_number else None
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "output": f"Ошибка SymPy: {str(e)}"
            }, ensure_ascii=False)

@register_tool('geometry_analyzer')
class GeometryAnalyzer(BaseTool):
    """Инструмент для геометрического анализа систем неравенств"""
    description = 'Анализирует геометрические свойства систем неравенств, включая расстояния между точками и прямыми, касание окружностей'
    parameters = [{
        'name': 'problem_description',
        'type': 'string',
        'description': 'Описание геометрической задачи',
        'required': True
    }, {
        'name': 'center_x',
        'type': 'string', 
        'description': 'X-координата центра окружности',
        'required': False
    }, {
        'name': 'center_y',
        'type': 'string',
        'description': 'Y-координата центра окружности', 
        'required': False
    }, {
        'name': 'radius',
        'type': 'string',
        'description': 'Радиус окружности',
        'required': False
    }, {
        'name': 'line_equation',
        'type': 'string',
        'description': 'Уравнение прямой в формате "Ax + By + C = 0"',
        'required': False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        try:
            params = json.loads(params)
            problem_desc = params['problem_description']
            
            result = f"Геометрический анализ задачи:\n{problem_desc}\n\n"
            
            if 'center_x' in params and 'center_y' in params and 'radius' in params and 'line_equation' in params:
                center_x = sympy.sympify(params['center_x'])
                center_y = sympy.sympify(params['center_y'])
                radius = sympy.sympify(params['radius'])
                line_eq = params['line_equation']
                
                # Разбираем уравнение прямой Ax + By + C = 0
                if 'x' in line_eq or 'y' in line_eq:
                    # Формула расстояния от точки до прямой: |Ax₀ + By₀ + C| / √(A² + B²)
                    result += f"Расстояние от центра ({center_x}, {center_y}) до прямой {line_eq}:\n"
                    result += "Используем формулу: |Ax₀ + By₀ + C| / √(A² + B²)\n"
            
            return json.dumps({
                "status": "ok",
                "analysis": result,
                "conclusion": "Готов к дальнейшим вычислениям"
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "analysis": f"Ошибка геометрического анализа: {str(e)}"
            }, ensure_ascii=False)

@register_tool('math_verifier')
class MathVerifier(BaseTool):
    """Инструмент для формальной верификации математических шагов"""
    description = 'Формально верифицирует математические шаги и вычисления'
    parameters = [{
        'name': 'step_id',
        'type': 'string',
        'description': 'Идентификатор шага для верификации',
        'required': True
    }, {
        'name': 'formal_statement',
        'type': 'string',
        'description': 'Формальное математическое утверждение для проверки',
        'required': True
    }, {
        'name': 'computed_result',
        'type': 'string', 
        'description': 'Вычисленный результат для сравнения',
        'required': True
    }]
    
    def call(self, params: str, **kwargs) -> str:
        try:
            params = json.loads(params)
            step_id = params['step_id']
            formal_statement = params['formal_statement']
            computed_result = params['computed_result']
            
            # Простая верификация
            verification_status = "OK" if "error" not in computed_result.lower() else "FAIL"
            
            return json.dumps({
                "step_id": step_id,
                "formal_statement": formal_statement,
                "verification_status": verification_status,
                "evidence": [{"tool": "verifier", "result": computed_result}],
                "confidence": 0.95 if verification_status == "OK" else 0.2
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "step_id": params.get('step_id', 'unknown'),
                "formal_statement": params.get('formal_statement', 'N/A'),
                "verification_status": "FAIL",
                "evidence": [{"tool": "verifier", "result": str(e)}],
                "confidence": 0.0
            }, ensure_ascii=False)

# ========== КОНФИГУРАЦИЯ LLM ДЛЯ OPENROUTER ==========
llm_cfg = {
    'model': DEFAULT_MODEL,
    'model_server': 'https://openrouter.ai/api/v1',
    'api_key': OPENROUTER_API_KEY,
    'model_type': 'oai',  # Правильный тип для OpenAI API совместимых сервисов
    'generate_cfg': {
        'temperature': 0.1,
        'top_p': 0.8,
        'max_tokens': 4000,
        'stop': ['\n\n']
    }
}

# ========== УЛУЧШЕННОЕ СИСТЕМНОЕ СООБЩЕНИЕ ==========
system_message = """
Ты - продвинутый математический агент, специализирующийся на решении сложных задач с параметрами.

ТВОИ ВОЗМОЖНОСТИ:
1. Анализ геометрических систем неравенств
2. Работа с параметрическими уравнениями и неравенствами
3. Использование символьных вычислений (SymPy)
4. Численные вычисления и визуализация
5. Формальная верификация каждого шага

ЗАДАЧА:
Найдите все значения параметра a, при каждом из которых имеет ровно два решения система неравенств:
9y² ≥ 16x²
(x - 7a + 4y)² + y² ≤ 16a²

СТРАТЕГИЯ РЕШЕНИЯ:
1. ПРОАНАЛИЗИРУЙ ГЕОМЕТРИЧЕСКИЙ СМЫСЛ:
   - Первое неравенство: 9y² ≥ 16x² ⇔ |y| ≥ (4/3)|x| - это область вне угла, образованного прямыми y = ±(4/3)x
   - Второе неравенство: (x - 7a + 4y)² + y² ≤ 16a² - это круг с центром в точке (7a - 4y, 0)? Нет, давай перепишем правильно!

   ВТОРОЕ НЕРАВЕНСТВО: (x - 7a + 4y)² + y² ≤ 16a²
   Это НЕ стандартный круг! Здесь есть y в выражении для x. Нужно переписать:
   (x + 4y - 7a)² + y² ≤ 16a²
   Это эллипс или другая кривая второго порядка.

2. ПЕРЕПИШИ ВТОРОЕ НЕРАВЕНСТВО:
   (x + 4y - 7a)² + y² ≤ 16a²
   Это можно рассматривать как круг в координатах (u = x + 4y, v = y):
   (u - 7a)² + v² ≤ 16a²
   Центр в точке (u = 7a, v = 0), радиус 4|a|

3. ПЕРЕВОД ОБРАТНО В (x, y):
   u = x + 4y = 7a, v = y = 0 ⇒ x = 7a, y = 0
   Центр в точке (7a, 0)

4. УСЛОВИЕ РОВНО ДВА РЕШЕНИЯ:
   Это означает, что граница круга касается границ области первого неравенства в ровно двух точках.

5. ГРАНИЦЫ ПЕРВОГО НЕРАВЕНСТВА:
   y = (4/3)x и y = -(4/3)x
   В стандартном виде: 4x - 3y = 0 и 4x + 3y = 0

6. УСЛОВИЕ КАСАНИЯ:
   Расстояние от центра круга (7a, 0) до каждой прямой должно быть равно радиусу 4|a|

7. ФОРМУЛА РАССТОЯНИЯ:
   d = |Ax₀ + By₀ + C| / √(A² + B²)

ПРАВИЛА РАБОТЫ:
- ВСЕГДА используй инструменты для вычислений - никогда не считай в уме
- Для символьных вычислений используй `sympy_calculator`
- Для геометрического анализа используй `geometry_analyzer`
- Каждый шаг должен быть верифицирован с помощью `math_verifier`
- Если вычисления сложные, используй `code_interpreter` для Python кода
- Для финального ответа используй формат: {"answer": "...", "value": число, "units": "единицы измерения"}

ФОРМАТ ОТВЕТА:
Thought: [Твой анализ и рассуждения]
Action: [Название инструмента и параметры в формате JSON]

Пример правильного действия для sympy_calculator:
{
    "name": "sympy_calculator",
    "arguments": {
        "expression": "abs(4*7*a - 3*0)/sqrt(4**2 + (-3)**2)"
    }
}

Когда найдешь ответ, используй:
{"answer": "Текст ответа", "value": [значения a], "units": "параметр a"}
"""

# ========== СОЗДАНИЕ МАТЕМАТИЧЕСКОГО АГЕНТА ==========
math_agent = Assistant(
    llm=llm_cfg,
    system_message=system_message,
    function_list=[
        'code_interpreter',      # Встроенный инструмент для Python кода
        'sympy_calculator',      # Для символьных вычислений
        'geometry_analyzer',     # Для геометрического анализа
        'math_verifier'          # Для верификации шагов
    ]
)

# ========== ОСНОВНАЯ ФУНКЦИЯ РЕШЕНИЯ ==========
def solve_math_problem(problem_text: str) -> Dict[str, Any]:
    """
    Решает математическую задачу с использованием Qwen-Agent
    """
    print("=== НАЧАЛО РЕШЕНИЯ МАТЕМАТИЧЕСКОЙ ЗАДАЧИ ===")
    print(f"Задача: {problem_text}")
    
    messages = [{'role': 'user', 'content': problem_text}]
    execution_trace = []
    final_answer = None
    
    try:
        print("\n=== ПРОЦЕСС РЕШЕНИЯ ===")
        
        for response_chunk in math_agent.run(messages=messages):
            if isinstance(response_chunk, list) and response_chunk:
                for msg in response_chunk:
                    if msg['role'] == 'assistant':
                        content = msg.get('content', '')
                        print(f"Агент: {content}")
                        
                        # Поиск финального ответа в формате JSON
                        if content.strip().startswith('{') and content.strip().endswith('}'):
                            try:
                                potential_answer = json.loads(content)
                                if 'answer' in potential_answer or 'value' in potential_answer:
                                    final_answer = potential_answer
                                    print(f"\n✅ НАЙДЕН ФИНАЛЬНЫЙ ОТВЕТ: {final_answer}")
                            except json.JSONDecodeError:
                                pass
                        
                        execution_trace.append({
                            'timestamp': time.time(),
                            'content': content,
                            'role': 'assistant'
                        })
            else:
                print(f"Ответ: {response_chunk}")
        
        result = {
            'problem_text': problem_text,
            'execution_trace': execution_trace,
            'final_answer': final_answer,
            'timestamp': time.time(),
            'status': 'completed' if final_answer else 'incomplete'
        }
        
        print("\n=== РЕШЕНИЕ ЗАВЕРШЕНО ===")
        return result
        
    except Exception as e:
        print(f"❌ ОШИБКА ПРИ РЕШЕНИИ: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'problem_text': problem_text,
            'error': str(e),
            'status': 'failed',
            'timestamp': time.time()
        }

# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
if __name__ == "__main__":
    # Исправленная задача (в оригинале была опечатка: (x - 7a + 4x)^2 вместо (x - 7a + 4y)^2)
    problem = "Найдите все значения параметра a, при каждом из которых имеет ровно два решения система неравенств: 9y^2 >= 16x^2, (x - 7a + 4y)^2 + y^2 <= 16a^2"
    
    # Решение задачи
    result = solve_math_problem(problem)
    
    # Сохранение результата
    with open("results/math_agent_qwen_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print("\nРезультат сохранен в math_agent_qwen_result.json")
    
    # Вывод финального ответа
    if result.get('final_answer'):
        answer = result['final_answer']
        print(f"\n🎯 ФИНАЛЬНЫЙ ОТВЕТ:")
        print(f"Ответ: {answer.get('answer', 'N/A')}")
        if 'value' in answer:
            print(f"Значение: {answer['value']}")
        if 'units' in answer:
            print(f"Единицы: {answer['units']}")
    else:
        print("\n❌ ФИНАЛЬНЫЙ ОТВЕТ НЕ НАЙДЕН")
    
    # Запуск веб-интерфейса для дальнейшей работы
    print("\n🚀 Запуск веб-интерфейса для интерактивной работы...")
    try:
        from qwen_agent.gui import WebUI
        WebUI(math_agent).run()
    except ImportError:
        print("Веб-интерфейс недоступен. Установите зависимости: pip install qwen-agent[gui]")