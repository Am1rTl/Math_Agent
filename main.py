#!/usr/bin/env python3
"""
MATH-AGENT-VL reference runtime.

This file implements a lightweight orchestration pipeline that mirrors the
specification provided earlier: Input Formalizer → Tree-of-Thought Planner →
ReAct Executor with PAL mid-thought execution → Verifier/Critic → Final answer.
It is intentionally compact, relies only on the Python standard library, and
falls back to heuristic logic whenever external APIs (Gemini, WolframAlpha)
are not configured. The goal is to provide developers with a runnable scaffold
that can be extended into a full production agent.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
import re
import sys
import textwrap
import time
import uuid
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Tuple

try:
    import sympy as sp
except ImportError:  # SymPy is optional; verifier will degrade gracefully.
    sp = None


# -----------------------------------------------------------------------------
# Helper data structures
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class ProblemObject:
    """Structured representation of the input task."""

    id: str
    types: List[str]
    statement: str
    entities: List[Dict[str, Any]]
    constraints: List[str]
    goals: List[Dict[str, str]]
    hypotheses: List[Dict[str, Any]]
    irrelevant: List[str]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Plan:
    id: str
    summary: str
    steps: List[str]
    estimated_complexity: str
    estimated_tooling: List[str]
    heuristic_score: float
    rationale: str
    status: str = "active"
    structured_steps: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    selection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Action:
    type: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "payload": self.payload}


@dataclasses.dataclass
class ExecutionStep:
    step_id: str
    plan_id: str
    thought: str
    action: Action
    observation: Optional[Dict[str, Any]]
    verifier_hook: str
    status: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "plan_id": self.plan_id,
            "thought": self.thought,
            "action": self.action.to_dict(),
            "observation": self.observation,
            "verifier_hook": self.verifier_hook,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclasses.dataclass
class FormalTraceEntry:
    step_id: str
    formal_statement: str
    verification_status: str
    evidence: List[Dict[str, Any]]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# -----------------------------------------------------------------------------
# External API placeholders (Gemini + Wolfram)
# -----------------------------------------------------------------------------


class GeminiClient:
    """Minimal wrapper around Gemini API 2.5 Flash-lite."""

    def __init__(
        self,
        api_key: Optional[str] = "AIzaSyBgtq6bV-S3XZpk-Tn6q8KrMn8-Wdcihm8",
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
        model: str = "gemini-2.5-flash-lite",
        default_temperature: float = 0.2,
        default_max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.endpoint = endpoint
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Placeholder implementation: if an API key is configured, the method
        describes the HTTP request developers should send. To keep this example
        offline-friendly, we do not perform the request automatically.
        """

        temp = temperature if temperature is not None else self.default_temperature
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens

        if not self.available:
            return (
                "[Gemini unavailable]\n"
                "Set GEMINI_API_KEY to enable live generation.\n"
                f"System prompt preview: {system_prompt[:80]}...\n"
                f"User prompt preview: {user_prompt[:80]}..."
            )

        payload = {
            "model": self.model,
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generation_config": {
                "temperature": temp,
                "max_output_tokens": max_tok,
                "top_p": 0.9,
                "top_k": 32,
            },
        }

        # Instead of sending the HTTP request (requires requests/urllib and
        # network), we provide a descriptive stub to show developers how to call
        # the API. Uncomment the block below to enable real requests.
        #
        # import requests
        # response = requests.post(
        #     self.endpoint,
        #     headers={
        #         "Authorization": f"Bearer {self.api_key}",
        #         "Content-Type": "application/json",
        #     },
        #     data=json.dumps(payload),
        #     timeout=30,
        # )
        # response.raise_for_status()
        # data = response.json()
        # return data["candidates"][0]["content"]["parts"][0]["text"]

        # For demo purposes return a deterministic string.
        return (
            "[Gemini stubbed response]\n"
            f"System prompt: {system_prompt[:80]}...\n"
            f"User prompt: {user_prompt[:80]}..."
        )


class WolframClient:
    """Wrapper for the Full Results API."""

    def __init__(
        self,
        app_id: Optional[str] = None,
        endpoint: str = "https://api.wolframalpha.com/v2/query",
    ) -> None:
        self.app_id = app_id or os.getenv("WOLFRAM_API_KEY")
        self.endpoint = endpoint

    def call(self, query: str, timeout_ms: int = 8000) -> Dict[str, Any]:
        payload = {
            "input": query,
            "appid": self.app_id,
            "output": "JSON",
            "scantimeout": timeout_ms / 1000.0,
        }
        if not self.app_id:
            return {
                "status": "error",
                "reason": "WOLFRAM_API_KEY not configured",
                "query": query,
            }

        # Offline-friendly stub; uncomment to perform real request.
        # import requests
        # response = requests.get(self.endpoint, params=payload, timeout=timeout_ms / 1000)
        # response.raise_for_status()
        # return {"status": "ok", "raw": response.json()}

        return {
            "status": "stubbed",
            "query": query,
            "info": "Replace with real HTTP call once WOLFRAM_API_KEY is set.",
        }


# -----------------------------------------------------------------------------
# Core subsystems
# -----------------------------------------------------------------------------


class InputFormalizer:
    """Heuristic formalizer with optional Gemini fallback."""

    def __init__(self, gemini: Optional[GeminiClient] = None) -> None:
        self.gemini = gemini

    def formalize(self, text: str) -> ProblemObject:
        if self.gemini and self.gemini.available:
            # Developers may replace the heuristic fallback with a live Gemini
            # call. We keep the stub for ease of experimentation.
            _ = self.gemini.generate(
                system_prompt="Input Formalizer system prompt (see specification).",
                user_prompt=f"Переформализуй задачу: ```{text}```",
            )
        return self._heuristic_formalize(text)

    def _heuristic_formalize(self, text: str) -> ProblemObject:
        statement = text.strip()
        problem_id = f"heur_{uuid.uuid4().hex[:8]}"
        matches = list(
            re.finditer(
                r"(\d+)\s+([A-Za-z]+)\s+pies?\s+at\s+\$?\s*(\d+(?:\.\d+)?)",
                text,
                flags=re.IGNORECASE,
            )
        )
        entities: List[Dict[str, Any]] = []
        constraints: List[str] = []
        notes_parts: List[str] = []

        for match in matches:
            count = int(match.group(1))
            flavor = match.group(2).lower()
            price = float(match.group(3))
            entities.append({"name": f"{flavor}_pies", "value": count, "unit": "count"})
            entities.append(
                {"name": f"{flavor}_price", "value": price, "unit": "USD"}
            )
            constraints.append(f"{flavor} pies cost ${price:.2f} each")

        types = {"word_problem"}
        if matches:
            types.add("arithmetic")

        goals: List[Dict[str, Any]] = []
        if matches:
            goals.append({"type": "value", "target": "total_revenue"})
        if not goals:
            goals.append({"type": "value", "target": "answer"})

        hypotheses: List[Dict[str, Any]] = []
        if not entities:
            hypotheses.append(
                {
                    "formalization": {"assumption": "Could not parse quantities"},
                    "confidence": 0.2,
                }
            )

        notes_parts.append(
            "Heuristic formalization (Gemini fallback not invoked). "
            "Extend this module to support richer parsing."
        )
        notes = " ".join(notes_parts)

        return ProblemObject(
            id=problem_id,
            types=sorted(types),
            statement=statement,
            entities=entities,
            constraints=constraints,
            goals=goals,
            hypotheses=hypotheses,
            irrelevant=[],
            notes=notes,
        )


class Planner:
    """Tree-of-Thought planner that can generate and rank multiple plans."""

    def __init__(self, gemini: Optional[GeminiClient] = None) -> None:
        self.gemini = gemini

    def propose_plans(self, problem: ProblemObject) -> List[Plan]:
        plans: List[Plan] = []
        if self.gemini:
            plans = self._request_plans_from_gemini(problem)
        if not plans:
            plans = self._fallback_plans()
        return plans

    def select_plan(
        self,
        problem: ProblemObject,
        plans: List[Plan],
    ) -> Dict[str, Any]:
        if not plans:
            raise RuntimeError("Planner produced no plans.")
        if len(plans) == 1:
            plans[0].selection_reason = "Единственный доступный план."
            return {"plan": plans[0], "reason": plans[0].selection_reason}

        if self.gemini:
            decision = self._ask_gemini_to_pick_plan(problem, plans)
            if decision:
                selected_id = decision.get("selected_plan_id")
                selected_plan = next(
                    (p for p in plans if p.id == selected_id), None
                )
                if selected_plan:
                    reason = decision.get("reason", "LLM decision.")
                    selected_plan.selection_reason = reason
                    return {"plan": selected_plan, "reason": reason}

        selected_plan = max(plans, key=lambda p: p.heuristic_score)
        selected_plan.selection_reason = "Выбран по наибольшему heuristic_score."
        return {"plan": selected_plan, "reason": selected_plan.selection_reason}

    def _request_plans_from_gemini(self, problem: ProblemObject) -> List[Plan]:
        prompt = textwrap.dedent(
            f"""
            Ты — стратегический ToT-планировщик. Сгенерируй 2-3 независимых плана
            для решения следующего ProblemObject. Каждый план должен быть JSON-объектом
            с полями:
            - id
            - summary
            - steps (список текстовых шагов)
            - estimated_complexity ∈ [low,medium,high]
            - estimated_tooling (подмножество ["Gemini","PAL","SymPy","Wolfram","Python"])
            - heuristic_score ∈ [0,1]
            - rationale
            - structured_steps: массив объектов вида
              {{
                "id": "analysis_1",
                "tool": "gemini|gemini_reason|python|sympy|wolfram|gemini_review|finish",
                "description": "<что делает шаг>",
                "units": "..." (optional),
                "formal_statement": "..." (optional)
              }}
            Верни строго JSON-массив планов без комментариев.

            ProblemObject:
            {json.dumps(problem.to_dict(), ensure_ascii=False, indent=2)}
            """
        ).strip()
        raw = self.gemini.generate(
            system_prompt="Tree-of-Thought Planner для MATH-AGENT-VL.",
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=2048,
        )
        plan_data = self._parse_plans_from_json(raw)
        return [self._dict_to_plan(entry) for entry in plan_data]

    def _ask_gemini_to_pick_plan(
        self,
        problem: ProblemObject,
        plans: List[Plan],
    ) -> Optional[Dict[str, Any]]:
        plans_payload = [
            {
                "id": plan.id,
                "summary": plan.summary,
                "estimated_complexity": plan.estimated_complexity,
                "estimated_tooling": plan.estimated_tooling,
                "heuristic_score": plan.heuristic_score,
                "rationale": plan.rationale,
                "steps": plan.steps,
            }
            for plan in plans
        ]
        prompt = textwrap.dedent(
            f"""
            Выбери лучший план для следующей задачи. Ответ дай JSON-объектом:
            {{
              "selected_plan_id": "...",
              "reason": "..."
            }}

            ProblemObject:
            {json.dumps(problem.to_dict(), ensure_ascii=False, indent=2)}

            Candidate plans:
            {json.dumps(plans_payload, ensure_ascii=False, indent=2)}
            """
        ).strip()
        raw = self.gemini.generate(
            system_prompt="ToT планировщик-селекционер.",
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=512,
        )
        try:
            decision = json.loads(self._extract_best_json(raw))
            return decision
        except Exception:  # pylint: disable=broad-except
            return None

    def _fallback_plans(self) -> List[Plan]:
        template_a = [
            {"id": "analysis", "tool": "gemini", "description": "LLM анализирует постановку и извлекает факты."},
            {"id": "reason_strategy", "tool": "gemini_reason", "description": "LLM предлагает стратегию и план PAL-кода."},
            {"id": "pal_exec", "tool": "python", "description": "Выполняет PAL-код для численного результата."},
            {"id": "sympy_check", "tool": "sympy", "description": "Проверяет результат символически."},
            {"id": "review", "tool": "gemini_review", "description": "LLM оценивает корректность решения."},
            {"id": "wolfram_cross", "tool": "wolfram", "description": "Кросс-проверяет результат внешним сервисом."},
            {"id": "final", "tool": "finish", "description": "Формирует финальный ответ."},
        ]
        template_b = [
            {"id": "analysis", "tool": "gemini", "description": "LLM анализирует и строит несколько гипотез."},
            {"id": "pal_exec", "tool": "python", "description": "Пробует получить численный ответ через PAL."},
            {"id": "reason_refine", "tool": "gemini_reason", "description": "LLM анализирует результат и предлагает улучшения."},
            {"id": "wolfram_cross", "tool": "wolfram", "description": "Получает справочные данные из WolframAlpha."},
            {"id": "sympy_check", "tool": "sympy", "description": "Сравнивает с символическим результатом."},
            {"id": "review", "tool": "gemini_review", "description": "Формирует оценку качества решения."},
            {"id": "final", "tool": "finish", "description": "Отправляет итог пользователю."},
        ]
        return [
            self._plan_from_template(
                template_a,
                summary="LLM-анализ → PAL → SymPy → Review → Wolfram → финал.",
                tooling=["Gemini", "PAL", "SymPy", "Wolfram"],
                heuristic=0.72,
            ),
            self._plan_from_template(
                template_b,
                summary="PAL → LLM-рефайн → Wolfram → SymPy → Review → финал.",
                tooling=["PAL", "Gemini", "Wolfram", "SymPy"],
                heuristic=0.7,
            ),
        ]

    def _plan_from_template(
        self,
        template: List[Dict[str, Any]],
        summary: str,
        tooling: List[str],
        heuristic: float,
    ) -> Plan:
        steps = [step["description"] for step in template]
        return Plan(
            id=f"plan_{uuid.uuid4().hex[:6]}",
            summary=summary,
            steps=steps,
            estimated_complexity="medium",
            estimated_tooling=tooling,
            heuristic_score=heuristic,
            rationale="Fallback ToT цепочка без привязки к задаче.",
            structured_steps=template,
        )

    def _dict_to_plan(self, data: Dict[str, Any]) -> Plan:
        structured_steps = data.get("structured_steps") or []
        return Plan(
            id=data.get("id", f"plan_{uuid.uuid4().hex[:6]}"),
            summary=data.get("summary", "План от Gemini."),
            steps=data.get("steps", []),
            estimated_complexity=data.get("estimated_complexity", "medium"),
            estimated_tooling=data.get("estimated_tooling", ["Gemini", "PAL"]),
            heuristic_score=float(data.get("heuristic_score", 0.6)),
            rationale=data.get("rationale", ""),
            structured_steps=structured_steps,
        )

    def _parse_plans_from_json(self, text: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        chunks = []
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            chunks.append(stripped)
        else:
            start = stripped.find("[")
            end = stripped.rfind("]")
            if start != -1 and end != -1 and end > start:
                chunks.append(stripped[start : end + 1])
            else:
                start = stripped.find("{")
                end = stripped.rfind("}")
                if start != -1 and end != -1:
                    chunks.append(f"[{stripped[start:end+1]}]")
        for chunk in chunks:
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                candidates.append(data)
            elif isinstance(data, list):
                candidates.extend([item for item in data if isinstance(item, dict)])
        return candidates

    @staticmethod
    def _extract_best_json(text: str) -> str:
        stripped = text.strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found.")
        return stripped[start : end + 1]

class PythonSandbox:
    """Very small exec-based sandbox with stdout capture."""

    SAFE_BUILTINS = {
        "abs": abs,
        "min": min,
        "max": max,
        "range": range,
        "len": len,
        "float": float,
        "int": int,
        "sum": sum,
        "math": math,
        "isinstance": isinstance,
        "dict": dict,
    }

    def run(self, code: str, context: Dict[str, Any], timeout_ms: int = 5000) -> Dict[str, Any]:
        # Timeout management is simplified; for real sandboxing use subprocesses.
        local_scope: Dict[str, Any] = {}
        global_scope = {"__builtins__": self.SAFE_BUILTINS.copy()}
        global_scope.update(context)
        stdout_buffer = io.StringIO()
        start = time.perf_counter()
        try:
            with redirect_stdout(stdout_buffer):
                exec(code, global_scope, local_scope)
        except Exception as exc:  # pylint: disable=broad-except
            elapsed = int((time.perf_counter() - start) * 1000)
            return {
                "status": "error",
                "stdout": stdout_buffer.getvalue(),
                "return_value": None,
                "trace_logs": [
                    f"exec_ms={elapsed}",
                    f"error={exc.__class__.__name__}",
                    f"message={exc}",
                ],
            }

        result = local_scope.get("result") or local_scope.get("return_value")
        elapsed = int((time.perf_counter() - start) * 1000)
        return {
            "status": "ok",
            "stdout": stdout_buffer.getvalue(),
            "return_value": result,
            "trace_logs": [f"exec_ms={elapsed}"],
        }


class ReActExecutor:
    """Executes structured ToT plans using Gemini reasoning and tool calls."""

    def __init__(
        self,
        sandbox: Optional[PythonSandbox] = None,
        gemini: Optional[GeminiClient] = None,
        wolfram: Optional[WolframClient] = None,
    ) -> None:
        self.sandbox = sandbox or PythonSandbox()
        self.gemini = gemini
        self.wolfram = wolfram or WolframClient(os.getenv("WOLFRAM_API_KEY"))
        self.heuristic_solver = HeuristicSolver()

    def execute_plan(
        self,
        problem: ProblemObject,
        plan: Plan,
    ) -> Dict[str, Any]:
        structured_steps = plan.structured_steps or self.heuristic_solver.default_structured_steps()
        execution_trace: List[ExecutionStep] = []
        state: Dict[str, Any] = {
            "final_units": None,
            "errors": [],
            "analysis": "",
            "reasoning": [],
            "reviews": [],
        }

        for step_meta in structured_steps:
            step = self._execute_structured_step(
                problem, plan, step_meta, state, len(execution_trace) + 1
            )
            execution_trace.append(step)

        final_value = state.get("final_value")
        final_units = state.get("final_units") or "unitless"
        status = "completed" if not state["errors"] else "error"

        return {
            "execution_trace": execution_trace,
            "final_value": final_value,
            "final_units": final_units,
            "status": status,
        }

    def _execute_structured_step(
        self,
        problem: ProblemObject,
        plan: Plan,
        step_meta: Dict[str, Any],
        state: Dict[str, Any],
        step_index: int,
    ) -> ExecutionStep:
        tool = step_meta.get("tool")
        thought = step_meta.get("description", f"Исполняю шаг {tool}.")
        step_id = f"react_{step_index}"

        handler_map = {
            "gemini": self._run_gemini_analysis,
            "gemini_reason": self._run_gemini_reasoner,
            "python": self._run_python_pal,
            "sympy": self._run_sympy_step,
            "wolfram": self._run_wolfram_step,
            "gemini_review": self._run_gemini_review,
            "finish": self._run_finish_step,
        }
        handler = handler_map.get(tool)
        if handler:
            return handler(problem, plan, step_id, thought, state, step_meta)

        action = Action(type="SKIP", payload={"tool": tool})
        return ExecutionStep(
            step_id=step_id,
            plan_id=plan.id,
            thought=thought,
            action=action,
            observation={
                "status": "skipped",
                "stdout": "",
                "return_value": None,
                "trace_logs": [],
            },
            verifier_hook="skip",
            status="completed",
            metadata={"info": f"tool {tool} not implemented"},
        )

    def _run_gemini_analysis(
        self,
        problem: ProblemObject,
        plan: Plan,
        step_id: str,
        thought: str,
        state: Dict[str, Any],
        step_meta: Dict[str, Any],
    ) -> ExecutionStep:
        prompt = f"""Проанализируй математическую задачу и перечисли:
- известные данные
- что требуется найти
- потенциальные методы (Python, SymPy, Wolfram и т.д.)

Ответ дай кратким списком.

ProblemObject:
{json.dumps(problem.to_dict(), ensure_ascii=False, indent=2)}"""
        response_text = self.gemini.generate(
            system_prompt="Аналитик MATH-AGENT-VL.",
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=512,
        )
        state["analysis"] = response_text
        action = Action(
            type="CALL_GEMINI",
            payload={
                "model": "gemini-2.5-flash-lite",
                "role": "assistant",
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 512,
                "stream": False,
            },
        )
        observation = {
            "status": "ok",
            "stdout": "",
            "return_value": {"text": response_text},
            "trace_logs": [],
        }
        return ExecutionStep(
            step_id=step_id,
            plan_id=plan.id,
            thought=thought,
            action=action,
            observation=observation,
            verifier_hook="skip",
            status="completed",
            metadata={"analysis_summary": response_text[:200]},
        )

    def _run_gemini_reasoner(
        self,
        problem: ProblemObject,
        plan: Plan,
        step_id: str,
        thought: str,
        state: Dict[str, Any],
        step_meta: Dict[str, Any],
    ) -> ExecutionStep:
        prompt = f"""У тебя есть следующая информация об анализе задачи:
{state.get('analysis', 'нет анализа')}

Сформулируй стратегию решения в 3-5 шагах, укажи какие инструменты вызвать
(Python, SymPy, Wolfram и т.д.). Если нужен PAL-код, опиши входы/выходы.
Ответ верни JSON-объектом с ключами strategy, recommended_units,
sympy_expression_hint, notes.

Problem statement:
{problem.statement}"""
        response_text = self.gemini.generate(
            system_prompt="Reasoner для MATH-AGENT-VL.",
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=768,
        )
        state.setdefault("reasoning", []).append(response_text)
        action = Action(
            type="CALL_GEMINI",
            payload={
                "model": "gemini-2.5-flash-lite",
                "role": "assistant",
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 768,
                "stream": False,
            },
        )
        observation = {
            "status": "ok",
            "stdout": "",
            "return_value": {"text": response_text},
            "trace_logs": [],
        }
        return ExecutionStep(
            step_id=step_id,
            plan_id=plan.id,
            thought=thought,
            action=action,
            observation=observation,
            verifier_hook="skip",
            status="completed",
            metadata={"reasoning_excerpt": response_text[:200]},
        )

    def _run_python_pal(
        self,
        problem: ProblemObject,
        plan: Plan,
        step_id: str,
        thought: str,
        state: Dict[str, Any],
        step_meta: Dict[str, Any],
    ) -> ExecutionStep:
        code, origin = self._obtain_pal_code(problem, step_meta, state)
        action = Action(
            type="CALL_PYTHON",
            payload={
                "code": code,
                "sandbox": "math-agent-default",
                "timeout_ms": 5000,
            },
        )
        observation = self.sandbox.run(
            code,
            context={"problem": problem.to_dict()},
            timeout_ms=5000,
        )
        if observation["status"] != "ok":
            state["errors"].append(step_id)
        else:
            return_value = observation.get("return_value") or {}
            result_value = return_value.get("result")
            state["python_return"] = return_value
            state["final_value"] = result_value
            state["final_units"] = origin.get("units") or state.get("final_units")
            state["sympy_expression"] = (
                origin.get("sympy_expression")
                or step_meta.get("sympy_expression")
                or result_value
            )
        metadata = {
            "tool_level": 1,
            "formal_statement": origin.get("formal_statement"),
            "pal_step_code": code,
            "origin": origin.get("source", "unknown"),
        }
        return ExecutionStep(
            step_id=step_id,
            plan_id=plan.id,
            thought=thought,
            action=action,
            observation=observation,
            verifier_hook="pending",
            status="completed" if observation["status"] == "ok" else "error",
            metadata=metadata,
        )

    def _run_sympy_step(
        self,
        problem: ProblemObject,
        plan: Plan,
        step_id: str,
        thought: str,
        state: Dict[str, Any],
        step_meta: Dict[str, Any],
    ) -> ExecutionStep:
        value = state.get("final_value")
        sympy_expr = step_meta.get("sympy_expression") or state.get("sympy_expression") or value
        if value is None:
            observation = {
                "status": "error",
                "stdout": "",
                "return_value": None,
                "trace_logs": ["reason=no_result"],
            }
            state["errors"].append(step_id)
        elif sp is None:
            observation = {
                "status": "error",
                "stdout": "",
                "return_value": None,
                "trace_logs": ["reason=sympy_not_installed"],
            }
            state["errors"].append(step_id)
        else:
            try:
                sympy_value = sp.nsimplify(sympy_expr)
                numeric = float(sympy_value)
                diff = abs(numeric - float(value))
                observation = {
                    "status": "ok",
                    "stdout": "",
                    "return_value": {"sympy_value": numeric, "difference": diff},
                    "trace_logs": [],
                }
            except Exception as exc:
                observation = {
                    "status": "error",
                    "stdout": "",
                    "return_value": {"error": str(exc)},
                    "trace_logs": [],
                }
                state["errors"].append(step_id)
        action = Action(
            type="CALL_SYMPY",
            payload={"expression": str(sympy_expr), "target_value": value},
        )
        status = "completed" if observation["status"] == "ok" else "error"
        return ExecutionStep(
            step_id=step_id,
            plan_id=plan.id,
            thought=thought,
            action=action,
            observation=observation,
            verifier_hook="skip",
            status=status,
            metadata={"formal_statement": f"sympy({sympy_expr}) = {value}"},
        )

    def _run_wolfram_step(
        self,
        problem: ProblemObject,
        plan: Plan,
        step_id: str,
        thought: str,
        state: Dict[str, Any],
        step_meta: Dict[str, Any],
    ) -> ExecutionStep:
        final_value = state.get("final_value")
        query = (
            f"{problem.statement}\n"
            f"LLM estimate: {final_value}\n"
            "Provide exact symbolic/analytic confirmation."
        )
        wolfram_response = self.wolfram.call(query)
        action = Action(
            type="CALL_WOLFRAM",
            payload={
                "query": query,
                "api_key": "<WOLFRAM_API_KEY>",
                "timeout_ms": 8000,
                "format": "structured",
            },
        )
        observation = {
            "status": wolfram_response.get("status", "ok"),
            "stdout": "",
            "return_value": wolfram_response,
            "trace_logs": [],
        }
        status = "completed" if observation["status"] == "ok" else "error"
        if status == "error":
            state["errors"].append(step_id)
        return ExecutionStep(
            step_id=step_id,
            plan_id=plan.id,
            thought=thought,
            action=action,
            observation=observation,
            verifier_hook="skip",
            status=status,
            metadata={"formal_statement": "wolfram_crosscheck"},
        )

    def _run_gemini_review(
        self,
        problem: ProblemObject,
        plan: Plan,
        step_id: str,
        thought: str,
        state: Dict[str, Any],
        step_meta: Dict[str, Any],
    ) -> ExecutionStep:
        summary = textwrap.shorten(
            f"analysis={state.get('analysis')}; python={state.get('python_return')}",
            width=800,
        )
        prompt = f"""Проанализируй текущий прогресс математической задачи. Укажи, достаточно ли обоснований,
нужны ли дополнительные проверки, и оцени риски ошибки. Ответ верни JSON-объектом:
{{
  "verdict": "proceed|revise|unknown",
  "issues": ["..."],
  "recommendations": ["..."]
}}

Context:
{summary}"""
        response_text = self.gemini.generate(
            system_prompt="Критик/ревьюер MATH-AGENT-VL.",
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=512,
        )
        state.setdefault("reviews", []).append(response_text)
        action = Action(
            type="CALL_GEMINI",
            payload={
                "model": "gemini-2.5-flash-lite",
                "role": "assistant",
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 512,
                "stream": False,
            },
        )
        observation = {
            "status": "ok",
            "stdout": "",
            "return_value": {"text": response_text},
            "trace_logs": [],
        }
        return ExecutionStep(
            step_id=step_id,
            plan_id=plan.id,
            thought=thought,
            action=action,
            observation=observation,
            verifier_hook="skip",
            status="completed",
            metadata={"review_excerpt": response_text[:160]},
        )

    def _run_finish_step(
        self,
        problem: ProblemObject,
        plan: Plan,
        step_id: str,
        thought: str,
        state: Dict[str, Any],
        step_meta: Dict[str, Any],
    ) -> ExecutionStep:
        final_value = state.get("final_value")
        confidence = 0.9 if not state["errors"] else 0.5
        action = Action(
            type="FINISH",
            payload={
                "final_answer": (
                    f"{final_value}" if final_value is not None else "Не удалось вычислить результат"
                ),
                "formal_trace_pointer": "formal_trace[-1]",
                "confidence": confidence,
            },
        )
        return ExecutionStep(
            step_id=step_id,
            plan_id=plan.id,
            thought=thought,
            action=action,
            observation=None,
            verifier_hook="OK" if not state["errors"] else "pending",
            status="completed",
            metadata={},
        )

    def _obtain_pal_code(
        self,
        problem: ProblemObject,
        step_meta: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        prompt = f"""Ты — генератор PAL-кода. Используй информацию:
Анализ: {state.get('analysis', 'нет анализа')}
Рассуждения: {state.get('reasoning', [])}

Требуется сгенерировать функцию solve_step(problem: dict) -> dict.
Требования:
- использовать данные ProblemObject;
- добавить assert-предусловия и постусловия;
- вернуть dict с ключами ok/result/witness;
- избегать сторонних библиотек кроме math.

ProblemObject:
{json.dumps(problem.to_dict(), ensure_ascii=False, indent=2)}"""
        origin = {
            "source": "gemini",
            "formal_statement": step_meta.get("formal_statement"),
            "units": step_meta.get("units"),
            "sympy_expression": step_meta.get("sympy_expression"),
        }
        if self.gemini:
            response = self.gemini.generate(
                system_prompt="PAL-кодогенератор MATH-AGENT-VL.",
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=1024,
            )
            code = self._extract_code_block(response)
            if code:
                return code, origin
        fallback = self.heuristic_solver.build(problem)
        return fallback["code"], fallback

    @staticmethod
    def _extract_code_block(text: str) -> Optional[str]:
        matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
        if matches:
            return matches[0].strip()
        if "def solve" in text:
            return text.strip()
        return None
class HeuristicSolver:
    """Generic deterministic fallback when Gemini is unavailable."""

    def build(self, problem: ProblemObject) -> Dict[str, Any]:
        numeric_entities = [
            entity["value"]
            for entity in problem.entities
            if isinstance(entity.get("value"), (int, float))
        ]
        if numeric_entities:
            return self._build_numeric_sum(numeric_entities)
        return self._build_default()

    @staticmethod
    def default_structured_steps() -> List[Dict[str, Any]]:
        return [
            {"id": "analysis", "tool": "gemini", "description": "LLM анализирует задачу."},
            {"id": "reason", "tool": "gemini_reason", "description": "LLM предлагает стратегию и инструменты."},
            {"id": "pal", "tool": "python", "description": "Выполняет PAL-код для вычисления результата."},
            {"id": "sympy", "tool": "sympy", "description": "Сверяет результат символически."},
            {"id": "review", "tool": "gemini_review", "description": "LLM оценивает корректность решения."},
            {"id": "wolfram", "tool": "wolfram", "description": "Выполняет внешний запрос для кросс-проверки."},
            {"id": "finish", "tool": "finish", "description": "Формирует финальный ответ."},
        ]

    @staticmethod
    def _build_numeric_sum(values: List[float]) -> Dict[str, Any]:
        values_literal = repr(values)
        expression = " + ".join(str(v) for v in values) if values else "0"
        code = f"""
def solve_step_auto(problem: dict) -> dict:
    # intent: sum numeric entities as a placeholder computation
    values = {values_literal}
    total = sum(values)
    return {{'ok': True, 'result': total, 'witness': {{'values': values}}}}

result = solve_step_auto(problem)
""".strip("\n")
        return {
            "code": code,
            "formal_statement": f"auto_sum = {expression}",
            "units": "unitless",
            "sympy_expression": expression,
            "source": "heuristic_numeric_sum",
        }

    @staticmethod
    def _build_default() -> Dict[str, Any]:
        code = """
def solve_step_auto(problem: dict) -> dict:
    # intent: fallback solver placeholder
    return {'ok': True, 'result': 0.0, 'witness': {'note': 'placeholder'} }

result = solve_step_auto(problem)
""".strip("\n")
        return {
            "code": code,
            "formal_statement": "fallback_result = 0",
            "units": "unitless",
            "sympy_expression": "0",
            "source": "heuristic_fallback",
        }


class Verifier:
    """Replays PAL code and optionally uses SymPy for extra certainty."""

    def __init__(self, sandbox: Optional[PythonSandbox] = None) -> None:
        self.sandbox = sandbox or PythonSandbox()

    def verify_steps(
        self,
        problem: ProblemObject,
        execution_trace: List[ExecutionStep],
    ) -> Dict[str, Any]:
        formal_trace: List[FormalTraceEntry] = []
        verifier_status = "OK"
        issues: List[Dict[str, Any]] = []

        for step in execution_trace:
            if step.action.type != "CALL_PYTHON" or not step.observation:
                continue
            code = step.action.payload.get("code", "")
            rerun = self.sandbox.run(
                code,
                context={"problem": problem.to_dict()},
                timeout_ms=step.action.payload.get("timeout_ms", 5000),
            )
            status = "OK"
            evidence = [{"tool": "PAL re-run", "result": rerun["return_value"]}]

            if (
                rerun["status"] != "ok"
                or rerun["return_value"] != step.observation["return_value"]
            ):
                status = "FAIL"
                verifier_status = "FAIL"
                issues.append(
                    {
                        "step_id": step.step_id,
                        "reason": "Mismatch between execution and verifier replay.",
                    }
                )

            formal_statement = step.metadata.get("formal_statement", "")
            if formal_statement and sp and status == "OK":
                try:
                    rhs = formal_statement.split("=", maxsplit=1)[-1].strip()
                    sympy_value = sp.sympify(rhs).evalf()
                    evidence.append({"tool": "SymPy", "result": float(sympy_value)})
                except Exception as exc:  # pylint: disable=broad-except
                    evidence.append({"tool": "SymPy", "result": f"error: {exc}"})

            confidence = 0.99 if status == "OK" else 0.1
            formal_trace.append(
                FormalTraceEntry(
                    step_id=step.step_id,
                    formal_statement=formal_statement or "N/A",
                    verification_status=status,
                    evidence=evidence,
                    confidence=confidence,
                )
            )

        report = {
            "summary": "All steps verified" if verifier_status == "OK" else "Issues found",
            "issues": issues,
        }
        return {
            "formal_trace": formal_trace,
            "verifier_status": verifier_status,
            "verifier_report": report,
        }


class MathAgent:
    """High-level orchestrator that wires all subsystems together."""

    def __init__(
        self,
        gemini: Optional[GeminiClient] = None,
        wolfram: Optional[WolframClient] = None,
    ) -> None:
        self.gemini = gemini or GeminiClient()
        self.wolfram = wolfram or WolframClient(app_id=os.getenv("WOLFRAM_API_KEY"))
        self.formalizer = InputFormalizer(self.gemini)
        self.planner = Planner(self.gemini)
        self.executor = ReActExecutor(gemini=self.gemini, wolfram=self.wolfram)
        self.verifier = Verifier()

    def solve(self, problem_text: str) -> Dict[str, Any]:
        problem = self.formalizer.formalize(problem_text)
        plans = self.planner.propose_plans(problem)
        plan_selection = self.planner.select_plan(problem, plans)
        selected_plan: Plan = plan_selection["plan"]
        selected_plan.selection_reason = plan_selection.get("reason", "")

        exec_result = self.executor.execute_plan(problem, selected_plan)
        execution_trace = exec_result["execution_trace"]
        total_value = exec_result.get("final_value")
        final_units = exec_result.get("final_units")

        verification = self.verifier.verify_steps(problem, execution_trace)
        formal_trace = verification["formal_trace"]
        verifier_status = verification["verifier_status"]
        verifier_report = verification["verifier_report"]

        confidence = 0.95 if exec_result.get("status") == "completed" else 0.4
        final_answer = {
            "value": total_value,
            "units": final_units,
            "confidence": confidence,
        }

        result = {
            "problem_id": problem.id,
            "problem_object": problem.to_dict(),
            "selected_plan": selected_plan.to_dict(),
            "plan_selection": {"reason": selected_plan.selection_reason},
            "execution_trace": [step.to_dict() for step in execution_trace],
            "formal_trace": [entry.to_dict() for entry in formal_trace],
            "final_answer": final_answer,
            "verifier_status": verifier_status,
            "verifier_report": verifier_report,
        }
        return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="MATH-AGENT-VL demo runner.")
    parser.add_argument(
        "--problem",
        type=str,
        help="Текст задачи. Если не указан, используется демонстрационный пример.",
    )
    parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Формат вывода: человекочитаемый или чистый JSON.",
    )
    args = parser.parse_args()

    if args.problem:
        problem_text = args.problem
    else:
        problem_text = (
            """Какова вероятность получить комбинацию "Фулл-Хаус" (Full House)?

Определение: Фулл-Хаус — это 5-карточная рука, в которой три карты имеют один номинал, а две оставшиеся карты — другой номинал.

Пример: 7, 7, 7, K, K (три семерки и два короля) """
        )

    agent = MathAgent()
    result = agent.solve(problem_text)
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(render_pretty(result))


def render_pretty(result: Dict[str, Any]) -> str:
    """Builds a human-readable report from the JSON output."""

    def fmt_return_value(rv: Any) -> str:
        if rv is None:
            return "нет данных"
        if isinstance(rv, dict):
            parts = []
            if "result" in rv:
                parts.append(f"result={rv['result']}")
            if "witness" in rv:
                parts.append(f"witness={rv['witness']}")
            if not parts:
                parts = [json.dumps(rv, ensure_ascii=False)]
            return ", ".join(parts)
        return str(rv)

    lines: List[str] = []
    lines.append("=== MATH-AGENT-VL :: Исполнение задачи ===")
    problem = result.get("problem_object", {})
    lines.append(f"Задача [{result.get('problem_id')}]: {problem.get('statement')}")
    lines.append(f"Типы: {', '.join(problem.get('types', [])) or 'не указаны'}")
    lines.append("")

    plan = result.get("selected_plan", {})
    lines.append("План:")
    lines.append(f"  Идентификатор: {plan.get('id')}")
    lines.append(f"  Кратко: {plan.get('summary')}")
    lines.append(
        f"  Сложность: {plan.get('estimated_complexity')} | Инструменты: "
        f"{', '.join(plan.get('estimated_tooling', [])) or '—'}"
    )
    selection_reason = plan.get("selection_reason") or result.get("plan_selection", {}).get("reason")
    if selection_reason:
        lines.append(f"  Обоснование выбора: {selection_reason}")
    for idx, step in enumerate(plan.get("steps", []), start=1):
        lines.append(f"    {idx}. {step}")
    lines.append("")

    lines.append("Ход выполнения:")
    execution_trace = result.get("execution_trace", [])
    for idx, step in enumerate(execution_trace, start=1):
        action = step.get("action", {})
        observation = step.get("observation")
        lines.append(
            f"- Шаг {idx} ({step.get('step_id')} / {action.get('type')}): "
            f"{step.get('status')}"
        )
        lines.append(f"    Мысль: {step.get('thought')}")
        if observation:
            lines.append(
                f"    Observation: status={observation.get('status')}, "
                f"{fmt_return_value(observation.get('return_value'))}"
            )
        else:
            lines.append("    Observation: отсутствует")
        formal = step.get("metadata", {}).get("formal_statement")
        if formal:
            lines.append(f"    Formal: {formal}")
    lines.append("")

    lines.append("Верификация:")
    lines.append(f"  Статус: {result.get('verifier_status')}")
    for entry in result.get("formal_trace", []):
        lines.append(
            f"    - {entry.get('step_id')}: {entry.get('verification_status')} "
            f"({entry.get('formal_statement')})"
        )
    if result.get("verifier_report", {}).get("issues"):
        lines.append("    Проблемы:")
        for issue in result["verifier_report"]["issues"]:
            lines.append(
                f"      • {issue.get('step_id')}: {issue.get('reason')}"
            )
    else:
        lines.append("    Нет проблем.")
    lines.append("")

    final = result.get("final_answer", {})
    units = final.get("units")
    units_str = f" {units}" if units else ""
    lines.append(
        f"Финальный ответ: {final.get('value')}{units_str} "
        f"(confidence={final.get('confidence')})"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    run_cli()
