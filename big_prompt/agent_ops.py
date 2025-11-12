from __future__ import annotations

import json
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from settings import AppSettings, get_settings


def _tokenize(text: str) -> Counter:
	tokens = [token.lower() for token in json.dumps(text, ensure_ascii=False).replace("\n", " ").split() if token]
	return Counter(tokens)


class SchemaValidator:
	def __init__(self, settings: Optional[AppSettings] = None):
		self.settings = settings or get_settings()
		self.schema = json.loads(self.settings.finish_schema_path.read_text(encoding="utf-8"))

	def validate_finish(self, action: Dict[str, Any]) -> Tuple[bool, List[str]]:
		errors: List[str] = []
		if not isinstance(action, dict):
			return False, ["Action is not a dict"]
		if action.get("type") != "FINISH":
			return False, ["Action type must be FINISH"]
		payload = action.get("payload")
		if not isinstance(payload, dict):
			return False, ["payload must be object"]
		for field in ("answer", "values", "trace_id"):
			if field not in payload:
				errors.append(f"payload.{field} is required")
		values = payload.get("values", {})
		if not isinstance(values, dict):
			errors.append("payload.values must be object")
		else:
			if "a" not in values or not isinstance(values["a"], list) or not values["a"]:
				errors.append("payload.values.a must be non-empty list")
			elif not all(isinstance(v, (int, float)) for v in values["a"]):
				errors.append("payload.values.a must contain only numbers")
		trace_id = payload.get("trace_id")
		if not isinstance(trace_id, str) or len(trace_id) < 6:
			errors.append("payload.trace_id must be string length>=6")
		confidence = payload.get("confidence")
		if confidence is not None and not (0.0 <= confidence <= 1.0):
			errors.append("payload.confidence must be between 0 and 1")
		return len(errors) == 0, errors


class FewShotRetriever:
	def __init__(self, data_path: Optional[Path] = None):
		settings = get_settings()
		path = data_path or Path("data/few_shot_examples.json")
		if not path.exists():
			path = settings.eval_set_path
		self.examples: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
		self.embeddings = [(example, _tokenize(example.get("statement", ""))) for example in self.examples]

	def most_relevant(self, query: str, limit: int, topic_hint: Optional[str] = None) -> List[Dict[str, Any]]:
		query_tokens = _tokenize(query)

		def similarity(counter_a: Counter, counter_b: Counter) -> float:
			intersection = sum((counter_a & counter_b).values())
			if intersection == 0:
				return 0.0
			norm = math.sqrt(sum(v * v for v in counter_a.values())) * math.sqrt(sum(v * v for v in counter_b.values()))
			return intersection / norm if norm else 0.0

		scored = []
		for example, emb in self.embeddings:
			if topic_hint and example.get("topic") and topic_hint not in example.get("topic"):
				continue
			score = similarity(query_tokens, emb)
			scored.append((score, example))
		scored.sort(key=lambda item: item[0], reverse=True)
		return [example for score, example in scored[:limit] if score > 0]


@dataclass
class RoutingDecision:
	model: str
	temperature: float
	reason: str
	system: str


class RoutingManager:
	def __init__(self, settings: Optional[AppSettings] = None):
		self.settings = settings or get_settings()

	def score(self, text: str) -> float:
		complexity = len(text.split())
		penalty = 0.0
		if any(keyword in text.lower() for keyword in ["система", "неравенств", "параметр"]):
			penalty += 0.2
		return min(1.0, 0.3 + complexity / 400.0 + penalty)

	def route(self, text: str, role: str = "generic") -> RoutingDecision:
		score = self.score(text)
		if score < 0.5:
			return RoutingDecision(
				model=self.settings.models.system1_model,
				temperature=self.settings.models.system1_temperature,
				reason=f"score={score:.2f} -> system1",
				system="system1",
			)
		temperature = self.settings.models.system2_temperature
		if role == "planner":
			temperature = self.settings.models.planner_temperature
		elif role in {"executor", "formalizer"}:
			temperature = self.settings.models.executor_temperature
		elif role == "verifier":
			temperature = self.settings.models.verifier_temperature
		return RoutingDecision(
			model=self.settings.models.system2_model,
			temperature=temperature,
			reason=f"score={score:.2f} -> system2",
			system="system2",
		)


class LMJudge:
	def __init__(self, call_model: Callable[..., str], settings: Optional[AppSettings] = None):
		self.settings = settings or get_settings()
		self.call_model = call_model
		self.system_prompt = (
			"Ты --- строгий судья. Проверяй, найдены ли числовые значения параметра 'a', "
			"нет ли текстовых описаний вместо чисел, и согласуются ли ответы с задачей."
		)

	def _judge_once(self, problem_text: str, answer: Dict[str, Any], shuffle: bool = False) -> Dict[str, Any]:
		instructions = [
			"Проверь, что values.a непустой список чисел.",
			"Убедись, что ответ согласуется с постановкой задачи.",
			"Верни JSON с полями score (0-1) и verdict (PASS/FAIL) и rationale.",
		]
		if shuffle:
			instructions = list(reversed(instructions))
		user_prompt = json.dumps({
			"problem": problem_text,
			"answer": answer,
			"instructions": instructions
		}, ensure_ascii=False, indent=2)
		response = self.call_model(
			self.system_prompt,
			user_prompt,
			role="judge"
		)
		try:
			return json.loads(response)
		except Exception:
			return {"score": 0.0, "verdict": "FAIL", "rationale": "Invalid JSON from judge"}

	def evaluate(self, problem_text: str, answer: Dict[str, Any]) -> Dict[str, Any]:
		first = self._judge_once(problem_text, answer, shuffle=False)
		second = self._judge_once(problem_text, answer, shuffle=True)
		score = (first.get("score", 0.0) + second.get("score", 0.0)) / 2.0
		verdict = "PASS" if (
			first.get("verdict") == "PASS" and
			second.get("verdict") == "PASS" and
			score >= self.settings.thresholds.judge_pass_score
		) else "FAIL"
		return {
			"score": score,
			"verdict": verdict,
			"first": first,
			"second": second
		}


@dataclass
class AgentMetrics:
	successful_tasks: int = 0
	total_tasks: int = 0
	total_latency: float = 0.0
	token_usage: int = 0

	def record_task(self, success: bool, latency_s: float, tokens: int = 0):
		self.total_tasks += 1
		self.total_latency += latency_s
		self.token_usage += tokens
		if success:
			self.successful_tasks += 1

	@property
	def success_rate(self) -> float:
		return (self.successful_tasks / self.total_tasks) if self.total_tasks else 0.0

	@property
	def avg_latency(self) -> float:
		return (self.total_latency / self.total_tasks) if self.total_tasks else 0.0


class HumanEscalationQueue:
	def __init__(self, settings: Optional[AppSettings] = None):
		self.settings = settings or get_settings()
		self.path = self.settings.human_queue_path
		self.path.parent.mkdir(parents=True, exist_ok=True)

	def enqueue(self, reason: str, context: Dict[str, Any]):
		record = {
			"id": uuid.uuid4().hex,
			"timestamp": time.time(),
			"reason": reason,
			"context": context
		}
		with self.path.open("a", encoding="utf-8") as fp:
			fp.write(json.dumps(record, ensure_ascii=False) + "\n")
