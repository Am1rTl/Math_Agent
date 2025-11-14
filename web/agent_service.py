import json
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from web.agent_core import (
	InputFormalizer,
	MATHAGENTVL,
	Plan,
	ProblemObject,
	ToTPlanner
)

logging.basicConfig(filename='agent_service.log', level=logging.INFO)


class TaskManager:
	"""Wraps the DeepSeek agent, exposing plan previews and task history for the web UI."""

	def __init__(self, history_path: Path):
		self.history_path = history_path
		self.history_path.parent.mkdir(parents=True, exist_ok=True)

		self._lock = threading.Lock()
		self._tasks: Dict[str, Dict[str, Any]] = {}

		self.agent = MATHAGENTVL()
		self._plan_threads: Dict[str, threading.Thread] = {}
		self._execution_threads: Dict[str, threading.Thread] = {}
		self._load_history()

	def _load_history(self) -> None:
		if not self.history_path.exists():
			logging.info("History file not found.")
			return

		with self.history_path.open("r", encoding="utf-8") as fh:
			try:
				data = json.load(fh)
				for task in data:
					self._tasks[task["id"]] = task
				logging.info(f"Loaded {len(self._tasks)} tasks from history.")
				self._resume_pending_generations()
			except json.JSONDecodeError:
				# Corrupted history shouldn't block the service; start fresh.
				logging.warning("History file is corrupted. Starting fresh.")
				self._tasks = {}
			except Exception as e:
				logging.error(f"Error loading history: {e}")
				self._tasks = {}

	def _persist(self) -> None:
		with self.history_path.open("w", encoding="utf-8") as fh:
			json.dump(list(self._tasks.values()), fh, ensure_ascii=False, indent=2)

	def _plan_to_dict(self, plan: Plan) -> Dict[str, Any]:
		return {
			"id": plan.id,
			"summary": plan.summary,
			"steps": plan.steps,
			"estimated_complexity": plan.estimated_complexity,
			"estimated_tooling": plan.estimated_tooling,
			"heuristic_score": plan.heuristic_score,
			"rationale": plan.rationale
		}

	def _problem_to_dict(self, problem: ProblemObject) -> Dict[str, Any]:
		return {
			"id": problem.id,
			"statement": problem.statement,
			"entities": deepcopy(problem.entities),
			"constraints": deepcopy(problem.constraints),
			"goals": deepcopy(problem.goals),
			"hypotheses": deepcopy(problem.hypotheses),
			"irrelevant": deepcopy(problem.irrelevant),
			"notes": problem.notes
		}

	def _resume_pending_generations(self) -> None:
		for task_id, task in list(self._tasks.items()):
			if task.get("status") == "thinking" and task.get("is_generating"):
				thread = threading.Thread(
					target=self._generate_plans_async,
					args=(task_id, task.get("problem_text", "")),
					daemon=True
				)
				thread.start()
				self._plan_threads[task_id] = thread

	def _public_view(self, task: Dict[str, Any]) -> Dict[str, Any]:
		public = {
			k: task.get(k)
			for k in [
				"id",
				"problem_text",
				"created_at",
				"updated_at",
				"status",
				"plans",
				"selected_plan_id",
				"final_answer",
				"result_preview",
				"execution_trace",
				"formal_trace",
				"progress_log",
				"is_generating",
				"error"
			]
		}
		public["problem_object"] = task.get("problem_object") or {}
		public["progress_log"] = public.get("progress_log") or []
		return public

	def list_tasks(self) -> List[Dict[str, Any]]:
		with self._lock:
			return sorted(
				[self._public_view(task) for task in self._tasks.values()],
				key=lambda item: item.get("created_at", 0),
				reverse=True
			)

	def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
		with self._lock:
			logging.info(f"Getting task {task_id}")
			task = self._tasks.get(task_id)
			if not task:
				logging.warning(f"Task {task_id} not found.")
				return None
			return self._public_view(task)

	def create_task(self, problem_text: str) -> Dict[str, Any]:
		task_id = str(uuid.uuid4())
		timestamp = time.time()

		task_payload = {
			"id": task_id,
			"problem_text": problem_text,
			"created_at": timestamp,
			"updated_at": timestamp,
			"status": "thinking",
			"problem_object": None,
			"plans": [],
			"selected_plan_id": None,
			"result_preview": None,
			"final_answer": None,
			"execution_trace": [],
			"formal_trace": [],
			"progress_log": [],
			"is_generating": True,
			"error": None
		}

		with self._lock:
			self._tasks[task_id] = task_payload
			self._persist()

		thread = threading.Thread(
			target=self._generate_plans_async,
			args=(task_id, problem_text),
			daemon=True
		)
		thread.start()
		self._plan_threads[task_id] = thread

		return self._public_view(task_payload)

	def run_task(
		self,
		task_id: str,
		plan_id: Optional[str],
		wolfram_key: Optional[str],
		long_poll: bool = False
	) -> Dict[str, Any]:
		with self._lock:
			task = self._tasks.get(task_id)
			if not task:
				raise ValueError("Задача не найдена.")

			if not task.get("plans") or not task.get("problem_object"):
				raise ValueError("Планы ещё формируются. Подождите немного и попробуйте снова.")

		best_plan_id = self._choose_best_plan(task)
		if best_plan_id is None:
			raise ValueError("Нет доступных планов для выполнения.")
		plan_id = best_plan_id

		with self._lock:
			task["selected_plan_id"] = plan_id
			task["status"] = "running"
			task["updated_at"] = time.time()
			self._persist()

		self._append_progress(task_id, {
			"type": "run_start",
			"message": f"Запуск выполнения плана {plan_id}"
		})

		def _execute():
			def progress_callback(event: Dict[str, Any]):
				self._append_progress(task_id, event)

			try:
				result = self._execute_with_plan(task, plan_id, wolfram_key, progress_callback)
			except Exception as exc:
				with self._lock:
					task["status"] = "failed"
					task["result_preview"] = {"answer": f"Ошибка выполнения: {exc}"}
					task["updated_at"] = time.time()
					self._persist()
				self._append_progress(task_id, {
					"type": "run_error",
					"message": f"Ошибка выполнения: {exc}"
				})
				return

			with self._lock:
				task["status"] = "completed" if result.get("final_answer") else "failed"
				task["result_preview"] = result.get("final_answer")
				task["final_answer"] = result.get("final_answer")
				task["execution_trace"] = result.get("execution_trace", [])
				task["formal_trace"] = result.get("formal_trace", [])
				task["updated_at"] = time.time()
				self._persist()

			self._append_progress(task_id, {
				"type": "run_complete",
				"message": "Выполнение задачи завершено."
			})

		execution_thread = threading.Thread(target=_execute, daemon=True)
		execution_thread.start()
		self._execution_threads[task_id] = execution_thread

		if long_poll:
			execution_thread.join()

		with self._lock:
			public_task = self._public_view(task)
		return {"task": public_task}

	def delete_task(self, task_id: str) -> None:
		with self._lock:
			if task_id not in self._tasks:
				raise ValueError("Задача не найдена.")
			del self._tasks[task_id]
			self._persist()

	def _generate_plans_async(self, task_id: str, problem_text: str) -> None:
		formalizer = InputFormalizer()
		planner = ToTPlanner()

		self._append_progress(task_id, {
			"type": "plan_generation_start",
			"message": "Генерация планов запущена."
		})
		logging.info(f"[{task_id}] Plan generation started.")

		try:
			logging.info(f"[{task_id}] Formalizing input...")
			problem_object = formalizer.formalize(problem_text)
			logging.info(f"[{task_id}] Input formalized. Proposing plans...")
			plans = planner.propose_plans(problem_object)
			logging.info(f"[{task_id}] Plans proposed. Sorting and converting to dicts...")
			plans.sort(key=lambda plan: getattr(plan, "heuristic_score", 0.0), reverse=True)
			plan_dicts = [self._plan_to_dict(plan) for plan in plans]
			best_plan_id = plan_dicts[0]["id"] if plan_dicts else None
			logging.info(f"[{task_id}] Plans processed. Updating task state...")

			with self._lock:
				task = self._tasks.get(task_id)
				if not task:
					logging.warning(f"[{task_id}] Task not found after plan generation.")
					return
				task["problem_object"] = self._problem_to_dict(problem_object)
				task["plans"] = plan_dicts
				task["selected_plan_id"] = best_plan_id
				task["status"] = "plans-ready"
				task["updated_at"] = time.time()
				task["is_generating"] = False
				task["error"] = None
				self._persist()
			logging.info(f"[{task_id}] Task state updated.")

			self._append_progress(task_id, {
				"type": "plan_generation_complete",
				"message": "Генерация планов завершена."
			})
		except Exception as exc:
			logging.error(f"[{task_id}] Error during plan generation: {exc}", exc_info=True)
			with self._lock:
				task = self._tasks.get(task_id)
				if not task:
					return
				task["status"] = "failed"
				task["error"] = str(exc)
				task["updated_at"] = time.time()
				task["is_generating"] = False
				self._persist()

			self._append_progress(task_id, {
				"type": "plan_generation_error",
				"message": f"Ошибка генерации планов: {exc}"
			})
		finally:
			logging.info(f"[{task_id}] Plan generation thread finished.")
			self._plan_threads.pop(task_id, None)

	def _execute_with_plan(
		self,
		task: Dict[str, Any],
		plan_id: str,
		wolfram_api_key: Optional[str],
		progress_callback=None
	) -> Dict[str, Any]:
		ordered_plans = self._reorder_plans(task["plans"], plan_id)
		plan_objects = [Plan(**plan) for plan in ordered_plans]
		problem_object = ProblemObject(**task["problem_object"])

		def fake_formalize(_: str) -> ProblemObject:
			return problem_object

		def fake_propose(_: ProblemObject) -> List[Plan]:
			return plan_objects

		self.agent.tot_planner.pruned_branches = set()

		with self._patch(self.agent.input_formalizer, "formalize", fake_formalize), \
			 self._patch(self.agent.tot_planner, "propose_plans", fake_propose):
			return self.agent.solve_problem(
				problem_text=task["problem_text"],
				wolfram_api_key=wolfram_api_key,
				progress_callback=progress_callback
			)

	@staticmethod
	def _reorder_plans(plans: List[Dict[str, Any]], plan_id: str) -> List[Dict[str, Any]]:
		selected = [plan for plan in plans if plan["id"] == plan_id]
		others = [plan for plan in plans if plan["id"] != plan_id]
		return selected + others

	@staticmethod
	def _choose_best_plan(task: Dict[str, Any]) -> Optional[str]:
		plans = task.get("plans") or []
		if not plans:
			return None
		best_plan = max(plans, key=lambda plan: plan.get("heuristic_score", 0.0))
		return best_plan.get("id")

	def _append_progress(self, task_id: str, event: Optional[Dict[str, Any]]) -> None:
		if not event:
			return
		event.setdefault("timestamp", time.time())
		with self._lock:
			task = self._tasks.get(task_id)
			if not task:
				return
			progress_log = task.setdefault("progress_log", [])
			progress_log.append(event)
			task["updated_at"] = time.time()
			self._persist()

	@contextmanager
	def _patch(self, obj: Any, attr: str, replacement):
		original = getattr(obj, attr)
		setattr(obj, attr, replacement)
		try:
			yield
		finally:
			setattr(obj, attr, original)
