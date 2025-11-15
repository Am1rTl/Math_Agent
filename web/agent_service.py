import json
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from web.agent_core import (
	InputFormalizer,
	MATHAGENTVL,
	Plan,
	ProblemObject,
	ToTPlanner
)


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
			return

		with self.history_path.open("r", encoding="utf-8") as fh:
			try:
				data = json.load(fh)
				for task in data:
					self._tasks[task["id"]] = task
				self._resume_pending_generations()
			except json.JSONDecodeError:
				# Corrupted history shouldn't block the service; start fresh.
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
			
			# Также возобновляем "зависшие" запущенные задачи
			# (Хотя в данном случае поток будет потерян, 
			# но для простоты мы просто сбросим статус)
			if task.get("status") == "running":
				task["status"] = "failed"
				task["error"] = "Выполнение было прервано перезапуском сервера."


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
		public["execution_trace"] = public.get("execution_trace") or []
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
			task = self._tasks.get(task_id)
			return self._public_view(task) if task else None

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

	def run_task(self, task_id: str, plan_id: Optional[str], wolfram_key: Optional[str]) -> Dict[str, Any]:
		with self._lock:
			task = self._tasks.get(task_id)
			if not task:
				raise ValueError("Задача не найдена.")

			if not task.get("plans") or not task.get("problem_object"):
				raise ValueError("Планы ещё формируются. Подождите немного и попробуйте снова.")
			
			if task_id in self._execution_threads:
				raise ValueError("Задача уже выполняется.")

		best_plan_id = self._choose_best_plan(task)
		if best_plan_id is None:
			raise ValueError("Нет доступных планов для выполнения.")
		
		selected_plan_id = plan_id or best_plan_id

		with self._lock:
			task["selected_plan_id"] = selected_plan_id
			task["status"] = "running"
			task["updated_at"] = time.time()
			# Очищаем логи и трассировку от предыдущих запусков
			task["execution_trace"] = [] 
			task["formal_trace"] = []
			task["final_answer"] = None
			task["progress_log"] = []
			task["error"] = None
			self._persist()

		self._append_progress(task_id, {
			"type": "run_start",
			"message": f"Запуск выполнения плана {selected_plan_id}"
		})

		# Запускаем выполнение в отдельном потоке
		thread = threading.Thread(
			target=self._run_task_async,
			args=(task_id, selected_plan_id, wolfram_key),
			daemon=True
		)
		thread.start()
		
		with self._lock:
			self._execution_threads[task_id] = thread
			public_task = self._public_view(task)

		# Немедленно возвращаем задачу в статусе "running"
		return {"task": public_task, "result": {"status": "started"}}

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

		try:
			problem_object = formalizer.formalize(problem_text)
			plans = planner.propose_plans(problem_object)
			plans.sort(key=lambda plan: getattr(plan, "heuristic_score", 0.0), reverse=True)
			plan_dicts = [self._plan_to_dict(plan) for plan in plans]
			best_plan_id = plan_dicts[0]["id"] if plan_dicts else None

			with self._lock:
				task = self._tasks.get(task_id)
				if not task:
					return
				task["problem_object"] = self._problem_to_dict(problem_object)
				task["plans"] = plan_dicts
				task["selected_plan_id"] = best_plan_id
				task["status"] = "plans-ready"
				task["updated_at"] = time.time()
				task["is_generating"] = False
				task["error"] = None
				self._persist()

			self._append_progress(task_id, {
				"type": "plan_generation_complete",
				"message": "Генерация планов завершена."
			})
		except Exception as exc:
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
			self._plan_threads.pop(task_id, None)

	def _execute_with_plan(
		self,
		task: Dict[str, Any],
		plan_id: str,
		wolfram_api_key: Optional[str],
		progress_callback=None,
		execution_callback=None # <-- Добавлено
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
				progress_callback=progress_callback,
				execution_callback=execution_callback # <-- Передано
			)

	# +++ НОВЫЙ МЕТОД: Цель для потока выполнения +++
	def _run_task_async(self, task_id: str, plan_id: str, wolfram_key: Optional[str]):
		task = None
		with self._lock:
			task = self._tasks.get(task_id)
			if not task:
				return 

		def progress_callback(event: Dict[str, Any]):
			self._append_progress(task_id, event)
		
		# +++ НОВЫЙ CALLBACK: Для 'execution_trace' +++
		def execution_callback(trace_entry: Dict[str, Any]):
			self._append_execution_trace(task_id, trace_entry)

		try:
			# Это блокирующая операция
			result = self._execute_with_plan(
				task, 
				plan_id, 
				wolfram_key, 
				progress_callback,
				execution_callback # <-- Передаем новый callback
			)
			
			# Как только выполнение завершено, обновляем финальный статус
			with self._lock:
				task = self._tasks.get(task_id) # Получаем свежую версию
				if not task:
					return 
				
				task["status"] = "completed" if result.get("final_answer") else "failed"
				task["result_preview"] = result.get("final_answer")
				task["final_answer"] = result.get("final_answer")
				# Синхронизируем с финальной версией трассировки
				task["execution_trace"] = result.get("execution_trace", []) 
				task["formal_trace"] = result.get("formal_trace", [])
				task["updated_at"] = time.time()
				self._persist()

			self._append_progress(task_id, {
				"type": "run_complete",
				"message": "Выполнение задачи завершено."
			})

		except Exception as exc:
			# В случае ошибки во время выполнения
			with self._lock:
				task = self._tasks.get(task_id)
				if not task:
					return
				task["status"] = "failed"
				task["result_preview"] = {"answer": f"Ошибка выполнения: {exc}"}
				task["error"] = str(exc)
				task["updated_at"] = time.time()
				self._persist()
			
			self._append_progress(task_id, {
				"type": "run_error",
				"message": f"Ошибка выполнения: {exc}"
			})
		finally:
			# Убираем поток из списка активных
			with self._lock:
				self._execution_threads.pop(task_id, None)

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

	# +++ НОВЫЙ МЕТОД: Сохранение шагов 'execution_trace' +++
	def _append_execution_trace(self, task_id: str, trace_entry: Dict[str, Any]) -> None:
		if not trace_entry:
			return
		with self._lock:
			task = self._tasks.get(task_id)
			if not task:
				return
			trace_log = task.setdefault("execution_trace", [])
			
			# Простой append, т.к. каждый шаг уникален
			trace_log.append(trace_entry)
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
