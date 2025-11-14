import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from web.agent_service import TaskManager


app = Flask(__name__, static_folder="static", template_folder="templates")
task_manager = TaskManager(Path(__file__).with_name("task_history.json"))


def _error(message: str, status: int = 400):
	return jsonify({"error": message}), status


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/tasks/<task_id>")
def task_view(task_id: str):
	return render_template("task.html", task_id=task_id)


@app.route("/api/tasks", methods=["GET"])
def list_tasks():
	return jsonify(task_manager.list_tasks())


@app.route("/api/tasks/<task_id>", methods=["GET"])
def get_task(task_id: str):
	task = task_manager.get_task(task_id)
	if not task:
		return _error("Задача не найдена.", 404)
	return jsonify(task)


@app.route("/api/tasks/<task_id>", methods=["DELETE"])
def delete_task(task_id: str):
	try:
		task_manager.delete_task(task_id)
	except ValueError as exc:
		return _error(str(exc), 404)
	return jsonify({"status": "deleted"})


@app.route("/api/tasks", methods=["POST"])
def create_task():
	payload = request.get_json(silent=True) or {}
	problem = (payload.get("problem") or "").strip()

	if not problem:
		return _error("Пожалуйста, укажите текст задачи.")

	try:
		task = task_manager.create_task(problem)
	except Exception as exc:
		return _error(f"Не удалось сгенерировать планы: {exc}", 500)

	return jsonify(task), 201


@app.route("/api/tasks/<task_id>/run", methods=["POST"])
def run_task(task_id: str):
	payload = request.get_json(silent=True) or {}
	plan_id = payload.get("plan_id")
	wolfram_key = payload.get("wolfram_key") or os.getenv("WOLFRAM_API_KEY")
	long_poll = payload.get("long_poll", False)

	try:
		result = task_manager.run_task(task_id, plan_id, wolfram_key, long_poll)
	except ValueError as exc:
		return _error(str(exc), 404)
	except Exception as exc:
		return _error(f"Ошибка выполнения: {exc}", 500)

	return jsonify(result)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
