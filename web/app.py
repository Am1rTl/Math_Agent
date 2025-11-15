import os
import sys
from pathlib import Path

import json
import time
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, make_response

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from web.agent_service import TaskManager


app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
task_manager = TaskManager(Path(__file__).with_name("task_history.json"))


def _error(message: str, status: int = 400):
	return jsonify({"error": message}), status


def login_required(f):
	@wraps(f)
	def decorated_function(*args, **kwargs):
		token = request.cookies.get('token')
		if not token:
			return redirect(url_for('login'))
		try:
			jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
		except jwt.ExpiredSignatureError:
			return redirect(url_for('login'))
		except jwt.InvalidTokenError:
			return redirect(url_for('login'))
		return f(*args, **kwargs)
	return decorated_function


@app.route("/login")
def login():
	return render_template("login.html")


@app.route("/api/login", methods=["POST"])
def api_login():
	payload = request.get_json(silent=True) or {}
	username = payload.get("username")
	password = payload.get("password")

	if username == "admin" and password == "admin":
		token = jwt.encode({
			'user': username,
			'exp': datetime.utcnow() + timedelta(hours=24)
		}, app.config['SECRET_KEY'], algorithm='HS256')
		response = make_response(jsonify({"token": token}))
		response.set_cookie('token', token, httponly=True, secure=False, samesite='Lax')
		return response
	else:
		return _error("Неверные учетные данные", 401)


@app.route("/")
@login_required
def index():
	return render_template("index.html")


@app.route("/tasks/<task_id>")
@login_required
def task_view(task_id: str):
	return render_template("task.html", task_id=task_id)


@app.route("/api/tasks", methods=["GET"])
@login_required
def list_tasks():
	return jsonify(task_manager.list_tasks())


@app.route("/api/tasks/<task_id>/stream")
@login_required
def stream_task(task_id: str):
	def event_stream():
		last_log_count = 0
		while True:
			task = task_manager.get_task(task_id)
			if not task:
				break

			# Stream new log entries
			logs = task.get("progress_log", [])
			if len(logs) > last_log_count:
				for i in range(last_log_count, len(logs)):
					yield f"event: log_entry\ndata: {json.dumps(logs[i])}\n\n"
				last_log_count = len(logs)

			# Send a heartbeat to keep the connection alive
			yield "event: heartbeat\ndata: \n\n"

			if task.get("status") in ("completed", "failed"):
				yield f"event: task_complete\ndata: {json.dumps(task)}\n\n"
				break

			time.sleep(0.5)

	return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/tasks/<task_id>", methods=["GET"])
@login_required
def get_task(task_id: str):
	task = task_manager.get_task(task_id)
	if not task:
		return _error("Задача не найдена.", 404)
	return jsonify(task)


@app.route("/api/tasks/<task_id>", methods=["DELETE"])
@login_required
def delete_task(task_id: str):
	try:
		task_manager.delete_task(task_id)
	except ValueError as exc:
		return _error(str(exc), 404)
	return jsonify({"status": "deleted"})


@app.route("/api/tasks", methods=["POST"])
@login_required
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
@login_required
def run_task(task_id: str):
	payload = request.get_json(silent=True) or {}
	plan_id = payload.get("plan_id")
	wolfram_key = payload.get("wolfram_key") or os.getenv("WOLFRAM_API_KEY")

	try:
		result = task_manager.run_task(task_id, plan_id, wolfram_key)
	except ValueError as exc:
		return _error(str(exc), 404)
	except Exception as exc:
		return _error(f"Ошибка выполнения: {exc}", 500)

	return jsonify(result)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)