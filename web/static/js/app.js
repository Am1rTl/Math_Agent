import { createTaskWindow, ensurePlanSelection } from "./ui.js";

const state = {
	tasks: [],
	loadingTasks: new Set()
};

const workspaceEl = document.getElementById("task-workspace");
const historyListEl = document.getElementById("history-list");
const formEl = document.getElementById("task-form");
const emptyStateEl = document.getElementById("workspace-empty");
const refreshHistoryBtn = document.getElementById("refresh-history");

const toast = document.createElement("div");
toast.className = "toast";
document.body.appendChild(toast);

const showToast = (message, variant = "info") => {
	toast.textContent = message;
	toast.dataset.variant = variant;
	toast.classList.add("visible");
	setTimeout(() => toast.classList.remove("visible"), 3200);
};

const fetchTasks = async () => {
	const res = await fetch("/api/tasks");
	const tasks = await res.json();
	state.tasks = tasks;
	render();
};

const createTask = async problem => {
	const response = await fetch("/api/tasks", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ problem })
	});

	if (!response.ok) {
		const payload = await response.json();
		throw new Error(payload.error || "Ошибка создания задачи.");
	}

	const task = await response.json();
	window.location.href = `/tasks/${task.id}`;
};

const runTask = async taskId => {
	const task = state.tasks.find(item => item.id === taskId);
	if (!task) return;

	if (!task.plans?.length) {
		showToast("Планы ещё формируются. Попробуйте чуть позже.", "warn");
		return;
	}

	state.loadingTasks.add(taskId);
	render();

	const response = await fetch(`/api/tasks/${taskId}/run`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ plan_id: task.selected_plan_id || null })
	});

	state.loadingTasks.delete(taskId);

	if (!response.ok) {
		const payload = await response.json();
		showToast(payload.error || "Ошибка выполнения плана.", "error");
		render();
		return;
	}

	const payload = await response.json();
	updateTask(payload.task);
	render();
};

const updateTask = updatedTask => {
	state.tasks = state.tasks.map(task => (task.id === updatedTask.id ? updatedTask : task));
};

const render = () => {
	renderWorkspace();
	renderHistory();
};

const renderWorkspace = () => {
	workspaceEl.innerHTML = "";
	if (!state.tasks.length) {
		workspaceEl.appendChild(emptyStateEl);
		emptyStateEl.style.display = "block";
		return;
	}
	emptyStateEl.style.display = "none";
	state.tasks.forEach(task => {
		ensurePlanSelection(task);
		const windowNode = createTaskWindow(task, {
			isTaskRunning: id => state.loadingTasks.has(id),
			onRunTask: runTask,
			onSelectPlan: null
		});
		workspaceEl.appendChild(windowNode);
	});
};

const renderHistory = () => {
	historyListEl.innerHTML = "";
	state.tasks.forEach(task => {
		const item = document.createElement("li");
		item.className = `history-item ${task.status || ""}`;
		item.innerHTML = `
			<div class="history-title">${task.problem_text.slice(0, 60)}${task.problem_text.length > 60 ? "…" : ""}</div>
			<div class="status-pill">${task.status || "неизвестно"}</div>
		`;
		item.addEventListener("click", () => {
			document.getElementById(`task-${task.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
		});
		const actions = document.createElement("div");
		actions.className = "history-actions";
		const deleteBtn = document.createElement("button");
		deleteBtn.className = "history-delete";
		deleteBtn.title = "Удалить задачу";
		deleteBtn.innerHTML = "&times;";
		deleteBtn.addEventListener("click", event => {
			event.stopPropagation();
			handleDeleteTask(task);
		});
		actions.appendChild(deleteBtn);
		item.appendChild(actions);
		historyListEl.appendChild(item);
	});
};


const deleteTaskRequest = async taskId => {
	const response = await fetch(`/api/tasks/${taskId}`, { method: "DELETE" });
	if (!response.ok) {
		const payload = await response.json();
		throw new Error(payload.error || "Не удалось удалить задачу.");
	}
};

const handleDeleteTask = async task => {
	try {
		await deleteTaskRequest(task.id);
		state.tasks = state.tasks.filter(item => item.id !== task.id);
		render();
		showToast("Задача удалена.", "success");
	} catch (error) {
		showToast(error.message, "error");
	}
};


formEl.addEventListener("submit", async event => {
	event.preventDefault();
	const problem = formEl.problem.value.trim();
	if (!problem) {
		showToast("Опишите задачу, чтобы продолжить.", "warn");
		return;
	}
	formEl.problem.disabled = true;
	try {
		await createTask(problem);
		formEl.reset();
		showToast("Планы готовы!", "success");
	} catch (error) {
		showToast(error.message, "error");
	} finally {
		formEl.problem.disabled = false;
	}
});

refreshHistoryBtn.addEventListener("click", fetchTasks);

fetchTasks().catch(err => {
	console.error(err);
	showToast("Не удалось загрузить историю.", "error");
});
