import { createTaskWindow, ensurePlanSelection } from "./ui.js";

const workspaceEl = document.getElementById("single-task-workspace");
const emptyStateEl = document.getElementById("single-task-empty");
const backButton = document.getElementById("back-home");
const refreshButton = document.getElementById("refresh-task");
const deleteButton = document.getElementById("delete-task");
const taskId = document.body.dataset.taskId;

const toast = document.createElement("div");
toast.className = "toast";
document.body.appendChild(toast);

const showToast = (message, variant = "info") => {
	toast.textContent = message;
	toast.dataset.variant = variant;
	toast.classList.add("visible");
	setTimeout(() => toast.classList.remove("visible"), 3200);
};

const state = {
	task: null,
	isRunning: false,
	errorShown: false
};

const fetchTask = async (silent = false) => {
	try {
		const response = await fetch(`/api/tasks/${taskId}`);
		if (!response.ok) {
			throw new Error("Не удалось загрузить задачу.");
		}
		state.task = await response.json();
		render();
	} catch (error) {
		if (!silent) {
			showToast(error.message, "error");
		}
	}
};

const render = () => {
	workspaceEl.innerHTML = "";
	if (!state.task) {
		emptyStateEl.style.display = "flex";
		return;
	}

	emptyStateEl.style.display = "none";
	if (state.task.status === "failed" && state.task.error && !state.errorShown) {
		showToast(state.task.error, "error");
		state.errorShown = true;
	} else if (state.task.status !== "failed") {
		state.errorShown = false;
	}
	ensurePlanSelection(state.task);

	const node = createTaskWindow(state.task, {
		isTaskRunning: () => state.isRunning,
		onRunTask: handleRunTask,
		onSelectPlan: null
	});

	workspaceEl.appendChild(node);
};

const handleRunTask = async taskIdentifier => {
	if (!state.task || state.isRunning) return;
	if (!state.task.plans?.length) {
		showToast("Планы ещё формируются. Подождите немного.", "warn");
		return;
	}

	state.isRunning = true;
	render();

	const response = await fetch(`/api/tasks/${taskIdentifier}/run`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ plan_id: state.task.selected_plan_id })
	});

	if (!response.ok) {
		const payload = await response.json();
		showToast(payload.error || "Ошибка выполнения плана.", "error");
		state.isRunning = false;
		render();
		return;
	}

	const payload = await response.json();
	state.task = payload.task;
	state.isRunning = false;
	render();
};

const handleDelete = async () => {
	if (!state.task) return;
	try {
		const response = await fetch(`/api/tasks/${state.task.id}`, { method: "DELETE" });
		if (!response.ok) {
			const payload = await response.json();
			throw new Error(payload.error || "Не удалось удалить задачу.");
		}
		showToast("Задача удалена.", "success");
		setTimeout(() => (window.location.href = "/"), 600);
	} catch (error) {
		showToast(error.message, "error");
	}
};

const initControls = () => {
	backButton?.addEventListener("click", () => (window.location.href = "/"));
	refreshButton?.addEventListener("click", () => fetchTask(false));
	deleteButton?.addEventListener("click", handleDelete);
};

initControls();
fetchTask(false);
setInterval(() => fetchTask(true), 2000);
