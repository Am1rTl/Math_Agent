import { appendLogEntry, createTaskWindow, ensurePlanSelection } from "./ui.js";

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
	errorShown: false,
	isCompleted: false
};

const fetchInitialTask = async () => {
	try {
		const response = await fetch(`/api/tasks/${taskId}`);
		if (!response.ok) throw new Error("Не удалось загрузить задачу.");
		state.task = await response.json();
		render();
	} catch (err) {
		showToast(err.message, "error");
	}
};

const render = () => {
	if (!workspaceEl || !state.task) {
		emptyStateEl.style.display = "flex";
		return;
	}

	const currentlyOpen = new Set();
	workspaceEl.querySelectorAll(".step-item.open").forEach(el => {
		const key = el.dataset.key;
		if (key) currentlyOpen.add(key);
	});

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

	node.querySelectorAll(".step-item").forEach(el => {
		if (currentlyOpen.has(el.dataset.key)) {
			el.classList.add("open");
		}
	});

	workspaceEl.appendChild(node);

	// Auto-scroll progress feed to bottom after render
	const progressList = node.querySelector('.progress-list');
	if (progressList) {
		progressList.scrollTop = progressList.scrollHeight;
	}
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

const initEventSource = () => {
	const eventSource = new EventSource(`/api/tasks/${taskId}/stream`);

	eventSource.addEventListener("log_entry", event => {
		const logEntry = JSON.parse(event.data);
		appendLogEntry(logEntry, workspaceEl);
		if (state.task && !state.task.progress_log) {
			state.task.progress_log = [];
		}
		state.task?.progress_log.push(logEntry);
	});

	eventSource.addEventListener("task_complete", event => {
		state.task = JSON.parse(event.data);
		state.isCompleted = true;
		render();
		eventSource.close();
	});

	eventSource.onerror = () => {
		showToast("Соединение с сервером потеряно. Попытка переподключения...", "error");
		setTimeout(() => {
			eventSource.close();
			initEventSource();
		}, 2000);
	};
};

const initControls = () => {
	backButton?.addEventListener("click", () => (window.location.href = "/"));
	refreshButton?.addEventListener("click", fetchInitialTask);
	deleteButton?.addEventListener("click", handleDelete);
};

const startAutoRefresh = () => {
	setInterval(() => {
		if (refreshButton && !state.isCompleted) {
			refreshButton.click();
		}
	}, 2000);
};

initControls();
fetchInitialTask();
initEventSource();
startAutoRefresh();
