const createLoader = () => {
	const el = document.createElement("div");
	el.className = "loader";
	return el;
};

const createDropletLoader = () => {
	const el = document.createElement("div");
	el.className = "droplet-loader";
	return el;
};

const createThinkingView = task => {
	const card = document.createElement("div");
	card.className = "thinking-card";
	const message = document.createElement("div");
	message.className = "agent-message";

	const bubble = document.createElement("div");
	bubble.className = "bubble";
	const isPlanGeneration = task?.status === "thinking" || task?.is_generating;
	const primaryText = isPlanGeneration
		? "Агент генерирует планы. Это займёт всего пару мгновений..."
		: "Агент подключается к мыслям...";
	bubble.innerHTML = `<p>${primaryText}</p>`;

	message.append(bubble, createDropletLoader());
	const hint = document.createElement("p");
	hint.className = "thinking-hint";
	hint.textContent = isPlanGeneration
		? "Как только планы будут готовы — они появятся в workspace автоматически."
		: "Как только план будет готов — он появится здесь автоматически.";

	card.append(message, hint);
	return card;
};

const createErrorCard = task => {
	const card = document.createElement("div");
	card.className = "error-card";
	card.innerHTML = `
		<h3>Не удалось сгенерировать планы</h3>
		<p>${task?.error || "Агент столкнулся с ошибкой во время планирования."}</p>
		<span>Попробуй отредактировать формулировку задачи или создать её заново.</span>
	`;
	return card;
};

const createAgentStatusCard = (task, statusOverride = null) => {
	const card = document.createElement("div");
	const status = statusOverride || task.status;
	card.className = `agent-status-card ${status || ""}`.trim();

	const header = document.createElement("div");
	header.className = "agent-status-header";
	const title = document.createElement("h3");
	title.textContent = status === "running" ? "Агент думает..." : "Последнее рассуждение";
	header.appendChild(title);
	if (status === "running") {
		header.appendChild(createDropletLoader());
	}

	const body = document.createElement("div");
	body.className = "agent-status-body";
	const currentEntry = getCurrentEntry(task);
	const currentTitle = currentEntry ? `Шаг ${currentEntry.step_id}` : "Ожидание";
	const currentThought = currentEntry?.thought || "Агент готовится к вычислениям.";

	const label = document.createElement("div");
	label.className = "status-label";
	label.textContent = currentTitle;
	const text = document.createElement("p");
	text.textContent = currentThought;

	body.append(label, text);
	card.append(header, body);
	return card;
};

const getCurrentEntry = task => {
	const trace = task.execution_trace || [];
	return trace[trace.length - 1];
};

const planStepState = new Map();

const createPlanCard = (task, plan, onSelectPlan) => {
	const card = document.createElement("article");
	card.className = `plan-card${task.selected_plan_id === plan.id ? " selected" : ""}`;
	card.addEventListener("click", () => {
		if (typeof onSelectPlan === "function") {
			onSelectPlan(task.id, plan.id);
		}
	});

	const meta = document.createElement("div");
	meta.className = "plan-meta";
	meta.innerHTML = `
		<span>Оценка: ${(plan.heuristic_score || 0).toFixed(2)}</span>
		<span class="badge">${plan.estimated_complexity}</span>
	`;

	const summary = document.createElement("div");
	summary.className = "plan-summary";
	const summaryParts = splitStep(plan.summary, 0);
	summary.textContent = summaryParts.title || plan.summary;

	const stepsList = document.createElement("div");
	stepsList.className = "step-list";

	const openSteps = planStepState.get(plan.id) || new Set();
	planStepState.set(plan.id, openSteps);

	(plan.steps || []).forEach((step, idx) => {
		const stepItem = document.createElement("div");
		stepItem.className = "step-item";
		stepItem.dataset.key = `plan-${plan.id}-step-${idx}`;

		const toggle = document.createElement("button");
		toggle.className = "step-toggle";
		const parsed = splitStep(step, idx);
		const stepBody = document.createElement("div");
		stepBody.className = "step-body";
		stepBody.textContent = parsed.body;
		stepBody.style.maxHeight = "0px";
		stepBody.style.paddingBottom = "0";
		toggle.innerHTML = `
			<span class="step-title">${parsed.title}</span>
			<span class="step-chevron">›</span>
		`;
		const toggleBody = e => {
			e.stopPropagation();
			const isOpen = !stepItem.classList.contains("open");
			setPlanStepOpenState(stepItem, stepBody, openSteps, idx, isOpen);
		};
		toggle.addEventListener("click", toggleBody);
		stepItem.addEventListener("click", toggleBody);

		stepItem.append(toggle, stepBody);
		stepsList.appendChild(stepItem);
		// Apply saved open state after DOM insertion to measure height correctly.
		requestAnimationFrame(() => {
			const shouldBeOpen = openSteps.has(idx);
			if (shouldBeOpen) {
				setPlanStepOpenState(stepItem, stepBody, openSteps, idx, true, {
					skipSetUpdate: true,
					instant: true
				});
			}
		});
	});

	card.append(meta, summary, stepsList);
	return card;
};

const setPlanStepOpenState = (
	stepItem,
	stepBody,
	openSet,
	idx,
	isOpen,
	options = {}
) => {
	const { skipSetUpdate = false, instant = false } = options;
	const applyState = () => {
		if (isOpen) {
			stepItem.classList.add("open");
			stepBody.style.maxHeight = `${stepBody.scrollHeight}px`;
			stepBody.style.paddingBottom = "14px";
			if (!skipSetUpdate) openSet.add(idx);
		} else {
			stepItem.classList.remove("open");
			stepBody.style.maxHeight = "0px";
			stepBody.style.paddingBottom = "0";
			if (!skipSetUpdate) openSet.delete(idx);
		}
	};

	if (instant) {
		const previousTransition = stepBody.style.transition;
		stepBody.style.transition = "none";
		applyState();
		// Force reflow to apply styles before restoring transition
		// eslint-disable-next-line no-unused-expressions
		stepBody.offsetHeight;
		requestAnimationFrame(() => {
			stepBody.style.transition = previousTransition || "";
		});
	} else {
		applyState();
	}
};

const splitStep = (text = "", index = 0) => {
	const [title, ...rest] = text.split(":");
	const body = rest.join(":").trim() || text.trim();
	return {
		title: title?.trim() || `Шаг ${index + 1}`,
		body
	};
};

const createTimeline = task => {
	const timeline = document.createElement("ul");
	timeline.className = "timeline";

	const entries = task.execution_trace || [];
	if (!entries.length) {
		const placeholder = document.createElement("li");
		placeholder.className = "timeline-item";
		placeholder.innerHTML = `
			<h4>Ещё нет действий</h4>
			<p>Запусти план, чтобы увидеть трассировку ReAct агента.</p>
		`;
		timeline.appendChild(placeholder);
		return timeline;
	}

	entries.forEach(entry => {
		const actionText = entry.action ? entry.action.type : "—";
		const observationText = formatObservation(entry.observation);
		if (actionText !== "—" && observationText !== "—") {
			const item = document.createElement("li");
			item.className = "timeline-item";
			item.innerHTML = `
				<h4>${entry.step_id}</h4>
				<p><strong>Мысль:</strong> ${entry.thought || "—"}</p>
				<p><strong>Действие:</strong> ${actionText}</p>
				<p><strong>Наблюдение:</strong> ${observationText}</p>
			`;
			timeline.appendChild(item);
		}
	});

	return timeline;
};

const createFinalAnswerBlock = task => {
	const block = document.createElement("div");
	block.className = "final-answer";
	block.innerHTML = task.final_answer
		? `<h3>Финальный ответ</h3><p>${task.final_answer.answer || "—"}</p>`
		: `<h3>Финальный ответ</h3><p>Пока нет результатов</p>`;
	return block;
};

const createProgressFeed = (task, collapsed = false) => {
	const feed = document.createElement("section");
	feed.className = `progress-feed ${collapsed ? "collapsed" : ""}`.trim();

	const header = document.createElement("div");
	header.className = "panel-header";
	header.style.cursor = "pointer";

	const title = document.createElement("h3");
	title.textContent = "Консоль агента";

	const toggleBtn = document.createElement("button");
	toggleBtn.className = "panel-toggle-btn";
	toggleBtn.innerHTML = collapsed ? "▼" : "▲";

	header.append(title, toggleBtn);

	header.addEventListener("click", () => {
		const isCollapsed = feed.classList.toggle("collapsed");
		toggleBtn.innerHTML = isCollapsed ? "▼" : "▲";
	});

	feed.appendChild(header);

	const logs = (task.progress_log || []).slice(-200);
	if (!logs.length) {
		const placeholder = document.createElement("article");
		placeholder.className = "timeline-item";
		placeholder.innerHTML = `
			<h4>Ждём событий</h4>
			<p>Как только агент начнёт выводить мысли, они появятся здесь автоматически.</p>
		`;
		feed.appendChild(placeholder);
		return feed;
	}

	const list = document.createElement("div");
	list.className = "progress-list";
	logs.forEach(event => list.appendChild(createLogEntry(event)));

	feed.appendChild(list);
	return feed;
};

export const appendLogEntry = (event, container) => {
	const list = container.querySelector(".progress-list");
	if (list) {
		const placeholder = list.querySelector(".timeline-item");
		if (placeholder) placeholder.remove();
		list.appendChild(createLogEntry(event));
		list.scrollTop = list.scrollHeight;
	}
};

const createLogEntry = event => {
	const entry = document.createElement("div");
	entry.className = `progress-entry ${event.type || "log"}`;

	const meta = document.createElement("div");
	meta.className = "progress-meta";

	const time = document.createElement("span");
	time.className = "progress-time";
	time.textContent = formatTimestamp(event.timestamp);

	const type = document.createElement("span");
	type.className = "progress-type";
	type.textContent = event.type || "log";

	meta.append(time, type);

	const body = document.createElement("pre");
	body.className = "progress-message";
	body.textContent = event.message || "";

	entry.append(meta, body);
	return entry;
};

const formatObservation = observation => {
	if (!observation) return "—";
	if (typeof observation === "string") return observation;
	return JSON.stringify(observation, null, 2);
};

const createExecutionPanel = task => {
	const panel = document.createElement("section");
	panel.className = "execution-panel";
	panel.innerHTML = `
		<div class="panel-header">
			<h3>Детали выполнения</h3>
			<span class="panel-badge">${task.status === "running" ? "в процессе" : task.execution_trace?.length ? "завершено" : "ожидание"}</span>
		</div>
	`;

	const list = document.createElement("div");
	list.className = "execution-list";

	const trace = task.execution_trace || [];
	if (!trace.length) {
		const placeholder = document.createElement("article");
		placeholder.className = "timeline-item";
		placeholder.innerHTML = `
			<h4>${task.status === "running" ? "Агент начал работу" : "Ещё нет действий"}</h4>
			<p>${task.status === "running"
				? "Следи за прогрессом: как только появятся новые мысли, они сразу отобразятся ниже."
				: "Запусти план, чтобы увидеть пошаговую картину: мысли, инструменты и выводы."
			}</p>
		`;
		panel.appendChild(placeholder);
	} else {
		trace.forEach((entry, index) => {
			const actionText = entry.action ? entry.action.type : "—";
			const observationText = formatObservation(entry.observation);
			if (actionText !== "—" && observationText !== "—") {
				const status = determineStepStatus(task, index, trace.length);
				list.appendChild(createExecutionStep(entry, status, index));
			}
		});

		if (list.children.length > 0) {
			panel.appendChild(list);
		} else {
			const placeholder = document.createElement("article");
			placeholder.className = "timeline-item";
			placeholder.innerHTML = `
				<h4>Ещё нет действий</h4>
				<p>Запусти план, чтобы увидеть пошаговую картину: мысли, инструменты и выводы.</p>
			`;
			panel.appendChild(placeholder);
		}
	}

	return panel;
};

const determineStepStatus = (task, index, total) => {
	if (task.status === "running" && index === total - 1) {
		return "current";
	}
	return "done";
};

const createExecutionStep = (entry, status, index) => {
	const item = document.createElement("div");
	item.className = `step-item execution ${status}`;
	item.dataset.key = `execution-step-${index}`;

	const toggle = document.createElement("button");
	toggle.className = "step-toggle";
	const headerTitle = entry.thought?.slice(0, 80) || `Шаг ${index + 1}`;
	const indicatorLabel = status === "current" ? "Сейчас" : "Готово";
	const indicatorIcon = status === "current" ? "●" : "✓";
	toggle.innerHTML = `
		<span class="step-title">${headerTitle}</span>
		<span class="step-meta">
			<span class="step-indicator ${status}">${indicatorIcon} ${indicatorLabel}</span>
			<span class="step-chevron">›</span>
		</span>
	`;
	toggle.addEventListener("click", () => item.classList.toggle("open"));

	const body = document.createElement("div");
	body.className = "step-body";
	body.innerHTML = `
	<p><strong>Мысль:</strong> ${entry.thought || "—"}</p>
		<p><strong>Действие:</strong> ${entry.action ? entry.action.type : "—"}</p>
		${renderCodeBlock(entry.action?.payload)}
		<p><strong>Наблюдение:</strong></p>
		<pre>${formatObservation(entry.observation)}</pre>
	`;

	item.append(toggle, body);
	return item;
};

const renderCodeBlock = payload => {
	if (!payload) return "";
	const code = payload.code || payload.expression || payload.query || payload.prompt;
	if (!code) return "";
	return `<p><strong>Код / выражение:</strong></p><pre>${code}</pre>`;
};

const formatTimestamp = timestamp => {
	if (!timestamp) return "";
	const date = new Date(timestamp * 1000);
	return date.toLocaleTimeString("ru-RU", {
		hour: "2-digit",
		minute: "2-digit",
		second: "2-digit"
	});
};

export const sortPlans = plans => {
	if (!plans) return [];
	return [...plans].sort((a, b) => (b.heuristic_score || 0) - (a.heuristic_score || 0));
};

export const ensurePlanSelection = task => {
	if (task.selected_plan_id || !(task.plans?.length)) {
		return;
	}
	const sorted = sortPlans(task.plans);
	if (sorted.length) {
		task.selected_plan_id = sorted[0].id;
	}
};

export const createTaskWindow = (task, options = {}) => {
	const { onRunTask, onSelectPlan, isTaskRunning, collapseConsole = false, showConsole = true } = options;
	const windowEl = document.createElement("section");
	windowEl.className = `task-window${task.selected_plan_id ? " plan-selected" : ""}`;
	windowEl.id = `task-${task.id}`;

	const header = document.createElement("div");
	header.innerHTML = `
		<h2>${task.problem_text}</h2>
		<div class="task-stats">
			<span>ID: ${task.id.slice(0, 8)}</span>
			<span>Статус: ${task.status || "draft"}</span>
		</div>
	`;

	if (task.status === "thinking" || task.isPlaceholder || task.is_generating) {
		windowEl.append(header, createThinkingView(task));
		return windowEl;
	}

	if (task.status === "failed") {
		windowEl.append(header, createErrorCard(task));
		return windowEl;
	}

	const headerFragments = [header];
	if (task.status === "running") {
		headerFragments.push(createAgentStatusCard(task, "running"));
	} else if ((task.execution_trace || []).length) {
		headerFragments.push(createAgentStatusCard(task, "completed"));
	}

	const planGrid = document.createElement("div");
	planGrid.className = "plans-grid";
	const plans = sortPlans(task.plans);
	plans.forEach(plan => planGrid.appendChild(createPlanCard(task, plan, onSelectPlan)));

	const planActions = document.createElement("div");
	planActions.className = "plan-actions";

	const arrow = document.createElement("div");
	arrow.className = "flow-arrow";
	arrow.textContent = "План → Действия";

	const isRunning = Boolean(isTaskRunning?.(task.id));
	const runBtn = document.createElement("button");
	runBtn.className = "run-btn";
	runBtn.textContent = isRunning ? "Выполнение..." : "Запустить план";
	runBtn.disabled = isRunning;
	runBtn.addEventListener("click", () => {
		if (typeof onRunTask === "function") {
			onRunTask(task.id);
		}
	});

	planActions.append(runBtn);
	if (isRunning) {
		planActions.append(createLoader());
	}

	const progressFeed = createProgressFeed(task, collapseConsole);
	const executionPanel = createExecutionPanel(task);
	const timeline = createTimeline(task);
	const finalAnswer = createFinalAnswerBlock(task);

	const contents = [
		...headerFragments,
		planGrid,
		planActions,
		arrow
	];

	if (showConsole) {
		contents.push(progressFeed);
	}

	contents.push(executionPanel, timeline, finalAnswer);

	windowEl.append(...contents);
	return windowEl;
};
