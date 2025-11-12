from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class ModelRoutingConfig:
	system1_model: str = field(default_factory=lambda: os.getenv("SYSTEM1_MODEL", "google/gemini-2.0-flash-lite-001"))
	system2_model: str = field(default_factory=lambda: os.getenv("SYSTEM2_MODEL", "google/gemini-2.5-pro"))
	system1_temperature: float = field(default_factory=lambda: float(os.getenv("SYSTEM1_TEMPERATURE", "0.3")))
	system2_temperature: float = field(default_factory=lambda: float(os.getenv("SYSTEM2_TEMPERATURE", "0.15")))
	planner_temperature: float = field(default_factory=lambda: float(os.getenv("PLANNER_TEMPERATURE", "0.6")))
	executor_temperature: float = field(default_factory=lambda: float(os.getenv("EXECUTOR_TEMPERATURE", "0.1")))
	verifier_temperature: float = field(default_factory=lambda: float(os.getenv("VERIFIER_TEMPERATURE", "0.0")))
	formalizer_temperature: float = field(default_factory=lambda: float(os.getenv("FORMALIZER_TEMPERATURE", "0.1")))


@dataclass
class ThresholdConfig:
	judge_pass_score: float = field(default_factory=lambda: float(os.getenv("JUDGE_PASS_SCORE", "0.78")))
	judge_consistency_drop: float = field(default_factory=lambda: float(os.getenv("JUDGE_CONSISTENCY_DROP", "0.15")))
	finish_retry_limit: int = field(default_factory=lambda: int(os.getenv("FINISH_RETRY_LIMIT", "1")))
	max_schema_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_SCHEMA_BYTES", "16000")))
	human_escalation_threshold: float = field(default_factory=lambda: float(os.getenv("HUMAN_ESCALATION_THRESHOLD", "0.55")))
	max_planner_examples: int = field(default_factory=lambda: int(os.getenv("MAX_PLANNER_EXAMPLES", "3")))
	max_verifier_examples: int = field(default_factory=lambda: int(os.getenv("MAX_VERIFIER_EXAMPLES", "2")))
	max_tool_runtime: float = field(default_factory=lambda: float(os.getenv("MAX_TOOL_RUNTIME", "12.0")))
	kpi_regression_tolerance: float = field(default_factory=lambda: float(os.getenv("KPI_REGRESSION_TOLERANCE", "0.03")))


@dataclass
class SandboxConfig:
	cpu_time_seconds: int = field(default_factory=lambda: int(os.getenv("SANDBOX_CPU_SECONDS", "5")))
	memory_limit_mb: int = field(default_factory=lambda: int(os.getenv("SANDBOX_MEMORY_MB", "256")))
	disable_network: bool = field(default_factory=lambda: os.getenv("SANDBOX_DISABLE_NETWORK", "1") == "1")
	blocked_modules: tuple = (
		"os",
		"sys",
		"socket",
		"subprocess",
		"requests",
		"urllib",
		"http",
	)


@dataclass
class TelemetryConfig:
	enable_tracing: bool = field(default_factory=lambda: os.getenv("ENABLE_OTEL", "1") == "1")
	otlp_endpoint: str = field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"))
	service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "math-agent"))


@dataclass
class AppSettings:
	models: ModelRoutingConfig = field(default_factory=ModelRoutingConfig)
	thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
	sandbox: SandboxConfig = field(default_factory=SandboxConfig)
	telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
	finish_schema_path: Path = field(default_factory=lambda: Path(os.getenv("FINISH_SCHEMA_PATH", "schemas/finish.json")))
	trace_export_path: Path = field(default_factory=lambda: Path(os.getenv("TRACE_EXPORT_PATH", "telemetry/traces.jsonl")))
	human_queue_path: Path = field(default_factory=lambda: Path(os.getenv("HUMAN_QUEUE_PATH", "human_escalations.jsonl")))
	eval_set_path: Path = field(default_factory=lambda: Path(os.getenv("EVAL_SET_PATH", "tests/evals/sample_tasks.json")))

	def as_dict(self) -> Dict[str, Any]:
		return {
			"models": self.models.__dict__,
			"thresholds": self.thresholds.__dict__,
			"sandbox": {
				"cpu_time_seconds": self.sandbox.cpu_time_seconds,
				"memory_limit_mb": self.sandbox.memory_limit_mb,
				"disable_network": self.sandbox.disable_network,
				"blocked_modules": self.sandbox.blocked_modules,
			},
			"telemetry": self.telemetry.__dict__,
			"finish_schema_path": str(self.finish_schema_path),
			"trace_export_path": str(self.trace_export_path),
			"human_queue_path": str(self.human_queue_path),
			"eval_set_path": str(self.eval_set_path),
		}


_SETTINGS: AppSettings | None = None


def get_settings() -> AppSettings:
	global _SETTINGS
	if _SETTINGS is None:
		_SETTINGS = AppSettings()
	return _SETTINGS
