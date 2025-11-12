from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Any

try:
	from opentelemetry import trace
	from opentelemetry.sdk.resources import SERVICE_NAME, Resource
	from opentelemetry.sdk.trace import TracerProvider
	from opentelemetry.sdk.trace.export import BatchSpanProcessor, OTLPSpanExporter
except Exception:  # pragma: no cover - optional dependency
	trace = None  # type: ignore
	TracerProvider = None  # type: ignore
	SERVICE_NAME = "service.name"  # type: ignore


class _NullSpan:
	def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
		self.name = name
		self.attributes = attributes or {}
		self.start_time = time.time()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.attributes["duration_ms"] = (time.time() - self.start_time) * 1000.0

	def set_attribute(self, key: str, value: Any):
		self.attributes[key] = value


class TraceContext:
	def __init__(self):
		self.trace_id = uuid.uuid4().hex
		self.enabled = False
		self.export_path: Optional[Path] = None

	def new_run(self) -> str:
		self.trace_id = uuid.uuid4().hex
		return self.trace_id

	def export_span(self, span_data: Dict[str, Any]):
		if self.export_path:
			self.export_path.parent.mkdir(parents=True, exist_ok=True)
			with self.export_path.open("a", encoding="utf-8") as fp:
				fp.write(json.dumps(span_data, ensure_ascii=False) + "\n")


tracer = None
trace_context = TraceContext()


def init_tracer(settings) -> None:
	global tracer
	trace_context.export_path = settings.trace_export_path

	if not settings.telemetry.enable_tracing or trace is None or TracerProvider is None:
		trace_context.enabled = False
		tracer = None
		return

	resource = Resource(attributes={SERVICE_NAME: settings.telemetry.service_name})
	provider = TracerProvider(resource=resource)
	exporter = OTLPSpanExporter(endpoint=settings.telemetry.otlp_endpoint, insecure=True)
	processor = BatchSpanProcessor(exporter)
	provider.add_span_processor(processor)
	trace.set_tracer_provider(provider)
	tracer = trace.get_tracer(settings.telemetry.service_name)
	trace_context.enabled = True


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
	if tracer is None or not trace_context.enabled or trace is None:
		yield _NullSpan(name, attributes)
		return

	with tracer.start_as_current_span(name) as span:
		if attributes:
			for key, value in attributes.items():
				span.set_attribute(key, value)
		yield span

