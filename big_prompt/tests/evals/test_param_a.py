import json
import unittest
from pathlib import Path

from agent_ops import SchemaValidator, LMJudge, AgentMetrics
from settings import get_settings


class FakeJudge:
	def __init__(self, verdicts):
		self.verdicts = verdicts
		self.calls = 0

	def __call__(self, system_prompt, user_prompt, role="judge"):
		response = self.verdicts[self.calls % len(self.verdicts)]
		self.calls += 1
		return json.dumps(response)


class AgentOpsTests(unittest.TestCase):
	def setUp(self):
		self.settings = get_settings()
		self.validator = SchemaValidator(self.settings)

	def test_finish_schema_rejects_missing_values(self):
		action = {
			"type": "FINISH",
			"payload": {
				"answer": "a = 1",
				"values": {},
				"trace_id": "abc123"
			}
		}
		is_valid, errors = self.validator.validate_finish(action)
		self.assertFalse(is_valid)
		self.assertTrue(any("a" in err for err in errors))

	def test_finish_schema_accepts_valid_payload(self):
		action = {
			"type": "FINISH",
			"payload": {
				"answer": "a = 1",
				"values": {"a": [1.0]},
				"trace_id": "trace123",
				"confidence": 0.9
			}
		}
		is_valid, errors = self.validator.validate_finish(action)
		self.assertTrue(is_valid)
		self.assertEqual(errors, [])

	def test_lm_judge_consistency_pass(self):
		fake_model = FakeJudge([
			{"score": 0.9, "verdict": "PASS", "rationale": "ok"},
			{"score": 0.88, "verdict": "PASS", "rationale": "ok"}
		])
		judge = LMJudge(fake_model, self.settings)
		report = judge.evaluate("test", {"answer": "a=1", "values": {"a": [1]}})
		self.assertEqual(report["verdict"], "PASS")

	def test_lm_judge_detects_inconsistency(self):
		fake_model = FakeJudge([
			{"score": 0.9, "verdict": "PASS", "rationale": "ok"},
			{"score": 0.2, "verdict": "FAIL", "rationale": "no"}
		])
		judge = LMJudge(fake_model, self.settings)
		report = judge.evaluate("test", {"answer": "a=1", "values": {"a": [1]}})
		self.assertEqual(report["verdict"], "FAIL")

	def test_agent_metrics_tracks_success_rate(self):
		metrics = AgentMetrics()
		metrics.record_task(True, 1.0, tokens=100)
		metrics.record_task(False, 2.0, tokens=50)
		self.assertAlmostEqual(metrics.success_rate, 0.5)
		self.assertAlmostEqual(metrics.avg_latency, 1.5)

	def test_eval_dataset_structure(self):
		path = Path("tests/evals/sample_tasks.json")
		self.assertTrue(path.exists())
		data = json.loads(path.read_text(encoding="utf-8"))
		self.assertTrue(all("statement" in item for item in data))
		self.assertTrue(all("expected_values" in item for item in data))


if __name__ == "__main__":
	unittest.main()
