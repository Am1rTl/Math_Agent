#!/usr/bin/env python3
"""
Offline evaluation harness for param-a tasks.

Usage:
    python scripts/run_evals.py --run-agent --max-tasks 1

Without --run-agent the script performs a dry run that only validates datasets.
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean

from settings import get_settings
from deepseek import MATHAGENTVL, lm_judge


def compare_values(actual, expected, tolerance=1e-6):
	def normalize(values):
		return sorted(round(float(value), 6) for value in values)

	return normalize(actual) == normalize(expected)


def run(args):
	settings = get_settings()
	eval_path = settings.eval_set_path
	if not eval_path.exists():
		raise FileNotFoundError(f"Eval set not found: {eval_path}")

	tasks = json.loads(eval_path.read_text(encoding="utf-8"))
	if args.max_tasks:
		tasks = tasks[: args.max_tasks]

	results = []
	agent = MATHAGENTVL() if args.run_agent else None

	for task in tasks:
		start = time.time()
		if agent:
			result = agent.solve_problem(task["statement"])
			final_answer = result.get("final_answer") or {}
			actual_values = final_answer.get("values", {}).get("a", [])
			success = compare_values(actual_values, task["expected_values"]["a"])
		else:
			final_answer = {}
			success = False
			result = {}
		latency = time.time() - start
		results.append({
			"id": task["id"],
			"success": success,
			"latency": latency,
			"final_answer": final_answer
		})

	success_rate = sum(1 for item in results if item["success"]) / len(results) if results else 0.0
	avg_latency = mean(item["latency"] for item in results) if results else 0.0

	summary = {
		"success_rate": success_rate,
		"avg_latency": avg_latency,
		"total": len(results),
		"results": results
	}

	if args.eval_judge:
		judge_scores = []
		judge_failures = 0
		for task in tasks:
			payload = {
				"answer": f"a = {task['expected_values']['a']}",
				"values": task["expected_values"],
				"trace_id": "judge_eval"
			}
			report = lm_judge.evaluate(task["statement"], payload)
			judge_scores.append(report.get("score", 0.0))
			if report.get("verdict") != "PASS":
				judge_failures += 1
		summary["judge_eval"] = {
			"avg_score": mean(judge_scores) if judge_scores else 0.0,
			"failures": judge_failures,
			"total": len(tasks)
		}

	if args.output:
		Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

	print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
	parser = argparse.ArgumentParser(description="Run offline evals for param-a tasks.")
	parser.add_argument("--run-agent", action="store_true", help="Execute the full agent (may incur API cost).")
	parser.add_argument("--max-tasks", type=int, help="Limit number of tasks.")
	parser.add_argument("--output", type=str, help="Path to save summary JSON.")
	parser.add_argument("--eval-judge", action="store_true", help="Calibrate the LM judge on the gold set.")
	args = parser.parse_args()
	run(args)


if __name__ == "__main__":
	main()
