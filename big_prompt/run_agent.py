#!/usr/bin/env python3
"""
Utility launcher for MATHAGENTVL with debug logging always enabled.

Usage:
    python run_agent.py --problem "Найдите значения параметра a ..."
    python run_agent.py --problem-file task.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import deepseek


def read_problem(args: argparse.Namespace) -> str:
	if args.problem:
		return args.problem.strip()
	if args.problem_file:
		return Path(args.problem_file).read_text(encoding="utf-8").strip()
	return "Найдите все значения параметра a, при которых система неравенств имеет ровно два решения."


def main() -> None:
	parser = argparse.ArgumentParser(description="Run math agent with logging enabled.")
	parser.add_argument("--problem", type=str, help="Текст задачи.")
	parser.add_argument("--problem-file", type=str, help="Путь к файлу с задачей.")
	parser.add_argument("--wolfram-key", type=str, help="WolframAlpha API key (overrides env).")
	args = parser.parse_args()

	# Force verbose logging inside deepseek module
	deepseek.DEBUG_MODE = True
	with open(deepseek.LOG_FILE, "w", encoding="utf-8") as log_fp:
		log_fp.write(f"--- Run started ---\n")

	problem_text = read_problem(args)
	wolfram_key = args.wolfram_key

	agent = deepseek.MATHAGENTVL()
	result = agent.solve_problem(problem_text, wolfram_api_key=wolfram_key)

	Path("math_agent_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
	print("\nРезультат сохранен в math_agent_result.json")
	print(f"Логи агента: {deepseek.LOG_FILE}")


if __name__ == "__main__":
	main()
