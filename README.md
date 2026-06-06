# MATH-AGENT-VL 🧮🤖

[**Русская версия (Russian Version)**](README_RU.md)

MATH-AGENT-VL is a reference implementation of a **Vision-Language Math Agent** pipeline. It is specifically designed to solve complex multi-step math and reasoning problems by combining structured planning (Tree of Thoughts) with tool-augmented execution (ReAct).

## 🚀 Features

- **Multi-step Reasoning:** Solves complex math problems by breaking them down using a cognitive architecture.
- **ToT-over-ReAct Architecture:** Employs Tree-of-Thought (ToT) for robust planning and ReAct (Reason + Act) for execution.
- **Tool Integration:** Built-in support for a Python sandbox (PAL), SymPy for symbolic math, and WolframAlpha for external knowledge retrieval.
- **Formal Verification:** Includes a verification step to mathematically prove and check the steps taken before arriving at a final answer.
- **Web UI:** A Flask-based web interface to manage tasks, visualize the execution trace, and see the thought processes of the agent in real time.

## 📂 Repository Structure

The project has been organized to keep everything clean and easily accessible:

- **`main.py`** — The primary entry point for the reference MATH-AGENT-VL pipeline.
- **`web/`** — Contains the Flask web application, static assets, and templates for the UI.
- **`agents/`** — Alternative implementations and standalone agent scripts (e.g., Qwen, DeepSeek, experimental scripts).
- **`docs/`** — Documentation files. Check out [`how-it-works.md`](docs/how-it-works.md) for an in-depth dive into the pipeline architecture.
- **`logs/`** — Execution logs for the agent and web server.
- **`results/`** — Output JSON files from previous agent runs.
- **`legacy/`** — Old test files, health checks, and big prompt data preserved for historical purposes.

## 🧠 Core Architecture

The core pipeline operates in four main stages:

1. **Input Formalizer:** Parses the unstructured problem text into a structured `ProblemObject` (extracting goals, constraints, and entities).
2. **Tree-of-Thought Planner:** Explores 2-3 different strategies for solving the problem and selects the best one based on heuristic scoring.
3. **ReAct Executor:** Steps through the chosen plan using `Thought`, `Action`, and `Observation`. It leverages tools like Python execution and SymPy.
4. **Verifier & Critic:** A formal verification process to ensure the reasoning trace and final answer are mathematically sound.

For a detailed breakdown, please read the [Architecture Documentation](docs/how-it-works.md).

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd math-agent-vl
   ```

2. **Install dependencies:**
   Make sure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: SymPy and other math tools are required. If `requirements.txt` is missing, you can run `pip install sympy flask requests openai google-generativeai`)*

3. **Set Environment Variables:**
   You will need to set API keys for the LLMs and tools you wish to use.
   ```bash
   export GEMINI_API_KEY="your_gemini_key"
   export OPENROUTER_API_KEY="your_openrouter_key"
   export WOLFRAM_API_KEY="your_wolfram_key"
   ```

## 🖥️ Running the Project

### Command Line
You can run the core pipeline directly:
```bash
python main.py
```

### Web Interface
To start the Flask web server:
```bash
python web/app.py
```
Then open your browser and navigate to `http://localhost:5051`. The web UI allows you to submit tasks, configure API keys, and watch the execution trace live.
