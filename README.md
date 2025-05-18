# ICPC-Eval: LLM Evaluation Framework for Competitive Programming

ICPC-Eval is a Python-based framework designed to evaluate the capabilities of Large Language Models (LLMs) in solving competitive programming problems, specifically those in the style of the International Collegiate Programming Contest (ICPC). It automates the process of prompting LLMs to generate C++ code for given problems, compiling the generated code, running it against local test cases (including support for Special Judge problems), and reporting detailed results.

## Key Features

*   **LLM-Powered Code Generation**: Interacts with LLM APIs (e.g., DeepSeek API) to generate C++ solutions.
*   **Local Code Evaluation**: Compiles (using `g++` by default) and runs generated C++ code in a controlled local environment.
*   **Test Case Management**: Supports evaluation against predefined test cases, including input/output pairs and special judge (SPJ) logic.
*   **Detailed Reporting**: Generates comprehensive JSON reports summarizing evaluation performance, including pass/fail status, resource usage (time/memory), and error details.
*   **Resumable Evaluation**: Can load previous results and continue evaluation, skipping already completed problems.
*   **Parallel Processing**: Utilizes multiple threads to evaluate problems concurrently, speeding up the overall process.
*   **Customizable**: Flexible design allowing for different LLMs, prompts, and evaluation parameters.

## Main Components

*   `eval_local.py`: The main script to orchestrate the entire evaluation pipeline. It loads problem datasets, manages LLM interactions for code generation, invokes local testing, and aggregates results.
*   `CodeGenerator.py`: A class responsible for interfacing with LLMs. It constructs prompts based on problem descriptions, sends requests to the LLM API, and extracts the generated code and reasoning.
*   `SubmitProblem_local.py`: A class that handles the local compilation and execution of C++ code. It runs the code against provided test cases, measures performance (time and memory), and determines the verdict (e.g., Accepted, Wrong Answer, Time Limit Exceeded).
*   `logs/`: Directory where detailed log files of the evaluation process are stored.
*   `eval_results_*.json`: Output files where the evaluation results are stored in JSON format.

## Prerequisites

*   Python>=3.10
*   A C++ compiler (e.g., `clang++` or `g++`). The default is `g++`, configurable in `SubmitProblem_local.py`.
*   Python Packages: Install the following core dependencies using pip:
    *   `openai`
    *   `datasets`
    *   `psutil`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ICPC-Eval
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install openai datasets psutil
    ```
3.  **Ensure a C++ compiler is installed and accessible in your system's PATH.**
    If `g++` is not your default or preferred compiler, you might need to adjust the `GPP` variable in `SubmitProblem_local.py`.

## How to Run

The evaluation is primarily controlled by the `eval_local.py` script. It accepts several command-line arguments to configure the evaluation run.

**Key command-line arguments for `eval_local.py`:**

*   `--api_key YOUR_API_KEY`: **(Required)** Your API key for the LLM service.
*   `--model MODEL_NAME`: Specifies the LLM model to use (e.g., `deepseek-reasoner`). Default: `deepseek-reasoner`.
*   `--base_url API_BASE_URL`: The base URL for the LLM API. Default: `https://api.deepseek.com/v1`.
*   `--num_threads N`: Number of threads for parallel problem evaluation. Default: `1`.
*   `--dataset_name DATASET_NAME`: Hugging Face dataset name (e.g., `RUC-AIBOX/ICPC-Eval`). Default: `RUC-AIBOX/ICPC-Eval`.
*   `--dataset_split SPLIT_NAME`: Dataset split to use (e.g., `test`). Default: `test`.
*   `--results_file FILE_PATH`: Path to save the JSON results. The model name is automatically appended if the default name is used (e.g., `eval_results_deepseek-reasoner.json`).
*   `--max_attempts N`: Maximum modification attempts per code generation cycle. Default: `5`.
*   `--max_generate N`: Maximum times to regenerate a solution from scratch for a problem. Default: `1`.
*   `--auto_choose`: If set, automatically selects uncompleted problems from the dataset for evaluation based on the results file.
*   `--problem_ids ID1 ID2 ...`: List of specific problem IDs to evaluate. Only effective if `--auto_choose` is not set.
*   `--stream`: Use streaming output for code generation.
*   `--timeout SECONDS`: API timeout in seconds. Default: `10000`.

**Example command:**

```bash
python eval_local.py \
    --api_key YOUR_SECRET_API_KEY \
    --model deepseek-reasoner \
    --num_threads 4 \
    --dataset_name RUC-AIBOX/ICPC-Eval \
    --auto_choose \
    --results_file eval_results_deepseek-reasoner.json
```

## Output

*   **JSON Results**: Detailed evaluation results are saved in the file specified by `--results_file` (e.g., `eval_results_deepseek-reasoner.json`). This file contains per-problem attempts, verdicts, code, and summary statistics.
*   **Log Files**: Execution logs are stored in the `logs/` directory (e.g., `logs/eval_local.log`), providing insights into the evaluation process and any errors encountered.

## Customization

*   **LLM Integration**: Modify `CodeGenerator.py` to support different LLM APIs or to adjust prompting strategies.
*   **Compiler & Flags**: Change the C++ compiler or compilation flags in `SubmitProblem_local.py` (variables `GPP` and `CXXFLAGS`).
*   **Problem Datasets**: Use any Hugging Face dataset compatible with the expected problem structure.
