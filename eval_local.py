import datasets
from CodeGenerator import CodeGenerator
from SubmitProblem_local import LocalCodeSubmitter
import json
import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler
import argparse


# Configure logger
def setup_logger():
    # Create log directory (if it doesn't exist)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger("eval_local")
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create file handler (using RotatingFileHandler to automatically rotate log files)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "eval_local.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Function to process a single problem
def process_problem(
    code_generator,
    submitter,
    problem_id,
    case,
    submit_lock,
    result_lock,
    logger,
    evaluation_report,
    passed_count,
    results_file,
    max_attempts,
    max_generate,
    stream,
    sleep,
    reasoning_history,
):
    thread_name = threading.current_thread().name

    logger.info(
        f"Thread {thread_name} - {case['id']}.{case['title']}: Evaluating problem"
    )

    problem_result = {
        "id": case["id"],
        "title": case["title"],
        "url": case.get("qoj_url"),
        "type": case["type"],
        "problem_len": len(case["description"]) if case.get("description") else 0,
        "attempts": [],
        "eval_status": "Running",
        "source": case.get("source"),
        "year": case.get("year"),
        "problem_label": case.get("problem_label"),
        "tags": case.get("tags"),
    }
    with result_lock:
        update_evaluation_report(
            evaluation_report,
            problem_result,
            logger,
            results_file,
        )

    correct_info = ""
    final_status = "Failed"
    passed_this_problem = 0

    # Outer loop, regenerate code from scratch
    for gen_idx in range(max_generate):
        if gen_idx > 0:
            logger.info(
                f"Thread {thread_name} - {case['id']}.{case['title']}: Regenerating solution (attempt {gen_idx+1}/{max_generate})"
            )
            # Reset state
            correct_info = ""

        # Inner loop, try to modify the currently generated code
        for attempt_idx in range(max_attempts):
            current_attempt_num = gen_idx * max_attempts + attempt_idx + 1
            logger.info(
                f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Attempting to solve problem (generate {gen_idx+1}/{max_generate}, attempt {attempt_idx+1}/{max_attempts})"
            )

            attempt = {
                "attempt_number": current_attempt_num,
                "sample_tests": {},
                "submission": None,
            }

            # Generate code
            code = ""
            if attempt_idx == 0:  # First attempt of each round
                start_time = time.time()
                code, output, output_len, code_len, token_count = (
                    code_generator.generate_code(
                        case,
                        sleep=sleep,
                        stream=stream,
                        reasoning_history=reasoning_history,
                    )
                )
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Thread {thread_name} - Generated {token_count} tokens in {elapsed_time:.2f} seconds, rate: {token_count/elapsed_time:.2f} tokens/sec"
                )
            else:  # Correct existing code
                start_time = time.time()
                code, output, output_len, code_len, token_count = (
                    code_generator.correct_code(
                        correct_info,
                        period,
                        sleep=sleep,
                        stream=stream,
                        reasoning_history=reasoning_history,
                    )
                )
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Thread {thread_name} - Generated {token_count} tokens in {elapsed_time:.2f} seconds, rate: {token_count/elapsed_time:.2f} tokens/sec"
                )
            attempt["output"] = output
            attempt["output_len"] = output_len
            attempt["code_len"] = code_len
            attempt["token_count"] = token_count
            attempt["code"] = code
            attempt["generation"] = gen_idx + 1

            period = "Sample"

            # Record sample test results
            if "examples" in case and case["examples"]:
                logger.info(
                    f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Running sample tests..."
                )
                result, correct_info_sample = code_generator.check_examples(code, case)
                attempt["sample_tests"] = {
                    "status": result,
                    "details": correct_info_sample,
                }

                if result == "Accepted":
                    period = "Submit"
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Sample tests passed, preparing to submit code..."
                    )
                else:
                    period = "Sample"
                    correct_info = correct_info_sample
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Sample tests failed: {result}"
                    )
            else:
                logger.info(
                    f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: No sample tests provided or 'examples' field is missing/empty. Proceeding to submission."
                )
                period = "Submit"

            if period == "Submit":
                with submit_lock:
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Acquired submission lock, submitting code..."
                    )
                    submission_result = submit_code_with_retry(
                        submitter,
                        case,
                        code,
                        thread_name,
                        logger,
                        attempt_idx,
                    )

                attempt["submission"] = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "details": submission_result,
                }

                if (
                    submission_result["success"]
                    and submission_result["verdict"] == "Accepted"
                ):
                    with result_lock:
                        passed_this_problem = 1
                    final_status = "Accepted"
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Problem passed"
                    )
                else:
                    correct_info = submission_result["verdict"]
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Submission failed: {submission_result['verdict']}"
                    )
            # else:
            #     logger.info(
            #         f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Sample tests failed: {result}"
            #     )

            problem_result["attempts"].append(attempt)
            problem_result["final_status"] = final_status
            problem_result["eval_status"] = "Running"

            if final_status == "Accepted":
                problem_result["eval_status"] = "Completed"
                with result_lock:
                    update_evaluation_report(
                        evaluation_report,
                        problem_result,
                        logger,
                        results_file,
                    )
                return problem_result, passed_this_problem

            with result_lock:
                update_evaluation_report(
                    evaluation_report,
                    problem_result,
                    logger,
                    results_file,
                )

            if period == "Submit" and final_status != "Accepted":
                pass
            elif period == "Sample" and result != "Accepted":
                pass
            else:
                if result == "Accepted" and period == "Sample":
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Sample tests passed but did not proceed to submission in this attempt loop."
                    )
                break

        if final_status == "Accepted":
            break

    problem_result["eval_status"] = "Completed"
    with result_lock:
        update_evaluation_report(
            evaluation_report,
            problem_result,
            logger,
            results_file,
        )
    return problem_result, passed_this_problem


# Submit code and handle duplicate submissions
def submit_code_with_retry(submitter, case, code, thread_name, logger, i):
    submission_result = submitter.submit_code(
        problem_id=case["id"], code=code, problem_info=case, all_judge=True
    )

    logger.info(
        f"Thread {thread_name} - {case['id']}.{case['title']}.{i+1}: Submission result: {submission_result['verdict']}"
    )

    return submission_result


# Update evaluation report
def update_evaluation_report(evaluation_report, problem_result, logger, results_file):
    # Update evaluation report in real-time
    # Check if a result with the corresponding id already exists
    existing_idx = -1
    for i, result in enumerate(evaluation_report["detailed_results"]):
        if result["id"] == problem_result["id"]:
            existing_idx = i
            break

    if existing_idx != -1:
        evaluation_report["detailed_results"][existing_idx] = problem_result
    else:
        insert_pos = 0
        while (
            insert_pos < len(evaluation_report["detailed_results"])
            and evaluation_report["detailed_results"][insert_pos]["id"]
            < problem_result["id"]
        ):
            insert_pos += 1
        evaluation_report["detailed_results"].insert(insert_pos, problem_result)

    running_problems = 0
    completed_problems = 0
    passed = 0
    for result in evaluation_report["detailed_results"]:
        if result["eval_status"] == "Running":
            running_problems += 1
        elif result["eval_status"] == "Completed":
            completed_problems += 1
            if result["final_status"] == "Accepted":
                passed += 1

    evaluation_report["summary"]["evaluated_problems"] = completed_problems
    evaluation_report["summary"]["running_problems"] = running_problems
    evaluation_report["summary"]["passed_problems"] = passed

    denominator = evaluation_report["summary"]["evaluated_problems"]
    if denominator > 0:
        evaluation_report["summary"]["success_rate"] = f"{passed/denominator*100:.2f}%"
    else:
        evaluation_report["summary"]["success_rate"] = "0.00%"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=4)

    logger.info("\nCurrent progress:")
    logger.info(
        f"Total problems (from dataset): {evaluation_report['summary']['total_problems']}"
    )
    logger.info(f"Evaluated: {evaluation_report['summary']['evaluated_problems']}")
    logger.info(f"Evaluating: {evaluation_report['summary']['running_problems']}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Success rate: {evaluation_report['summary']['success_rate']}")


# Load evaluation report
def load_evaluation_report(total_problems_in_dataset, logger, results_file):
    passed = 0
    completed_results = []
    running_results_to_discard = []

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            try:
                evaluation_report = json.load(f)
                if not isinstance(
                    evaluation_report.get("detailed_results"), list
                ) or not isinstance(evaluation_report.get("summary"), dict):
                    raise ValueError("Invalid report format")

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Error loading or parsing existing results file '{results_file}': {e}. Starting fresh."
                )
                evaluation_report = None

        if evaluation_report:
            for result in evaluation_report.get("detailed_results", []):
                if result.get("eval_status") == "Completed":
                    completed_results.append(result)
                    if result.get("final_status") == "Accepted":
                        passed += 1
                elif result.get("eval_status") == "Running":
                    running_results_to_discard.append(result)

            evaluation_report["detailed_results"] = completed_results
            evaluation_report["summary"]["total_problems"] = total_problems_in_dataset
            evaluation_report["summary"]["evaluated_problems"] = len(completed_results)
            evaluation_report["summary"]["passed_problems"] = passed
            evaluation_report["summary"]["running_problems"] = 0

            if len(completed_results) > 0:
                evaluation_report["summary"][
                    "success_rate"
                ] = f"{passed/len(completed_results)*100:.2f}%"
            else:
                evaluation_report["summary"]["success_rate"] = "0.00%"

            logger.info(f"Loaded previous evaluation report from '{results_file}'.")
            logger.info(
                f"Total problems (current dataset): {total_problems_in_dataset}"
            )
            logger.info(
                f"Completed evaluations (from old report): {len(completed_results)}"
            )
            logger.info(
                f"Previously uncompleted evaluations (discarded): {len(running_results_to_discard)}"
            )
            logger.info(f"Passed (from old report): {passed}")
            logger.info(
                f"Success rate (based on completed part of old report): {evaluation_report['summary']['success_rate']}"
            )

        else:
            evaluation_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total_problems": total_problems_in_dataset,
                    "passed_problems": 0,
                    "evaluated_problems": 0,
                    "running_problems": 0,
                    "success_rate": "0.00%",
                },
                "detailed_results": [],
            }
            logger.info(
                "Failed to load valid old report, starting a new evaluation report."
            )
    else:
        evaluation_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_problems": total_problems_in_dataset,
                "passed_problems": 0,
                "evaluated_problems": 0,
                "running_problems": 0,
                "success_rate": "0.00%",
            },
            "detailed_results": [],
        }
        logger.info(
            f"Results file '{results_file}' not found, starting a new evaluation report."
        )

    evaluation_report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    evaluation_report["summary"]["total_problems"] = total_problems_in_dataset

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=4)

    return evaluation_report, passed, completed_results


def evaluate_test_cases(
    num_threads,
    model,
    base_url,
    api_key,
    timeout,
    dataset_name,
    dataset_config_name,
    dataset_split,
    results_file,
    max_attempts,
    max_generate,
    stream,
    sleep,
    reasoning_history,
    auto_choose,
    problem_ids_to_run,
):
    logger = setup_logger()

    logger.info("Starting test evaluation")
    logger.info(f"Using model: {model}")
    logger.info(
        f"Dataset: {dataset_name}, Configuration: {dataset_config_name}, Split: {dataset_split}"
    )
    logger.info(f"Number of threads: {num_threads}")
    logger.info(f"Max modification attempts per generation: {max_attempts}")
    logger.info(f"Max regeneration attempts per problem: {max_generate}")

    submitter = LocalCodeSubmitter()

    try:
        logger.info(
            f"Loading dataset '{dataset_name}' (Configuration: {dataset_config_name}, Split: {dataset_split})..."
        )
        loaded_dataset = datasets.load_dataset(
            dataset_name, name=dataset_config_name, split=dataset_split
        )
        all_problems_list = list(loaded_dataset)
        logger.info(f"Successfully loaded {len(all_problems_list)} test cases.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        return 0

    total_problems_in_current_run = len(all_problems_list)

    evaluation_report, initial_passed_count, completed_results = load_evaluation_report(
        total_problems_in_current_run, logger, results_file
    )

    current_run_passed_count = 0

    submit_lock = threading.Lock()
    result_lock = threading.Lock()
    task_queue = queue.Queue()

    problems_for_this_run = []
    if auto_choose:
        already_evaluated_ids = {result["id"] for result in completed_results}
        for case in reversed(all_problems_list):
            if case["id"] not in already_evaluated_ids:
                problems_for_this_run.append(case)
        logger.info(
            f"Auto choose mode: Will evaluate {len(problems_for_this_run)} new problems."
        )
    else:
        if problem_ids_to_run:
            cases_by_id_map = {case["id"]: case for case in all_problems_list}
            for pid in problem_ids_to_run:
                if pid in cases_by_id_map:
                    problems_for_this_run.append(cases_by_id_map[pid])
                else:
                    logger.warning(f"Problem ID {pid} not found in dataset, skipped.")
            logger.info(
                f"Manual choose mode: Will evaluate {len(problems_for_this_run)} specified problems: {problem_ids_to_run}."
            )
        else:
            logger.info(
                "Manual choose mode: No problem IDs specified (problem_ids_to_run is empty), no problems will be evaluated this time."
            )

    for case in problems_for_this_run:
        task_queue.put(case)

    if task_queue.empty():
        logger.info("No problems to evaluate.")
    else:
        max_workers = min(num_threads, task_queue.qsize())
        logger.info(f"Starting {max_workers} worker threads for parallel processing.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_to_case_id = {}

            for _ in range(max_workers):
                if not task_queue.empty():
                    case_to_process = task_queue.get_nowait()
                    code_generator = CodeGenerator(
                        base_url=base_url, api_key=api_key, model=model, timeout=timeout
                    )
                    future = executor.submit(
                        process_problem,
                        code_generator,
                        submitter,
                        case_to_process["id"],
                        case_to_process,
                        submit_lock,
                        result_lock,
                        logger,
                        evaluation_report,
                        None,
                        results_file,
                        max_attempts,
                        max_generate,
                        stream,
                        sleep,
                        reasoning_history,
                    )
                    futures_to_case_id[future] = case_to_process["id"]

            while futures_to_case_id:
                done, _ = concurrent.futures.wait(
                    futures_to_case_id.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    case_id_processed = futures_to_case_id.pop(future)
                    try:
                        _, problem_passed_status = future.result()
                        if problem_passed_status > 0:
                            current_run_passed_count += 1
                    except Exception as e:
                        logger.error(
                            f"Processing problem ID {case_id_processed} failed: {e}",
                            exc_info=True,
                        )

                    if not task_queue.empty():
                        case_to_process = task_queue.get_nowait()
                        code_generator = CodeGenerator(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            timeout=timeout,
                        )
                        new_future = executor.submit(
                            process_problem,
                            code_generator,
                            submitter,
                            case_to_process["id"],
                            case_to_process,
                            submit_lock,
                            result_lock,
                            logger,
                            evaluation_report,
                            None,
                            results_file,
                            max_attempts,
                            max_generate,
                            stream,
                            sleep,
                            reasoning_history,
                        )
                        futures_to_case_id[new_future] = case_to_process["id"]

    final_passed_count = 0
    final_evaluated_count = 0
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            final_report_state = json.load(f)
        final_report_state["detailed_results"].sort(key=lambda x: x["id"])

        for res in final_report_state["detailed_results"]:
            if res.get("eval_status") == "Completed":
                final_evaluated_count += 1
                if res.get("final_status") == "Accepted":
                    final_passed_count += 1

        final_report_state["summary"]["evaluated_problems"] = final_evaluated_count
        final_report_state["summary"]["passed_problems"] = final_passed_count
        final_report_state["summary"]["running_problems"] = 0

        if final_evaluated_count > 0:
            final_report_state["summary"][
                "success_rate"
            ] = f"{final_passed_count/final_evaluated_count*100:.2f}%"
        else:
            final_report_state["summary"]["success_rate"] = "0.00%"

        final_report_state["summary"]["total_problems"] = total_problems_in_current_run

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(final_report_state, f, ensure_ascii=False, indent=4)

        summary_to_log = final_report_state["summary"]
    else:
        summary_to_log = evaluation_report["summary"]

    logger.info("\nEvaluation complete!")
    logger.info(
        f"Total problems (current run dataset): {summary_to_log.get('total_problems', 'N/A')}"
    )
    logger.info(
        f"Evaluated (all records): {summary_to_log.get('evaluated_problems', 'N/A')}"
    )
    logger.info(f"Passed (all records): {summary_to_log.get('passed_problems', 'N/A')}")
    logger.info(
        f"Success rate (all records): {summary_to_log.get('success_rate', 'N/A')}"
    )

    if summary_to_log.get("evaluated_problems", 0) > 0:
        return summary_to_log.get("passed_problems", 0) / summary_to_log.get(
            "evaluated_problems"
        )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locally evaluate code generated by large models"
    )
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads")
    parser.add_argument(
        "--model", type=str, default="deepseek-reasoner", help="Model name to use"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.deepseek.com/v1",
        help="API base URL",
    )
    parser.add_argument("--api_key", type=str, help="API key (required)")
    parser.add_argument(
        "--timeout", type=int, default=10000, help="API timeout (seconds)"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="RUC-AIBOX/ICPC-Eval",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Hugging Face dataset configuration name (optional)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Hugging Face dataset split (e.g., test, train)",
    )

    parser.add_argument(
        "--results_file",
        type=str,
        default="eval_results_default.json",
        help="Output file path for results (model name will be added automatically)",
    )

    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum modification attempts per generation (including initial submission)",
    )
    parser.add_argument(
        "--max_generate",
        type=int,
        default=1,
        help="How many times to resubmit from scratch for each problem",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Whether to use streaming output for code generation (default: False)",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="Time to wait after generating code (seconds)",
    )
    parser.add_argument(
        "--reasoning_history",
        action="store_true",
        help="Whether to use reasoning history (default: False)",
    )
    parser.add_argument(
        "--auto_choose",
        action="store_true",
        help="Whether to automatically select uncompleted problems for evaluation (default: False)",
    )
    parser.add_argument(
        "--problem_ids",
        type=int,
        nargs="*",
        help="List of problem IDs to run (only effective if auto_choose=False)",
    )

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api_key is required.")

    if "eval_results_default.json" in args.results_file:
        model_name_sanitized = args.model.replace("/", "_")
        args.results_file = f"eval_results_{model_name_sanitized}.json"
        print(f"Results file will be saved as: {args.results_file}")

    return args


if __name__ == "__main__":
    args = parse_args()

    evaluate_test_cases(
        num_threads=args.num_threads,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        results_file=args.results_file,
        max_attempts=args.max_attempts,
        max_generate=args.max_generate,
        stream=args.stream,
        sleep=args.sleep,
        reasoning_history=args.reasoning_history,
        auto_choose=args.auto_choose,
        problem_ids_to_run=args.problem_ids,
    )
