from __future__ import annotations
import os, subprocess, tempfile, threading, time, shutil, textwrap
import resource, psutil
import json


# -------- Helper function: Normalize output --------
def normalize_output(text: str | None) -> list[str]:
    """
    Normalize output text for OJ comparison:
    1. Unify newline characters to \\n
    2. Remove trailing whitespace from each line
    3. Remove trailing blank lines from the text
    """
    if text is None:
        return []  # If input is None (e.g., RE/TL/ML), return an empty list
    # 1. Unify newline characters and split into lines
    lines = text.replace("\\r\\n", "\\n").replace("\\r", "\\n").split("\\n")

    # 2. Remove trailing whitespace from each line
    processed_lines = [line.rstrip() for line in lines]

    # 3. Remove trailing blank lines
    while processed_lines and processed_lines[-1] == "":
        processed_lines.pop()

    return processed_lines


# -------- Compilation parameters ---------
GPP = "g++"
CXXFLAGS = ["-std=c++23", "-O2", "-pipe", "-static", "-s"]


# -------- Compilation phase ---------
def compile_source_code(
    code: str,
    workdir: str,
    src_file_name: str = "main.cpp",
    exe_file_name: str = "main",
) -> tuple[bool, str]:
    """
    Compiles C++ source code.
    ok == True  -> info = exe_path
    ok == False -> info = compiler_error_message
    """
    src_path = os.path.join(workdir, src_file_name)
    exe_path = os.path.join(workdir, exe_file_name)

    # Adjust exe_path for Windows to ensure it's executable
    if os.name == "nt":
        exe_path += ".exe"

    with open(src_path, "w", encoding="utf-8") as f:
        f.write(code)

    compile_command = [GPP, *CXXFLAGS, src_path, "-o", exe_path]
    # print(f"Compilation command: {' '.join(compile_command)}") # For debugging

    try:
        res = subprocess.run(
            compile_command,
            capture_output=True,  # More modern usage
            text=True,
            timeout=30,  # Add compilation timeout
        )
        if res.returncode == 0:
            if not os.path.exists(exe_path):  # Double-check if exe exists
                return (
                    False,
                    f"Compiler reported success but executable '{exe_path}' not found.",
                )
            return True, exe_path
        # Prefer stderr, if empty use stdout
        error_message = res.stderr.strip() if res.stderr.strip() else res.stdout.strip()
        return False, error_message if error_message else "Unknown compilation error"
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out."
    except Exception as e:
        return False, f"Compilation failed with exception: {e}"


# -------- Resource limit (Unix-like systems only) ---------
def limit_resources_unix(mem_mb: int):
    if os.name != "posix":  # resource module is not available on Windows
        return
    try:
        limit_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        # resource.setrlimit(resource.RLIMIT_NPROC, (64, 64)) # RLIMIT_NPROC might have restrictions or require permissions
    except Exception as e:
        print(f"Warning: Failed to set resource limits: {e}")


# -------- Run phase ---------
def run_with_limits(
    executable: str,
    input_data: str,
    time_limit_sec: float,
    mem_limit_mb: int,  # time_limit in seconds
) -> tuple[str | None, str, float, float]:
    """
    Returns (stdout or None, reason, time (ms), memory (MiB))
        Reason: '' | 'TL' | 'ML' | 'RE'
    """
    start_time = time.monotonic()

    preexec_fn_to_use = None
    if os.name == "posix":
        preexec_fn_to_use = lambda: limit_resources_unix(mem_limit_mb)

    try:
        proc = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=preexec_fn_to_use,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ),  # Windows: Do not create a console window
        )
    except FileNotFoundError:
        return None, "RE", 0, 0  # Executable not found is also considered RE
    except Exception as e:
        return None, "RE", 0, 0  # Other startup errors

    ps_proc = None
    try:
        ps_proc = psutil.Process(proc.pid)
    except (
        psutil.NoSuchProcess
    ):  # Process might have finished before creating psutil object
        # Try to get process exit code
        proc.poll()  # Update returncode
        stdout, stderr = proc.communicate()  # Get output
        end_time = time.monotonic()
        exec_time_ms = (end_time - start_time) * 1000
        if proc.returncode != 0:
            return (
                stdout if stdout else None,
                "RE",
                exec_time_ms,
                0,
            )  # Assume memory is 0
        return stdout, "", exec_time_ms, 0

    killed_for_mem = threading.Event()
    peak_memory_kb = 0
    mem_thread = None

    # Memory monitoring thread (psutil)
    def mem_watch_psutil():
        nonlocal peak_memory_kb
        try:
            current_peak_rss_kb = 0
            while proc.poll() is None:  # As long as the process is running
                try:
                    mem_info = ps_proc.memory_info()
                    current_rss_bytes = mem_info.rss
                    current_peak_rss_kb = max(
                        current_peak_rss_kb, current_rss_bytes / 1024
                    )

                    if current_rss_bytes > mem_limit_mb * 1024 * 1024:
                        killed_for_mem.set()
                        proc.kill()  # Try to terminate the process
                        # print(f"Process {proc.pid} killed for exceeding memory limit.")
                        break
                except psutil.NoSuchProcess:  # Process has ended
                    break
                except Exception:  # Other psutil errors
                    break
                time.sleep(0.01)  # Polling interval

            # After the process ends, get the peak value again (if not terminated due to exceeding limit before)
            if not killed_for_mem.is_set():
                try:
                    # If the process is still running (unlikely), or try to get the last information
                    if ps_proc.is_running():  # Should be false here generally
                        current_peak_rss_kb = max(
                            current_peak_rss_kb, ps_proc.memory_info().rss / 1024
                        )
                except psutil.NoSuchProcess:
                    pass  # Process is no longer there
                except Exception:
                    pass
            peak_memory_kb = current_peak_rss_kb
        except Exception:  # Unexpected error in mem_watch_psutil itself
            peak_memory_kb = max(0, peak_memory_kb)  # At least keep the recorded value

    if (
        ps_proc
    ):  # Start monitoring only if psutil.Process object was successfully created
        mem_thread = threading.Thread(target=mem_watch_psutil, daemon=True)
        mem_thread.start()

    stdout_val, stderr_val = None, None
    reason = ""
    exec_time_ms = 0

    try:
        stdout_val, stderr_val = proc.communicate(
            input=input_data, timeout=time_limit_sec
        )
        # communicate will wait for the process to end
        if mem_thread:
            mem_thread.join(
                timeout=0.2
            )  # Give the memory monitoring thread some time to finish

        end_time = time.monotonic()
        exec_time_ms = (end_time - start_time) * 1000

        if killed_for_mem.is_set():  # Prioritize checking for memory limit exceeded
            reason = "ML"
            stdout_val = None
        elif proc.returncode != 0:
            reason = "RE"
            # print(f"Runtime Error. Exit code: {proc.returncode}. Stderr: {stderr_val[:500] if stderr_val else ''}")
            stdout_val = None
        else:
            reason = ""  # OK

    except subprocess.TimeoutExpired:
        proc.kill()  # Ensure the process is terminated
        if mem_thread:
            mem_thread.join(timeout=0.2)
        end_time = time.monotonic()
        # exec_time_ms = time_limit_sec * 1000 # Or use actual timeout time
        exec_time_ms = (end_time - start_time) * 1000
        reason = "TL"
        stdout_val = None
        # print(f"Process timed out after {exec_time_ms:.2f} ms.")
    except Exception as e:  # Other communicate errors
        # print(f"Communicate Error: {e}")
        proc.kill()
        if mem_thread:
            mem_thread.join(timeout=0.2)
        end_time = time.monotonic()
        exec_time_ms = (end_time - start_time) * 1000
        reason = "RE"
        stdout_val = None

    # If memory monitoring is not complete or an exception occurs, try to get psutil memory one last time
    # peak_memory_kb should be updated by mem_watch_psutil
    # This is mainly to ensure there is a value
    if peak_memory_kb == 0 and not killed_for_mem.is_set() and ps_proc:
        try:
            # Try to get memory information again, in case the monitoring thread did not fully record it
            if ps_proc.is_running():  # Generally, the process should have ended by now
                peak_memory_kb = max(peak_memory_kb, ps_proc.memory_info().rss / 1024)
            # else: # If the process has ended, peak_memory_kb is the final value
            #     pass
        except psutil.NoSuchProcess:
            pass
        except Exception:
            pass

    peak_memory_mb_val = (
        min(peak_memory_kb / 1024, mem_limit_mb) if peak_memory_kb > 0 else 0
    )
    peak_memory_mb_val = max(0, peak_memory_mb_val)  # Ensure non-negative

    return stdout_val, reason, exec_time_ms, peak_memory_mb_val


# -------- Local testing class ---------
class LocalCodeSubmitter:
    def __init__(self):  # Removed data_dirs
        pass

    # _find_test_cases method removed

    def submit_code(
        self,
        problem_id: int,  # Passed from case['id']
        code: str,  # User-generated code
        problem_info: dict,  # case object containing all problem information
        all_judge: bool = True,  # Defaults to True
    ) -> dict:
        """
        Compile and run code locally, and judge it against test data.
        problem_info is the case object from the dataset.
        """
        # Extract required fields from problem_info
        problem_type = problem_info.get("type", "traditional")
        time_limit_ms = problem_info.get("time_limit_ms", 1000)
        memory_limit_mb = problem_info.get("memory_limit_mb", 256)

        # New test case source
        # problem_info['test_cases'] is a list, each element is [input_string, output_string]
        test_cases_data = problem_info.get("test_cases")
        if not test_cases_data or not isinstance(test_cases_data, list):
            return {
                "success": False,
                "message": f"No 'test_cases' found or invalid format in problem_info for ID: {problem_id}",
                "verdict": "System Error",
                "time": 0,
                "memory": 0,
                "passed": False,
                "failed_test": None,
            }

        temp_dir = tempfile.mkdtemp(prefix=f"local_submit_{problem_id}_")
        try:
            # 1. Compile user code
            compile_ok, compile_info = compile_source_code(
                code, temp_dir, "main.cpp", "user_main"
            )
            if not compile_ok:
                shutil.rmtree(temp_dir)
                return {
                    "success": False,
                    "message": f"Compilation Error: {compile_info}",
                    "verdict": "Compilation Error",
                    "time": 0,
                    "memory": 0,
                    "passed": False,
                    "failed_test": None,
                }
            executable_path = compile_info

            # 2. SPJ compilation (if needed)
            spj_executable_path = None
            if problem_type == "spj":
                spj_code_str = problem_info.get("spj_code")
                if not spj_code_str:
                    shutil.rmtree(temp_dir)
                    return {
                        "success": False,
                        "message": f"SPJ code string not found in problem_info for SPJ problem ID: {problem_id}",
                        "verdict": "System Error",
                        "time": 0,
                        "memory": 0,
                        "passed": False,
                        "failed_test": None,
                    }

                # Compile SPJ code, use different executable file name
                spj_compile_ok, spj_compile_info = compile_source_code(
                    spj_code_str, temp_dir, "spj_checker.cpp", "spj_checker_exe"
                )
                if not spj_compile_ok:
                    shutil.rmtree(temp_dir)
                    return {
                        "success": False,
                        "message": f"SPJ Compilation Error: {spj_compile_info}",
                        "verdict": "System Error",
                        "time": 0,
                        "memory": 0,
                        "passed": False,
                        "failed_test": None,
                    }
                spj_executable_path = spj_compile_info

            # 3. Run test cases
            time_limit_sec_float = time_limit_ms / 1000.0
            max_time_ms_overall = 0.0
            max_memory_mb_overall = 0.0
            all_individual_case_results = []
            first_failure_details = None

            for i, (input_data_str, expected_output_str) in enumerate(test_cases_data):
                test_number = i + 1
                current_case_res_dict = {"test_number": test_number}

                # Ensure input_data_str and expected_output_str are strings
                if not isinstance(input_data_str, str) or not isinstance(
                    expected_output_str, str
                ):
                    # print(f"Warning: Test case {test_number} for problem {problem_id} has non-string input/output. Skipping.")
                    current_case_res_dict.update(
                        {
                            "verdict": "System Error",
                            "time": 0,
                            "memory": 0,
                            "message": "Invalid test case data (non-string input/output)",
                        }
                    )
                    all_individual_case_results.append(current_case_res_dict)
                    if first_failure_details is None:
                        first_failure_details = {
                            "verdict": "System Error",
                            "time": 0,
                            "memory": 0,
                            "test_number": test_number,
                            "message": "Invalid test case data",
                        }
                    if not all_judge:
                        break
                    continue

                expected_output_lines = normalize_output(expected_output_str)

                actual_stdout, run_reason, exec_time, peak_mem = run_with_limits(
                    executable_path,
                    input_data_str,
                    time_limit_sec_float,
                    memory_limit_mb,
                )

                max_time_ms_overall = max(max_time_ms_overall, exec_time)
                max_memory_mb_overall = max(max_memory_mb_overall, peak_mem)

                verdict_this_case = ""
                if run_reason == "TL":
                    verdict_this_case = "Time Limit Exceeded"
                elif run_reason == "ML":
                    verdict_this_case = "Memory Limit Exceeded"
                elif run_reason == "RE":
                    verdict_this_case = "Runtime Error"
                elif (
                    actual_stdout is None and run_reason == ""
                ):  # May indicate other RE or unhandled errors
                    verdict_this_case = "Runtime Error"
                else:  # Normal termination, need to compare output
                    actual_output_lines = normalize_output(actual_stdout)
                    if problem_type == "traditional":
                        if actual_output_lines == expected_output_lines:
                            verdict_this_case = "Accepted"
                        else:
                            verdict_this_case = "Wrong Answer"
                    elif problem_type == "spj":
                        if (
                            not spj_executable_path
                        ):  # SPJ code not compiled or path error
                            verdict_this_case = "System Error"  # SPJ execution error
                            current_case_res_dict["message"] = (
                                "SPJ executable not available."
                            )
                        else:
                            # SPJ needs input_file_path, output_file_path, user_output_file_path
                            # We have strings here, need to write to temporary files
                            temp_input_path = os.path.join(
                                temp_dir, f"case_{test_number}_input.txt"
                            )
                            temp_expected_output_path = os.path.join(
                                temp_dir, f"case_{test_number}_expected.txt"
                            )
                            temp_user_output_path = os.path.join(
                                temp_dir, f"case_{test_number}_user.txt"
                            )

                            with open(temp_input_path, "w", encoding="utf-8") as f:
                                f.write(input_data_str)
                            with open(
                                temp_expected_output_path, "w", encoding="utf-8"
                            ) as f:
                                f.write(expected_output_str)
                            with open(
                                temp_user_output_path, "w", encoding="utf-8"
                            ) as f:
                                f.write(actual_stdout or "")

                            spj_command = [
                                spj_executable_path,
                                temp_input_path,  # Pass input file to SPJ
                                temp_expected_output_path,  # Pass standard answer file to SPJ
                                temp_user_output_path,  # Pass user output file to SPJ
                            ]
                            try:
                                spj_run_res = subprocess.run(
                                    spj_command,
                                    capture_output=True,
                                    text=True,
                                    timeout=10,  # SPJ itself should also have a timeout
                                )
                                # SPJ's exit code usually indicates the result, or stdout contains the result string
                                # Here we assume SPJ's stdout is directly "Accepted", "Wrong Answer", etc.
                                # Or SPJ's return code 0 means AC, non-zero means WA/PE, etc. This needs to be adjusted according to SPJ specifications.
                                # Simple assumption: SPJ outputs verdict string
                                spj_output_verdict = spj_run_res.stdout.strip()
                                if (
                                    spj_run_res.returncode != 0
                                    and not spj_output_verdict
                                ):  # If SPJ errors out and has no output
                                    verdict_this_case = (
                                        "System Error"  # SPJ execution error
                                    )
                                    current_case_res_dict["message"] = (
                                        f"SPJ execution failed. stderr: {spj_run_res.stderr[:200]}"
                                    )
                                elif spj_output_verdict:  # Prioritize SPJ's output
                                    verdict_this_case = spj_output_verdict
                                elif (
                                    spj_run_res.returncode == 0
                                ):  # If no output but returns 0, temporarily assume AC
                                    verdict_this_case = "Accepted"
                                else:  # No output and returns non-zero
                                    verdict_this_case = "Wrong Answer"  # Or other, depending on SPJ convention

                            except subprocess.TimeoutExpired:
                                verdict_this_case = "System Error"  # SPJ timeout
                                current_case_res_dict["message"] = (
                                    "SPJ timed out during execution."
                                )
                            except Exception as spj_e:
                                verdict_this_case = "System Error"  # Other SPJ errors
                                current_case_res_dict["message"] = (
                                    f"SPJ execution error: {spj_e}"
                                )
                    else:  # Unknown problem_type
                        verdict_this_case = "System Error"
                        current_case_res_dict["message"] = (
                            f"Unknown problem type: {problem_type}"
                        )

                current_case_res_dict.update(
                    {
                        "verdict": verdict_this_case,
                        "time": exec_time,
                        "memory": peak_mem,
                    }
                )
                all_individual_case_results.append(current_case_res_dict)

                if verdict_this_case != "Accepted":
                    if first_failure_details is None:
                        first_failure_details = {
                            "verdict": verdict_this_case,
                            "time": exec_time,
                            "memory": peak_mem,
                            "test_number": test_number,
                            "message": current_case_res_dict.get(
                                "message", f"Failed on test {test_number}"
                            ),
                        }
                    if not all_judge:
                        break  # Stop and return the first error

                # print(f"[Problem_id]:{problem_id} [Test]:{test_number} [Verdict]: {verdict_this_case} [Time]: {exec_time:.2f}ms [Mem]: {peak_mem:.2f}MB")

            # 4. Processing after the loop ends
            shutil.rmtree(temp_dir)

            if first_failure_details:
                # Failed test case exists
                is_sys_error = first_failure_details["verdict"] == "System Error"
                final_overall_result = {
                    "success": not is_sys_error,  # Only system errors count as unsuccessful execution
                    "message": first_failure_details.get(
                        "message",
                        f"Failed on test {first_failure_details['test_number']}",
                    ),
                    "verdict": first_failure_details["verdict"],
                    "time": first_failure_details[
                        "time"
                    ],  # Time of the first failed test case
                    "memory": first_failure_details[
                        "memory"
                    ],  # Memory of the first failed test case
                    "passed": False,
                    "failed_test": first_failure_details["test_number"],
                }
                if all_judge:  # If all test cases were judged, add the cases key
                    final_overall_result["cases"] = all_individual_case_results
                return final_overall_result
            else:
                # All test cases passed
                final_overall_result = {
                    "success": True,
                    "message": "All tests passed",
                    "verdict": "Accepted",
                    "time": max_time_ms_overall,
                    "memory": max_memory_mb_overall,
                    "passed": True,
                    "failed_test": None,
                }
                if all_judge:  # If all test cases were judged (even if all AC, return)
                    final_overall_result["cases"] = all_individual_case_results
                return final_overall_result

        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            # print(f"Unexpected error in submit_code for problem {problem_id}: {e}", traceback.format_exc())
            return {
                "success": False,
                "message": f"An unexpected error occurred: {e}",
                "verdict": "System Error",
                "time": 0,
                "memory": 0,
                "passed": False,
                "failed_test": None,
            }


# -------- Example usage (needs to be updated to adapt to the new data format) --------
if __name__ == "__main__":
    # Example: Construct a problem_info object that conforms to the new format
    sample_problem_info_spj = {
        "id": 5,  # Problem ID
        "title": "Sample SPJ Problem",
        "type": "spj",  # Type is spj
        "time_limit_ms": 10000,
        "memory_limit_mb": 1024,
        "description": "A sample SPJ problem.",
        "input": "Any integer n.",
        "output": "n+1 if n is even, n-1 if n is odd.",
        "examples": [  # Examples, used for CodeGenerator.check_examples
            ["2\n", "3\n"],
            ["3\n", "2\n"],
        ],
        "test_cases": [  # Actual test cases
            ["10\n", "11\n"],
            ["15\n", "14\n"],
            ["0\n", "1\n"],
            ["-1\n", "-2\n"],  # Assume SPJ can handle negative numbers
        ],
        "spj_code": textwrap.dedent(
            """\
            #include <iostream>
            #include <fstream>
            #include <string>
            #include <vector>

            // Helper to read entire file into string
            std::string readFile(const std::string& filePath) {
                std::ifstream t(filePath);
                if (!t) return "";
                return std::string((std::istreambuf_iterator<char>(t)),
                                   std::istreambuf_iterator<char>());
            }
            
            // Helper to normalize output (basic version)
            std::vector<std::string> normalize(const std::string& text) {
                std::vector<std::string> lines;
                std::string current_line;
                for (char ch : text) {
                    if (ch == '\\r') continue;
                    if (ch == '\\n') {
                        while (!current_line.empty() && isspace(current_line.back())) {
                            current_line.pop_back();
                        }
                        lines.push_back(current_line);
                        current_line.clear();
                    } else {
                        current_line += ch;
                    }
                }
                if (!current_line.empty()) {
                     while (!current_line.empty() && isspace(current_line.back())) {
                        current_line.pop_back();
                    }
                    lines.push_back(current_line);
                }
                while (!lines.empty() && lines.back().empty()) {
                    lines.pop_back();
                }
                return lines;
            }

            int main(int argc, char* argv[]) {
                if (argc < 4) {
                    // std::cerr << "Usage: spj <input_file> <expected_output_file> <user_output_file>" << std::endl;
                    // Exit with a code that indicates an issue to the judge, or print to stdout
                    // For this example, let's assume printing verdict to stdout is the contract
                    std::cout << "System Error" << std::endl; // Or specific error code
                    return 1; // SPJ error
                }

                // std::string inputFile = argv[1];
                std::string expectedOutputFile = argv[2];
                std::string userOutputFile = argv[3];

                std::string expected_str = readFile(expectedOutputFile);
                std::string user_str = readFile(userOutputFile);

                std::vector<std::string> expected_lines = normalize(expected_str);
                std::vector<std::string> user_lines = normalize(user_str);
                
                if (expected_lines == user_lines) {
                    std::cout << "Accepted" << std::endl;
                    return 0; // Accepted
                } else {
                    // For debugging, you might print differences to stderr
                    // std::cerr << "Expected: " << expected_str << std::endl;
                    // std::cerr << "User: " << user_str << std::endl;
                    std::cout << "Wrong Answer" << std::endl;
                    return 1; // Wrong Answer (or specific non-zero code)
                }
            }
            """
        ),
        "note": "This is a note for SPJ.",
    }

    # Example user code (assume it is correct)
    user_correct_code_for_spj_problem = textwrap.dedent(
        """\
        #include <iostream>
        int main() {
            long long n;
            std::cin >> n;
            if (n % 2 == 0) {
                std::cout << n + 1 << std::endl;
            } else {
                std::cout << n - 1 << std::endl;
            }
            return 0;
        }
        """
    )

    user_wrong_code_for_spj_problem = textwrap.dedent(
        """\
        #include <iostream>
        int main() {
            long long n;
            std::cin >> n;
            std::cout << n * 2 << std::endl; // Wrong logic
            return 0;
        }
        """
    )

    submitter = LocalCodeSubmitter()
    print("Testing SPJ problem with correct user code:")
    result_spj_correct = submitter.submit_code(
        problem_id=sample_problem_info_spj["id"],
        code=user_correct_code_for_spj_problem,
        problem_info=sample_problem_info_spj,
        all_judge=True,
    )
    print(json.dumps(result_spj_correct, indent=4))

    print("\nTesting SPJ problem with wrong user code:")
    result_spj_wrong = submitter.submit_code(
        problem_id=sample_problem_info_spj["id"],
        code=user_wrong_code_for_spj_problem,
        problem_info=sample_problem_info_spj,
        all_judge=True,
    )
    print(json.dumps(result_spj_wrong, indent=4))

    # Example traditional problem
    sample_problem_info_traditional = {
        "id": 1,
        "title": "A+B Problem",
        "type": "traditional",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "description": "Read two integers a and b, print their sum.",
        "input": "Two integers a and b (-10^9 <= a, b <= 10^9).",
        "output": "The sum a+b.",
        "examples": [["1 2\n", "3\n"]],
        "test_cases": [
            ["1 2\n", "3\n"],
            ["-1 5\n", "4\n"],
            ["1000000000 1000000000\n", "2000000000\n"],
        ],
        "note": "",
    }
    user_code_aplusb = textwrap.dedent(
        """\
    #include <iostream>
    int main() {
        long long a, b;
        std::cin >> a >> b;
        std::cout << a + b << std::endl;
        return 0;
    }
    """
    )
    print("\nTesting traditional A+B problem:")
    result_trad = submitter.submit_code(
        problem_id=sample_problem_info_traditional["id"],
        code=user_code_aplusb,
        problem_info=sample_problem_info_traditional,
        all_judge=True,
    )
    print(json.dumps(result_trad, indent=4))
