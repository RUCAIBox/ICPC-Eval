from asyncio.log import logger
import openai
import os
import subprocess
import time
import tempfile
import shutil
import re


class CodeGenerator:
    def __init__(
        self,
        base_url,
        api_key,
        model="deepseek-r1",
        timeout=3600,
        init_prompt="""You are a coding expert. Given a competition-level coding problem, you need to write a C++ program(C++23) to solve it. Please consider the efficiency and time complexity of the algorithm to meet the time limit requirements of the problem. You may start by outlining your thought process.\nIn the end, YOU MUST provide the complete code in a code block enclosed with ``` ```.\nIn the end, YOU MUST provide the complete code in a code block enclosed with ``` ```.\nIn the end, YOU MUST provide the complete code in a code block enclosed with ``` ```.""",
    ):  # deepseek-r1-250120
        self.model = model
        self.init_prompt = init_prompt
        self.generator = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self.history = []

    def extract_code_block(self, code):
        code_blocks = re.findall(r"```(?:cpp)?\n(.*?)```", code, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        # Fallback: if no ```cpp, try to find any ``` block
        if not code_blocks:
            code_blocks = re.findall(r"```\n(.*?)```", code, re.DOTALL)
            if code_blocks:
                return code_blocks[-1].strip()
        return ""  # Return empty if no block found, or the original code if no block

    def extract_reasoning(self, output):
        if any(
            m_name in self.model
            for m_name in ["QwQ", "Qwen3", "deepseek-r1", "deepseek-reasoner", "STILL"]
        ):
            # Assuming "<think>" and "</think>" tags are used by these models for reasoning
            match = re.match(r"(.*?)</think>(.*)", output, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning_part = match.group(1)
                # Remove leading <think> if present
                reasoning_part = re.sub(
                    r"^\s*<think>\s*", "", reasoning_part, flags=re.IGNORECASE
                )
                code_part = match.group(2)
                return reasoning_part.strip(), code_part.strip()
            else:  # No explicit </think> tag found
                # print(f"Warning: No </think> tag found in model output for reasoning extraction. Output: {output[:200]}...")
                return "", output  # Assume no separate reasoning, all is output
        else:
            return "", output  # Default: no separate reasoning

    def generate_code(
        self, problem_info, sleep=0, stream=False, reasoning_history=True
    ):
        time.sleep(problem_info["id"] % 4)
        self.problem_prompt = (
            self.init_prompt
            + "\n\n"
            + f"Problem: {problem_info['title']}\n"
            + f"Time limit: {problem_info['time_limit_ms']}ms\n"
            + f"Memory limit: {problem_info['memory_limit_mb']}MB\n"
            + f"[Description]\n{problem_info['description']}\n"
            + f"[Input]\n{problem_info['input']}\n"
            + f"[Output]\n{problem_info['output']}\n"
            + "\n".join(
                [
                    f"[Sample Input {idx + 1}]\n{content[0]}\n[Sample Output {idx + 1}]\n{content[1]}"
                    for idx, content in enumerate(problem_info["examples"])
                ]
            )
            + f"\n[Note]\n{problem_info['note']}"
        )

        retry_attempts = 5
        retry_delay = 10

        token_count = 0
        full_response_content_for_history = ""
        reasoning_part = ""
        output_part_for_code_extraction = ""

        for attempt in range(retry_attempts):
            try:
                print(
                    f"-----Attempting API call (attempt {attempt + 1}/{retry_attempts}) for problem ID {problem_info.get('id')}"
                )
                api_messages = [{"role": "user", "content": self.problem_prompt}]

                if stream:
                    response_stream = self.generator.chat.completions.create(
                        model=self.model,
                        messages=api_messages,
                        stream=True,
                        temperature=0.2,
                    )

                    collected_chunks = []
                    usage_info = None

                    for chunk in response_stream:
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            collected_chunks.append(chunk.choices[0].delta.content)
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage_info = chunk.usage

                    full_response_content_for_history = "".join(collected_chunks)

                else:
                    response = self.generator.chat.completions.create(
                        model=self.model,
                        messages=api_messages,
                        temperature=0.2,
                    )
                    full_response_content_for_history = response.choices[
                        0
                    ].message.content
                    if response.usage:
                        token_count = response.usage.total_tokens

                reasoning_part, output_part_for_code_extraction = (
                    self.extract_reasoning(full_response_content_for_history)
                )

                break
            except openai.APIConnectionError as e:
                print(
                    f"APIConnectionError (problem {problem_info.get('id')}, attempt {attempt+1}): {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            except openai.RateLimitError as e:
                print(
                    f"RateLimitError (problem {problem_info.get('id')}, attempt {attempt+1}): {e}. Retrying in {retry_delay*2}s..."
                )
                time.sleep(retry_delay * 2)
            except openai.APIStatusError as e:
                print(
                    f"APIStatusError status={e.status_code} (problem {problem_info.get('id')}, attempt {attempt+1}): {e.response}. Retrying..."
                )
                time.sleep(retry_delay)
            except Exception as e:
                print(
                    f"Generic API error (problem {problem_info.get('id')}, attempt {attempt+1}): {str(e)}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
        else:
            print(
                f"Failed to get API response for problem {problem_info.get('id')} after {retry_attempts} attempts."
            )
            return "", "", 0, 0, 0

        history_entry_content = full_response_content_for_history
        if reasoning_history and reasoning_part:
            pass

        self.history = [
            {"role": "user", "content": self.problem_prompt},
            {"role": "assistant", "content": history_entry_content},
        ]

        extracted_code = self.extract_code_block(output_part_for_code_extraction)

        time.sleep(sleep)

        return (
            extracted_code,
            full_response_content_for_history,
            len(full_response_content_for_history),
            len(extracted_code),
            token_count,
        )

    def correct_code(
        self, correct_info, period, sleep=0, stream=False, reasoning_history=True
    ):
        suggestion_map = {
            "Runtime Error": "It means your code crashes or fails to execute properly. Check for issues like division by zero, out-of-bounds array access, or null pointer dereferences.",
            "Time Limit Exceeded": "Your code took too long. Analyze its time complexity. Consider more efficient algorithms or data structures. Optimize loops and I/O operations.",
            "Memory Limit Exceeded": "Your code used too much memory. Check for large data structures, memory leaks, or inefficient recursion. Optimize memory usage.",
            "Compilation Error": "Your code has syntax errors or missing includes. Carefully review the compiler messages and fix the indicated issues.",
            "Wrong Answer": "Your code's output is incorrect. Debug your logic. Test with edge cases. Ensure you understand the problem constraints and output format correctly.",
        }
        suggestion = suggestion_map.get(
            correct_info,
            "An unknown error occurred. Please review your code logic and try to identify the issue.",
        )

        correction_prompt = (
            f"The previous code attempt resulted in a '{correct_info}' during the '{period}' testing phase. "
            f"{suggestion} "
            "Please analyze the potential reasons for this error and provide a corrected C++ solution. "
            "Outline your thought process for the correction before presenting the code.\n"
            "In the end, YOU MUST provide the complete corrected code in a code block enclosed with ``` ```."
        )

        if not self.history:
            print("Error: correct_code called with empty history.")
            return "", "", 0, 0, 0

        current_history_for_api = self.history + [
            {"role": "user", "content": correction_prompt}
        ]

        retry_attempts = 3
        retry_delay = 10

        token_count = 0
        full_response_content_for_history = ""
        reasoning_part = ""
        output_part_for_code_extraction = ""

        for attempt in range(retry_attempts):
            try:
                if stream:
                    response_stream = self.generator.chat.completions.create(
                        model=self.model,
                        messages=current_history_for_api,
                        stream=True,
                        temperature=0.3,
                    )
                    collected_chunks = []
                    for chunk in response_stream:
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            collected_chunks.append(chunk.choices[0].delta.content)
                    full_response_content_for_history = "".join(collected_chunks)
                else:
                    response = self.generator.chat.completions.create(
                        model=self.model,
                        messages=current_history_for_api,
                        temperature=0.3,
                    )
                    full_response_content_for_history = response.choices[
                        0
                    ].message.content
                    if response.usage:
                        token_count = response.usage.total_tokens

                reasoning_part, output_part_for_code_extraction = (
                    self.extract_reasoning(full_response_content_for_history)
                )
                break
            except Exception as e:
                print(
                    f"API error during correction (attempt {attempt+1}): {str(e)}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
        else:
            print(
                f"Failed to get API response for correction after {retry_attempts} attempts."
            )
            return "", "", 0, 0, 0

        self.history.append({"role": "user", "content": correction_prompt})
        self.history.append(
            {"role": "assistant", "content": full_response_content_for_history}
        )

        extracted_code = self.extract_code_block(output_part_for_code_extraction)

        time.sleep(sleep)
        return (
            extracted_code,
            full_response_content_for_history,
            len(full_response_content_for_history),
            len(extracted_code),
            token_count,
        )

    def check_examples(self, code, problem_info):
        examples_to_check = problem_info.get("examples", [])
        if not examples_to_check:
            return "Accepted", "No examples provided to check."

        tmp_dir = ""
        try:
            tmp_dir = tempfile.mkdtemp(prefix="code_check_")

            gpp_compiler = "g++"
            cxx_flags_list = ["-std=c++23", "-O2", "-pipe", "-static", "-s"]

            src_file = os.path.join(tmp_dir, "example_main.cpp")
            exe_file = os.path.join(tmp_dir, "example_main_exe")
            if os.name == "nt":
                exe_file += ".exe"

            with open(src_file, "w", encoding="utf-8") as f:
                f.write(code)

            compile_cmd = [gpp_compiler, *cxx_flags_list, src_file, "-o", exe_file]
            compile_res = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=20
            )

            if compile_res.returncode != 0:
                err_msg = (
                    compile_res.stderr.strip()
                    if compile_res.stderr.strip()
                    else compile_res.stdout.strip()
                )
                return (
                    "Compilation Error",
                    f"Compile error:\n{err_msg if err_msg else 'Unknown compilation error'}",
                )

            time_limit_s = problem_info.get("time_limit_ms", 1000) / 1000.0
            mem_limit_mb_val = problem_info.get("memory_limit_mb", 256)

            for idx, example_pair in enumerate(examples_to_check):
                if not (
                    isinstance(example_pair, (list, tuple))
                    and len(example_pair) == 2
                    and isinstance(example_pair[0], str)
                    and isinstance(example_pair[1], str)
                ):
                    continue

                input_str, expected_output_str = example_pair

                try:
                    start_run_time = time.monotonic()
                    proc_example = subprocess.Popen(
                        [exe_file],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                        ),
                    )
                    actual_stdout, actual_stderr = proc_example.communicate(
                        input=input_str, timeout=time_limit_s + 0.5
                    )
                    run_duration_ms = (time.monotonic() - start_run_time) * 1000

                    if proc_example.returncode != 0:
                        return (
                            "Runtime Error",
                            f"Runtime Error on example {idx + 1}. Stderr:\n{actual_stderr[:500]}",
                        )

                    def normalize_local(text):
                        if text is None:
                            return []
                        lines = (
                            text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                        )
                        processed = [line.rstrip() for line in lines]
                        while processed and processed[-1] == "":
                            processed.pop()
                        return processed

                    actual_lines = normalize_local(actual_stdout)
                    expected_lines = normalize_local(expected_output_str)

                    if actual_lines != expected_lines:
                        return (
                            "Wrong Answer",
                            f"Wrong Answer on example {idx + 1}.\nActual:\n{actual_stdout[:500]}\nExpected:\n{expected_output_str[:500]}",
                        )

                except subprocess.TimeoutExpired:
                    proc_example.kill()
                    return (
                        "Time Limit Exceeded",
                        f"Time Limit Exceeded on example {idx + 1} (>{time_limit_s}s)",
                    )
                except Exception as e_run:
                    return "Runtime Error", f"Error running example {idx + 1}: {e_run}"

            return "Accepted", "All examples passed!"

        except Exception as e_outer:
            return "System Error", f"Error during example check setup: {e_outer}"
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)


# code_generator = CodeGenerator()
# problem_info = {
#     "source": "2023 ICPC World Finals",
#     "year": 2023,
#     "problem_id": "105255B",
#     "url": "https://codeforces.com/gym/105255/problem/B",
#     "title": "Schedule",
#     "type": "traditional",
#     "time_limit_ms": 2000,
#     "memory_limit_mb": 1024,
#     "description": 'The Institute for Creative Product Combinations (ICPC) tries to find unusual and innovative ways to unite seemingly unrelated products or technologies, opening up new markets and creating new jobs. (For instance, their most recent success was the "hairbachi," a hair-dryer with a hibachi grill top attachment for preparing on-the-go hot meals.) The company employs $n$ teams of size 2 to research individual products, then members of the different teams get together to explore ways of combining products.\n\nDuring the pandemic, the ICPC management organized everyone\'s schedule in such a way that there were never more than $n$ people in the office at the same time, and things ran so smoothly that they continued the process once things began to return to normal. Here is the scheme they used. Label the teams with integers 1 through $n$ and the two people on the $i^{\\text {th }}$ team as $(i, 1)$ and $(i, 2)$ for each $i$ from 1 to $n$. Each week, exactly one person from each team is allowed in the office, while the other has to stay away. The employees $(i, 1)$ and $(i, 2)$ know each other well and collaborate productively regardless of being isolated from each other, so members of the same team do not need to meet in person in the office. However, isolation between members from different teams is still a concern.\n\nEach pair of teams $i$ and $j$ for $i \\neq j$ has to collaborate occasionally. For a given number $w$ of weeks and for fixed team members $(i, a)$ and $(j, b)$, let $w_1<w_2<\\ldots<w_k$ be the weeks in which these two team members meet in the office. The isolation of those two people is the maximum of\n$$\n\\left\\{w_1, w_2-w_1, w_3-w_2, \\ldots, w_k-w_{k-1}, w+1-w_k\\right\\}\n$$\nor infinity if those two people never meet. The isolation of the whole company is the maximum isolation across all choices of $i, j, a$, and $b$.\n\nYou have been tasked to find a weekly schedule that minimizes the isolation of the whole company over a given number $w$ of weeks.',
#     "input": "The input consists of a single line containing two integers $n\\left(2 \\leq n \\leq 10^4\\right)$ and $w(1 \\leq w \\leq 52)$, where $n$ is the number of teams and $w$ is the number of weeks that need to be scheduled.",
#     "output": "Output a line containing either an integer representing the minimum isolation achievable for $n$ teams or the word infinity if no schedule guarantees that every pair of individuals on different teams can meet. If the isolation is finite, it is followed by $w$ lines representing a schedule that achieves this isolation. The $j^{\\text {th }}$ line of the schedule is a string of length $n$ containing only the symbols 1 and 2 , where the $i^{\\text {th }}$ symbol indicates which of the two members from team $i$ comes into the office on week $j$.",
#     "interaction": "",
#     "examples": [["2 6\n", "4\n11\n12\n21\n22\n11\n12\n"], ["2 1\n", "infinity\n"]],
#     "note": "",
# }
# code = code_generator.generate_code(problem_info)
# print(code)
# result, message = code_generator.check_examples(code, problem_info)
# print(result)
# print(message)
