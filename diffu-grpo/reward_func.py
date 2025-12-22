import numpy as np
import re
from math500_utils import remove_boxed, last_boxed_only_string, is_equiv, boxed_in_answer


def extract_completion_content(completion) -> str:
    """
    Extract content from completion, handling both string and dict formats.
    
    Args:
        completion: Can be either:
            - A string: "content text"
            - A list with dict: [{"role": "assistant", "content": "content text"}]
            - A dict: {"role": "assistant", "content": "content text"}
    
    Returns:
        str: The content text
    """
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list) and len(completion) > 0:
        if isinstance(completion[0], dict) and "content" in completion[0]:
            return completion[0]["content"]
        elif isinstance(completion[0], str):
            return completion[0]
    elif isinstance(completion, dict) and "content" in completion:
        return completion["content"]
    # Fallback: try to convert to string
    return str(completion)


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [extract_completion_content(completion) for completion in completions]
    q = prompts[0][-1]["content"] if isinstance(prompts[0], list) and len(prompts[0]) > 0 and isinstance(prompts[0][-1], dict) else str(prompts[0])
    extracted_responses = [extract_xml_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = []
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    )
    print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards


def extract_code_from_response(response_text: str) -> str | None:
    """
    Extract code from response text, looking for <code> tags or function definitions.
    Returns the extracted code string, or None if no code is found.
    """
    # Try to extract from <code> tags first
    code_pattern = r"<code>(.*?)</code>"
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    if matches:
        code = matches[-1].strip()  # Return the last match (most likely the final code)
        if code and len(code) > 0:
            return code
    
    # If no <code> tags, try to extract everything after <reasoning>...</reasoning>
    # This assumes the format is <reasoning>...</reasoning><code>...</code> or just code
    reasoning_pattern = r"</reasoning>\s*(.*?)(?=\n\n|$)"
    matches = re.findall(reasoning_pattern, response_text, re.DOTALL)
    if matches:
        potential_code = matches[-1].strip()
        # Check if it looks like code (contains def, import, or other Python keywords)
        if "def " in potential_code or "import " in potential_code or "class " in potential_code:
            return potential_code
    
    # Last resort: try to find function definition
    # Look for "def " keyword and extract until the end or next blank line followed by non-code
    def_pattern = r"(def\s+\w+\s*\([^)]*\):.*?)(?=\n\n\s*[A-Z]|\n```|$)"
    matches = re.findall(def_pattern, response_text, re.DOTALL)
    if matches:
        code = matches[0].strip()
        if code and len(code) > 10:  # Make sure it's not too short
            return code
    
    return None


def code_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for code format: checks if code is properly formatted in <code> tags."""
    responses = [extract_completion_content(completion) for completion in completions]
    rewards = []
    
    for response in responses:
        reward = 0.0
        # Check for <code> tags
        if "<code>" in response and "</code>" in response:
            reward += 0.5
            # Extract code and check if it contains function definition
            code = extract_code_from_response(response)
            if code and "def " in code:
                reward += 0.3
        rewards.append(reward)
    
    return rewards


def code_extraction_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for successful code extraction."""
    responses = [extract_completion_content(completion) for completion in completions]
    rewards = []
    
    for response in responses:
        code = extract_code_from_response(response)
        reward = 1.0 if code is not None and len(code.strip()) > 0 else 0.0
        rewards.append(reward)
    
    return rewards


import ast
import io
import sys
import signal
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

def check_correctness(problem: dict, completion: str, timeout: float = 10.0) -> dict:
    """
    Check if the completion passes the tests for a given problem.
    
    Args:
        problem: A dict containing:
            - 'task_id': str
            - 'prompt': str (function signature)
            - 'test': str (test code)
            - 'entry_point': str (function name)
        completion: The generated code string
        timeout: Timeout in seconds (not fully implemented, but can use signal)
    
    Returns:
        dict with keys:
            - 'passed': bool
            - 'result': str ('passed', 'failed', 'error')
            - 'error': Optional[str] (error message if any)
    """
    try:
        # Extract the function signature from prompt
        # The prompt usually ends with the function signature
        signature = problem.get('prompt', '').strip()
        
        # Combine signature and completion to form the complete function
        # Remove any leading whitespace from completion
        completion_clean = completion.strip()
        
        # Construct the full code
        # The completion should be the function body, but sometimes includes the full function
        # We need to handle both cases
        if completion_clean.startswith('def '):
            # Full function definition
            full_code = completion_clean
        else:
            # Just the function body, need to combine with signature
            # Extract function name from signature
            if 'def ' in signature:
                # If signature already has 'def', use it
                full_code = signature + '\n' + '    ' + completion_clean.replace('\n', '\n    ')
            else:
                full_code = completion_clean
        
        # Add the test code
        test_code = problem.get('test', '')
        code_to_execute = full_code + '\n\n' + test_code
        
        # Execute the code in a safe namespace
        namespace = {}
        exec(code_to_execute, namespace)
        
        # Check if there's a result (test code usually has assertions)
        # Most HumanEval tests use assert statements which will raise AssertionError on failure
        # If we get here without exception, tests passed (unless test uses different pattern)
        return {
            'passed': True,
            'result': 'passed',
            'error': None
        }
        
    except SyntaxError as e:
        return {
            'passed': False,
            'result': 'error',
            'error': f'SyntaxError: {str(e)}'
        }
    except AssertionError:
        # Test failed
        return {
            'passed': False,
            'result': 'failed',
            'error': 'Assertion failed'
        }
    except Exception as e:
        return {
            'passed': False,
            'result': 'error',
            'error': f'{type(e).__name__}: {str(e)}'
        }


def humaneval_correctness_reward_func(
    prompts, completions, solution, step=None, run_name=None, **kwargs
) -> list[float]:
    """
    Reward function for HumanEval: executes generated code and runs test cases.
    Uses actual code execution instead of string matching.
    """
    responses = [extract_completion_content(completion) for completion in completions]
    extracted_codes = [extract_code_from_response(r) for r in responses]
    
    # Get test information from kwargs (passed from dataset)
    # HumanEval dataset should have 'test' field with test cases
    test_codes = kwargs.get('test', [])
    function_signatures = kwargs.get('function_signature', [])
    entry_points = kwargs.get('entry_point', [])
    task_ids = kwargs.get('task_id', [])
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    rewards = []
    for i, extracted_code in enumerate(extracted_codes):
        if extracted_code is None:
            reward = 0.0
        else:
            # Construct problem dict for check_correctness
            problem = {
                'task_id': task_ids[i] if i < len(task_ids) else f'task_{i}',
                'prompt': function_signatures[i] if i < len(function_signatures) else '',
                'test': test_codes[i] if i < len(test_codes) else '',
                'entry_point': entry_points[i] if i < len(entry_points) else 'f'
            }
            
            # Check correctness by executing code
            result = check_correctness(problem, extracted_code)
            
            if result['passed']:
                reward = 2.0  # Full reward for passing all tests
            else:
                # No reward if tests fail or error occurs
                reward = 0.0
        
        rewards.append(reward)
        
        # Print sample for debugging
        if i == 0 and step is not None and step % 100 == 0:
            print(
                "-" * 20,
                f"\n{RED}Task ID:{RESET} {task_ids[i] if i < len(task_ids) else 'N/A'}\n",
                "-" * 20,
                f"\n{GREEN}Function Signature:{RESET}\n{function_signatures[i] if i < len(function_signatures) else 'N/A'}\n",
                "-" * 20,
                f"\n{BLUE}Generated Code:{RESET}\n{extracted_code[:300] if extracted_code else 'None'}\n",
                "-" * 20,
                f"\n{YELLOW}Test Result:{RESET} {result.get('result', 'unknown')}\n",
            )
            if result.get('error'):
                print(f"{RED}Error:{RESET} {result['error']}\n")
            print("✅" if reward > 0 else "❌")
    
    return rewards


def _timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Code execution timed out")


def check_correctness_mbpp(problem: dict, completion: str, timeout: float = 20.0) -> dict:
    """
    Check if the completion passes the tests for a given MBPP problem.
    Similar to check_correctness but adapted for MBPP format.
    
    Args:
        problem: A dict containing:
            - 'task_id': str
            - 'test': str (combined test code with setup)
            - 'test_setup_code': str (optional setup code)
        completion: The generated code string
        timeout: Timeout in seconds (default: 10.0)
    
    Returns:
        dict with keys:
            - 'passed': bool
            - 'result': str ('passed', 'failed', 'error')
            - 'error': Optional[str] (error message if any)
    """
    # Set up timeout handler
    old_handler = None
    try:
        if timeout > 0:
            # Set signal handler for timeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout))
        
        try:
            completion_clean = completion.strip()
            
            # For MBPP, the completion should be a complete function or script
            # Combine with test setup code if available
            test_setup = problem.get('test_setup_code', '')
            test_code = problem.get('test', '')
            
            # Construct the full code to execute
            if test_setup:
                full_code = test_setup + '\n\n' + completion_clean + '\n\n' + test_code
            else:
                full_code = completion_clean + '\n\n' + test_code
            
            # Execute the code in a safe namespace
            namespace = {}
            exec(full_code, namespace)
            
            # Cancel alarm if execution completes successfully
            if timeout > 0:
                signal.alarm(0)
            
            # If we get here without exception, tests passed
            return {
                'passed': True,
                'result': 'passed',
                'error': None
            }
            
        except TimeoutError:
            return {
                'passed': False,
                'result': 'error',
                'error': f'Timeout: Code execution exceeded {timeout} seconds'
            }
        except SyntaxError as e:
            return {
                'passed': False,
                'result': 'error',
                'error': f'SyntaxError: {str(e)}'
            }
        except AssertionError:
            # Test failed
            return {
                'passed': False,
                'result': 'failed',
                'error': 'Assertion failed'
            }
        except Exception as e:
            return {
                'passed': False,
                'result': 'error',
                'error': f'{type(e).__name__}: {str(e)}'
            }
    finally:
        # Restore original signal handler and cancel alarm
        if timeout > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def mbpp_correctness_reward_func(
    prompts, completions, solution, step=None, run_name=None, **kwargs
) -> list[float]:
    """
    Reward function for MBPP: executes generated code and runs test cases.
    Uses actual code execution instead of string matching.
    Similar to HumanEval but adapted for MBPP dataset structure.
    """
    responses = [extract_completion_content(completion) for completion in completions]
    extracted_codes = [extract_code_from_response(r) for r in responses]
    
    # Get test information from kwargs (passed from dataset)
    # MBPP dataset has 'test' field with combined test code
    test_codes = kwargs.get('test', [])
    task_ids = kwargs.get('task_id', [])
    test_setup_codes = kwargs.get('test_setup_code', [])
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    rewards = []
    for i, extracted_code in enumerate(extracted_codes):
        if extracted_code is None:
            reward = 0.0
            result = {'passed': False, 'result': 'error', 'error': 'No code extracted'}
        else:
            # Construct problem dict for check_correctness_mbpp
            problem = {
                'task_id': task_ids[i] if i < len(task_ids) else f'task_{i}',
                'test': test_codes[i] if i < len(test_codes) else '',
                'test_setup_code': test_setup_codes[i] if i < len(test_setup_codes) else ''
            }
            
            # Check correctness by executing code
            result = check_correctness_mbpp(problem, extracted_code)
            
            if result['passed']:
                reward = 2.0  # Full reward for passing all tests
            else:
                # No reward if tests fail or error occurs
                reward = 0.0
        
        rewards.append(reward)
        
        # Print sample for debugging
        if i == 0 and step is not None and step % 100 == 0:
            print(
                "-" * 20,
                f"\n{RED}Task ID:{RESET} {task_ids[i] if i < len(task_ids) else 'N/A'}\n",
                "-" * 20,
                f"\n{GREEN}Problem Text:{RESET}\n{kwargs.get('text', ['N/A'])[i] if i < len(kwargs.get('text', [])) else 'N/A'}\n",
                "-" * 20,
                f"\n{BLUE}Generated Code:{RESET}\n{extracted_code[:300] if extracted_code else 'None'}\n",
                "-" * 20,
                f"\n{YELLOW}Test Result:{RESET} {result.get('result', 'unknown')}\n",
            )
            if result.get('error'):
                print(f"{RED}Error:{RESET} {result['error']}\n")
            print("✅" if reward > 0 else "❌")
    
    return rewards