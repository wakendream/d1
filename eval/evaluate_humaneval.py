
#!/usr/bin/env python3
import os
import json
import re
import argparse
from human_eval.evaluation import evaluate_functional_correctness

def extract_function_code(completion: str) -> str:
    # 移除 Markdown 代码块
    completion = re.sub(r"```(?:python)?\s*", "", completion, flags=re.IGNORECASE)
    completion = re.sub(r"```.*", "", completion, flags=re.IGNORECASE)
    
    # 移除解释性文本（如 "### Explanation", "Here is the code:" 等）
    lines = completion.split("\n")
    code_lines = []
    in_code = False
    
    for line in lines:
        stripped = line.strip()
        # 跳过空行和解释性注释
        if stripped == "" or stripped.startswith("#") and any(kw in stripped.lower() for kw in ["explanation", "note", "test"]):
            continue
        # 跳过 if __name__ 和 doctest
        if stripped.startswith("if __name__") or "doctest" in stripped:
            break
        
        # 如果遇到 def，说明模型重复写了函数签名 → 跳过这一行，后面才是函数体
        if stripped.startswith("def "):
            in_code = True
            continue  # 跳过 def 行
        
        # 如果还没遇到 def，但已经有代码 → 假设是函数体（需缩进）
        if not in_code and stripped != "":
            in_code = True
        
        if in_code:
            code_lines.append(line)
        elif stripped == "":
            continue
        else:
            # 遇到非空非def行，且不在函数体内 → 可能是解释，跳过
            continue

    # 确保函数体有缩进（至少 4 空格或 1 tab）
    if code_lines:
        first_line = code_lines[0].lstrip()
        if first_line and not code_lines[0].startswith((" ", "\t")):
            # 如果第一行没缩进，自动加 4 空格
            code_lines = ["    " + line if line.strip() else line for line in code_lines]

    return "\n".join(code_lines).rstrip()

def load_samples_from_dir(input_dir):
    samples = []
    for filename in os.listdir(input_dir):
        if not filename.endswith("_generations.json"):
            continue
        path = os.path.join(input_dir, filename)
        print(f"Loading: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("generations", []):
                raw_code = item["generations"]
                cleaned_code = extract_function_code(raw_code)
                samples.append({
                    "task_id": item["task_id"],
                    "completion": cleaned_code
                })
    return samples


def write_jsonl(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--temp_file", type=str, default="samples_temp.jsonl")
    parser.add_argument("--k", type=str, default="1")
    args = parser.parse_args()

    samples = load_samples_from_dir(args.input_dir)
    print(f"Total samples: {len(samples)}")

    # 可选：打印前几个看看清洗效果
    print("\n--- Sample of cleaned completions ---")
    for i in range(min(3, len(samples))):
        print(f"[{samples[i]['task_id']}]")
        print(samples[i]["completion"])
        print("-" * 50)

    write_jsonl(samples, args.temp_file)
    k_list = [int(k) for k in args.k.split(",")]

    results = evaluate_functional_correctness(
        sample_file=args.temp_file,
        k=k_list,
        n_workers=4,
        timeout=3.0
    )

    print("\n✅ Evaluation Results:")
    print(json.dumps(results, indent=2))

    with open("humaneval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    os.remove(args.temp_file)


if __name__ == "__main__":
    main()