# --- Add this inside eval.py or in a separate datasets.py ---
from human_eval.data import read_problems
import torch
from transformers import AutoTokenizer


class HumanEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        max_length=1024,
        num_examples=0,
        subsample=-1,  # -1 means use all
        system_prompt=None,      # 可选：加其他可能传入的参数（如 system_prompt）
        add_reasoning=False,     # 可选：保持签名一致
        **kwargs            
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.load_dataset()
        
        if subsample != -1:
            import numpy as np
            self.subsample_indices = np.random.choice(len(self.dataset), subsample, replace=False)
            print(f"Subsampling HumanEval to {len(self.subsample_indices)} examples")
        else:
            self.subsample_indices = list(range(len(self.dataset)))
        print(f"Evaluating {len(self.subsample_indices)} HumanEval examples")

    def load_dataset(self):
        """Load official HumanEval problems as list of dicts"""
        raw_problems = read_problems()  # dict: task_id -> problem
        self.dataset = list(raw_problems.values())  # list of problem dicts
        self.task_ids = list(raw_problems.keys())

    def extract_function_signature_and_docstring(self, prompt: str):
        """Split prompt into function signature and docstring"""
        lines = prompt.strip().split("\n")
        if len(lines) == 0:
            return "", ""
        func_sig = lines[0].rstrip(":")  # e.g., "def add(a, b)"
        docstring = "\n".join(lines[1:]).strip()
        return func_sig, docstring

    def __len__(self):
        return len(self.subsample_indices)

    def __getitem__(self, idx):
        real_idx = self.subsample_indices[idx]
        problem = self.dataset[real_idx]
        task_id = self.task_ids[real_idx]
        prompt = problem["prompt"]
        canonical_solution = problem["canonical_solution"]

        func_sig, docstring = self.extract_function_signature_and_docstring(prompt)

        return {
            "prompt": prompt,
            "function_signature": func_sig,
            "docstring": docstring,
            "canonical_solution": canonical_solution,
            "task_id": task_id,
        }

    def collate_fn(self, batch):
        prompts = [item["prompt"] for item in batch]
        function_signatures = [item["function_signature"] for item in batch]
        docstrings = [item["docstring"] for item in batch]
        canonical_solutions = [item["canonical_solution"] for item in batch]
        task_ids = [item["task_id"] for item in batch]

        # Tokenize with left padding (common for decoder-only models)
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding_side="left",  # important for generation
        )

        return {
            "input_ids": tokenized.input_ids,
            "prompts": prompts,
            "function_signatures": function_signatures,
            "docstrings": docstrings,
            "canonical_solutions": canonical_solutions,
            "task_ids": task_ids,
        }