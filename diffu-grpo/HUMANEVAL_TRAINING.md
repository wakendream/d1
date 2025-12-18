# HumanEval 训练说明

本文档说明如何在 diffu-GRPO 框架中使用 HumanEval 数据集进行强化学习训练。

## 已完成的集成

### 1. 数据加载 (`data_utils.py`)
- 添加了 `get_humaneval_questions()` 函数
- 使用专门的 `HUMANEVAL_SYSTEM_PROMPT`，要求模型以 `<reasoning>` 和 `<code>` 标签格式输出
- 返回包含 `prompt`, `solution`, `function_signature`, `docstring`, `task_id` 字段的数据集

### 2. 奖励函数 (`reward_func.py`)
实现了三个奖励函数用于代码生成任务：

#### `code_format_reward_func`
- 检查代码是否正确地包含在 `<code>` 标签中
- 奖励：0.5（有标签）+ 0.3（包含函数定义）= 总共最多 0.8

#### `code_extraction_reward_func`
- 检查是否能成功提取代码
- 奖励：1.0（成功提取）或 0.0（提取失败）

#### `humaneval_correctness_reward_func`
- 检查生成的代码与标准答案的匹配程度
- 使用字符串匹配（简化版本）
- 奖励：
  - 2.0：完全匹配
  - 1.0：部分匹配
  - 0.0：不匹配或无法提取代码

**注意**：当前实现的正确性奖励是基于字符串匹配的简化版本。对于更准确的评估，需要在后续版本中实现实际的代码执行和测试用例验证。

### 3. 训练脚本 (`diffu_grpo_train.py`)
- 添加了 `humaneval` 数据集支持
- 配置了相应的奖励函数组合
- 支持使用 HumanEval 的全部数据（164 个样本）进行训练

## 使用方法

### 基本训练命令

```bash
cd diffu-grpo

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path /path/to/your/model \
    --num_iterations 12 \
    --dataset humaneval \
    --run_name humaneval_base_bs12 \
    --output_dir checkpoints/humaneval_base_bs12 \
    --max_completion_length 512 \
    --block_length 64 \
    --diffusion_steps 128
```

### 使用提供的脚本

```bash
cd diffu-grpo
bash run_humaneval.sh
```

（记得先更新脚本中的 `MODEL_PATH`）

## 参数建议

对于 HumanEval 数据集（164 个样本），建议：

- **max_completion_length**: 512 或更长（代码生成通常需要较长序列）
- **block_length**: 64（与 gen_length 匹配）
- **diffusion_steps**: 128 或更多（代码生成可能需要更多步数）
- **batch_size**: 根据 GPU 内存调整（代码生成内存占用较大）
- **num_iterations**: 12（与其他数据集保持一致）

## 数据集特性

- **大小**: HumanEval 只有 164 个测试样本，这是一个较小的数据集
- **分割**: HumanEval 只提供 test split，所有数据用于训练
- **格式**: 每个样本包含函数签名、文档字符串和标准答案

## 奖励函数配置

当前使用的奖励函数组合：
```python
reward_functions = [
    code_format_reward_func,           # 格式奖励（0.8 max）
    code_extraction_reward_func,       # 提取奖励（1.0 max）
    humaneval_correctness_reward_func, # 正确性奖励（2.0 max）
]
```

总奖励范围：0.0 - 3.8

## 后续改进建议

1. **代码执行奖励**：实现实际的代码执行环境，运行测试用例验证代码正确性
2. **更智能的代码提取**：改进代码提取逻辑，处理更复杂的代码结构
3. **语法检查奖励**：添加 Python 语法检查，奖励语法正确的代码
4. **代码质量奖励**：考虑代码风格、可读性等因素

## 注意事项

1. HumanEval 数据集较小，可能需要与其他数据集混合训练或使用数据增强
2. 代码生成的序列通常较长，需要足够的 GPU 内存
3. 当前的正确性奖励是简化版本，建议在评估阶段使用完整的测试用例验证
4. 代码执行需要安全考虑，避免执行恶意代码

