# MBPP 训练说明

本文档说明如何在 diffu-GRPO 框架中使用 MBPP (Mostly Basic Python Problems) 数据集进行强化学习训练。

## 已完成的集成

### 1. 数据加载 (`data_utils.py`)
- 添加了 `get_mbpp_questions()` 函数
- 使用专门的 `MBPP_SYSTEM_PROMPT`，要求模型以 `<reasoning>` 和 `<code>` 标签格式输出
- 返回包含 `prompt`, `solution`, `text`, `task_id`, `test`, `test_list`, `test_setup_code`, `entry_point` 字段的数据集
- MBPP 数据集有 `train` 和 `test` 两个分割，训练时使用 `train` 分割

### 2. 奖励函数 (`reward_func.py`)
实现了三个奖励函数用于代码生成任务：

#### `code_format_reward_func`
- 检查代码是否正确地包含在 `<code>` 标签中
- 奖励：0.5（有标签）+ 0.3（包含函数定义）= 总共最多 0.8

#### `code_extraction_reward_func`
- 检查是否能成功提取代码
- 奖励：1.0（成功提取）或 0.0（提取失败）

#### `mbpp_correctness_reward_func`
- 执行生成的代码并运行测试用例
- 使用 `check_correctness_mbpp` 函数来执行代码和测试
- 奖励：
  - 2.0：所有测试通过
  - 0.0：测试失败或出现错误

**注意**：MBPP 的测试用例格式与 HumanEval 类似，都是 assert 语句。奖励函数会执行生成的代码并运行所有测试用例。

### 3. 训练脚本 (`diffu_grpo_train.py`)
- 添加了 `mbpp` 数据集支持
- 配置了相应的奖励函数组合
- 使用 MBPP 的 train 分割进行训练（约 374 个样本）

## 使用方法

### 基本训练命令

```bash
cd diffu-grpo

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12347 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path /path/to/your/model \
    --num_iterations 12 \
    --dataset mbpp \
    --run_name mbpp_base_bs12 \
    --output_dir checkpoints/mbpp_base_bs12 \
    --max_completion_length 512 \
    --block_length 64 \
    --diffusion_steps 128
```

### 使用提供的脚本

```bash
cd diffu-grpo
bash run_mbpp.sh
```

（记得先更新脚本中的 `MODEL_PATH`）

## 参数建议

对于 MBPP 数据集（train split 约 374 个样本），建议：

- **max_completion_length**: 512 或更长（代码生成通常需要较长序列）
- **block_length**: 64（与 gen_length 匹配）
- **diffusion_steps**: 128 或更多（代码生成可能需要更多步数）
- **batch_size**: 根据 GPU 内存调整（代码生成内存占用较大）
- **num_iterations**: 12（与其他数据集保持一致）

## 数据集特性

- **大小**: MBPP train split 有约 374 个样本，test split 有约 90 个样本
- **分割**: MBPP 提供 train 和 test 两个分割
- **格式**: 每个样本包含问题描述、标准答案代码和测试用例列表
- **测试用例**: MBPP 的测试用例是 assert 语句列表，包含测试设置代码

## 奖励函数配置

当前使用的奖励函数组合：
```python
reward_functions = [
    code_format_reward_func,        # 格式奖励（0.8 max）
    code_extraction_reward_func,    # 提取奖励（1.0 max）
    mbpp_correctness_reward_func,   # 正确性奖励（2.0 max）
]
```

总奖励范围：0.0 - 3.8

## 与 HumanEval 的区别

1. **数据集大小**: MBPP 的训练集比 HumanEval 大（374 vs 164）
2. **数据格式**: MBPP 包含问题描述文本，而 HumanEval 主要依赖函数签名和文档字符串
3. **测试用例**: MBPP 的测试用例可能包含测试设置代码（test_setup_code）
4. **奖励函数**: MBPP 使用专门的 `check_correctness_mbpp` 函数来处理测试设置代码

## 后续改进建议

1. **代码执行安全性**: 改进代码执行环境，确保安全性
2. **更智能的代码提取**: 改进代码提取逻辑，处理更复杂的代码结构
3. **语法检查奖励**: 添加 Python 语法检查，奖励语法正确的代码
4. **代码质量奖励**: 考虑代码风格、可读性等因素
5. **部分正确性奖励**: 对于部分测试通过的情况，给予部分奖励

## 注意事项

1. MBPP 数据集比 HumanEval 大，训练时间可能更长
2. 代码生成的序列通常较长，需要足够的 GPU 内存
3. 代码执行需要安全考虑，避免执行恶意代码
4. MBPP 的测试用例可能包含导入语句和设置代码，需要正确处理
5. 建议在训练前先验证数据加载和奖励函数是否正常工作

