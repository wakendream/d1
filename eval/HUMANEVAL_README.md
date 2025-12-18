# HumanEval 评估说明

这个文档说明如何使用 HumanEval 数据集进行代码生成的评估。

## 文件说明

- `humaneval.py`: HumanEval 数据集的 Dataset 类实现
- `eval.py`: 更新后的评估脚本，支持 HumanEval 数据集
- `test_humaneval.sh`: 用于快速测试的评估脚本

## 使用方法

### 基本评估命令

```bash
cd eval
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node 1 \
    --master_port 29411 \
    eval.py \
    --dataset humaneval \
    --model_path /path/to/your/model \
    --batch_size 2 \
    --gen_length 512 \
    --block_length 64 \
    --diffusion_steps 128 \
    --output_dir eval_results/humaneval \
    --add_reasoning
```

### 使用测试脚本

```bash
cd eval
bash test_humaneval.sh /path/to/your/model
```

### 参数说明

- `--dataset humaneval`: 指定使用 HumanEval 数据集
- `--model_path`: 模型路径（必需）
- `--batch_size`: 批次大小（建议 2-4，根据 GPU 内存调整）
- `--gen_length`: 生成长度（代码生成建议 512 或更长）
- `--block_length`: 扩散块的长度（建议 64）
- `--diffusion_steps`: 扩散步数（建议 128 或更多）
- `--output_dir`: 输出目录，生成的 JSON 文件会保存在这里
- `--add_reasoning`: 添加推理前缀（`<reasoning>`）

## 输出格式

评估脚本会生成 JSON 文件，包含以下字段：

```json
{
  "generations": [
    {
      "task_id": "HumanEval/0",
      "function_signature": "def f(...)",
      "docstring": "...",
      "prompt_input": "...",
      "generations": "模型生成的完整文本",
      "canonical_solution": "标准答案"
    }
  ],
  "metrics": {
    "wall_time": 平均生成时间,
    "total_processed": 处理的样本数
  },
  "model_path": "...",
  "gen_length": 512,
  "diffusion_steps": 128,
  "block_length": 64
}
```

## 注意事项

1. **生成长度**: 代码生成通常需要较长的序列（512+ tokens），请根据实际情况调整 `gen_length`
2. **批次大小**: 代码生成的内存占用较大，建议使用较小的 batch_size（2-4）
3. **扩散参数**: 代码生成可能需要更多的扩散步数以获得更好的质量
4. **评估指标**: 当前的评估脚本只保存生成结果，需要使用单独的解析脚本来计算 pass@k 等指标

## 下一步

评估脚本只生成模型输出，需要：
1. 从生成文本中提取代码（解析 `<code>` 标签或提取函数实现）
2. 运行单元测试验证代码正确性
3. 计算 pass@k 指标

可以参考 HumanEval 官方的评估代码来实现这些功能。

