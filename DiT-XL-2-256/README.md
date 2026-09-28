# DiT-XL-2-256 本地推理

使用 uv 来管理依赖库。

```bash
uv sync --locked
```

激活 uv 环境：

```bash
source .venv/bin/activate
```

模型文件保存在 `weights/`。如果尚未下载，执行：

```bash
hf download facebook/DiT-XL-2-256 --revision main --local-dir ./weights
```

参考：[facebook/DiT-XL-2-256](https://huggingface.co/facebook/DiT-XL-2-256)

## 1 生成图片

此模型按 ImageNet 的 1000 个类别生成 256×256 图片，不接受自由文本提示词。
默认生成一张金毛犬图片：

```bash
python generate.py
```

输出位置为 `outputs/golden_retriever.png`。脚本从本地加载权重，使用 GPU 和 FP16，
通过 DPM-Solver 进行 50 步采样。

可以设置类别、步数、随机种子和输出位置：

```bash
python generate.py --class-id 1 --steps 50 --seed 123 --output outputs/goldfish.png
```

| 类别 ID | 内容 |
| --- | --- |
| 1 | 金鱼 |
| 207 | 金毛犬 |
| 980 | 火山 |

完整类别名称见 `weights/model_index.json` 中的 `id2label`。
同一输出路径再次运行会覆盖已有图片。

参考：[Diffusers DiT 文档](https://huggingface.co/docs/diffusers/api/pipelines/dit)。


## 分析模型结构

使用 torchinfo 分析本地 DiT Transformer，显示各模块的输入输出形状和参数量：

```bash
uv run python model_analyze/analyze_dit.py
```

报告同时保存到 `model_analyze/dit_summary.txt`，再次运行会覆盖该文件。
默认展示 3 层模块；使用 `--depth 5` 查看 Attention、MLP 等模块的更深层结构。
有 CUDA 时使用 GPU/FP16，否则使用 CPU/FP32。

分析执行 batch size 为 1 的单次 Transformer 前向计算，输入为 `[1, 4, 32, 32]`
的潜变量、时间步和类别 ID；不包含 VAE、采样循环或 CFG 的批次翻倍。
torchinfo 显示的 Mult-Adds 和内存估算不代表完整 FLOPs 或实测峰值显存。



