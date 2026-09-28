# DiT-XL-2-256 本地推理

使用 Linux、Python 3.12.3、uv 和 NVIDIA GPU。直接依赖使用 `==` 固定版本，
完整依赖记录在 `uv.lock` 中；PyTorch 使用官方 CUDA 12.6 软件源。

```bash
uv sync --locked
```

模型文件保存在 `weights/`。如果尚未下载，执行：

```bash
uv run hf download facebook/DiT-XL-2-256 --revision main --local-dir ./weights
```

## 生成图片

此模型按 ImageNet 的 1000 个类别生成 256×256 图片，不接受自由文本提示词。
默认生成一张金毛犬图片：

```bash
uv run python generate.py
```

输出位置为 `outputs/golden_retriever.png`。脚本从本地加载权重，使用 GPU 和 FP16，
通过 DPM-Solver 进行 50 步采样。

可以设置类别、步数、随机种子和输出位置：

```bash
uv run python generate.py --class-id 1 --steps 50 --seed 123 --output outputs/goldfish.png
```

| 类别 ID | 内容 |
| --- | --- |
| 1 | 金鱼 |
| 207 | 金毛犬 |
| 980 | 火山 |

完整类别名称见 `weights/model_index.json` 中的 `id2label`。
同一输出路径再次运行会覆盖已有图片。

参考：[Diffusers DiT 文档](https://huggingface.co/docs/diffusers/api/pipelines/dit)。
