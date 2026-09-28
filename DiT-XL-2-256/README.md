# DiT-XL-2-256 本地推理

## 0 实验环境

- NVIDIA RTX 2080Ti
- Intel CPU

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

* 参考
    * [facebook/DiT-XL-2-256](https://huggingface.co/facebook/DiT-XL-2-256)
    * [论文：Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748)

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

