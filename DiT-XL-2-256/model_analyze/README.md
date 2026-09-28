# DiT 模型结构分析

## 1 torchinfo 

在项目根目录运行：

```bash
python model_analyze/analyze_dit.py
```

脚本使用 torchinfo 展示 DiT Transformer 的模块层级、输入输出形状和参数量，
并将报告保存到 `model_analyze/dit_summary.txt`。再次运行会覆盖该报告。

### `--depth` 参数

`depth` 控制报告展示模块嵌套结构的深度，允许范围为 `1–6`（包含两端）的整数。
数值越大，展开的子模块越详细。
当前 DiT 模型的最大模块嵌套深度为 `6`，因此脚本将其设为上限；这不是 torchinfo 本身的限制。
传入范围外的值时，脚本会报参数错误并退出。

| 参数 | 展示内容示例 |
| --- | --- |
| `--depth 1` | 顶层模块，如 PatchEmbed、Transformer 模块列表和输出层 |
| `--depth 2` | 展开模块列表中的 28 个 Transformer Block |
| `--depth 3` | 展开每个 Block 内的 Attention、FeedForward 和归一化模块 |
| `--depth 5` | 进一步展开内部的线性层、嵌入层等细节 |
| `--depth 6`（默认） | 展开到当前模型最深层的子模块 |

> `depth` 仅影响报告的展开层级，不改变模型结构或实际执行的层数。
> 无论设置为多少，脚本都会执行完整的 28 个 Transformer Block。
> 这里的“深度”指模块的嵌套层级，而不是 Transformer Block 的数量。

### 模型加载代码

下面这段代码从本地加载 DiT 模型结构和训练好的权重，将模型放到指定设备，并切换到评估模式：

```python
model = DiTTransformer2DModel.from_pretrained(
    project_dir / "weights" / "transformer",
    torch_dtype=dtype,
    local_files_only=True,
    use_safetensors=False,
).to(device).eval()
```

| 代码 | 含义 |
| --- | --- |
| `DiTTransformer2DModel` | Diffusers 提供的 DiT Transformer 模型类，这里只加载 Transformer 部分，不包含 VAE 和调度器 |
| `from_pretrained(...)` | 根据目录中的配置创建模型，并加载训练好的权重 |
| `project_dir / "weights" / "transformer"` | 用 `Path` 拼接本地路径，即项目下的 `weights/transformer` |
| `torch_dtype=dtype` | 指定权重的数据类型；当前脚本在 GPU 上使用 `float16`，在 CPU 上使用 `float32` |
| `local_files_only=True` | 只读取本地文件，不从 Hugging Face 下载；缺少必要文件会报错 |
| `use_safetensors=False` | 使用 PyTorch 格式的权重文件，而不使用 `.safetensors` 文件 |
| `.to(device)` | 将模型参数和缓冲区移动到指定的 GPU 或 CPU |
| `.eval()` | 切换到评估模式，使 Dropout 等模块采用推理时的行为 |

这三个连续步骤可以拆开理解：

```python
# 1. 创建模型并加载本地权重（参数同上）
model = DiTTransformer2DModel.from_pretrained(...)

# 2. 移动到计算设备
model = model.to(device)

# 3. 设置为评估模式
model.eval()
```

`.eval()` 不会关闭梯度计算，也不会冻结参数。脚本后面的
`with torch.inference_mode():` 才负责在分析时关闭梯度记录，减少额外开销。
