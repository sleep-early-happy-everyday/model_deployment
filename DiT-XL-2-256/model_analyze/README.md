# DiT 模型结构分析

## torchinfo 

在项目根目录运行：

```bash
python model_analyze/analyze_dit.py
```

脚本使用 torchinfo 展示 DiT Transformer 的模块层级、输入输出形状和参数量，
并将报告保存到 `model_analyze/dit_summary.txt`。再次运行会覆盖该报告。

## `--depth` 参数

`depth` 控制报告展示模块嵌套结构的深度，默认值为 `3`，允许范围为 `1–6`（包含两端）的整数。
数值越大，展开的子模块越详细。
当前 DiT 模型的最大模块嵌套深度为 `6`，因此脚本将其设为上限；这不是 torchinfo 本身的限制。
传入范围外的值时，脚本会报参数错误并退出。

| 参数 | 展示内容示例 |
| --- | --- |
| `--depth 1` | 顶层模块，如 PatchEmbed、Transformer 模块列表和输出层 |
| `--depth 2` | 展开模块列表中的 28 个 Transformer Block |
| `--depth 3`（默认） | 展开每个 Block 内的 Attention、FeedForward 和归一化模块 |
| `--depth 5` | 进一步展开内部的线性层、嵌入层等细节 |
| `--depth 6` | 展开到当前模型最深层的子模块 |

例如，查看更深层的模块结构：

```bash
python model_analyze/analyze_dit.py --depth 5
```

`depth` 仅影响报告的展开层级，不改变模型结构或实际执行的层数。
无论设置为多少，脚本都会执行完整的 28 个 Transformer Block。
这里的“深度”指模块的嵌套层级，而不是 Transformer Block 的数量。
