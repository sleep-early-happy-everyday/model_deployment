# DiT 模型结构

## CombinedTimestepLabelEmbeddings

这个模块继承自 PyTorch `nn.Module`，将扩散时间步和类别标签分别转换为向量，
再逐元素相加，得到供 DiT 使用的条件向量。
源码位于项目虚拟环境的 `.venv/lib/python3.12/site-packages/diffusers/models/embeddings.py` 中。

以下用 `N` 表示批大小（batch size），用 `D` 表示 `embedding_dim`。当前模型中 `D = 1152`，
分析脚本使用 `N = 1`、时间步 `500` 和类别编号 `207`（金毛犬）。

### 1. `__init__`：创建子模块

初始化方法负责创建时间步编码、时间步嵌入网络和类别嵌入表，供后续前向计算使用。

```python
class CombinedTimestepLabelEmbeddings(nn.Module):
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)
```

**函数参数**

| 参数 | 含义 | 当前模型使用的值 |
| --- | --- | --- |
| `num_classes` | 实际类别数量，不包含额外的无条件标签 | `1000` |
| `embedding_dim` | 时间步嵌入和类别嵌入的共同维度 `D` | `1152` |
| `class_dropout_prob` | 训练时将每个样本的类别标签替换为无条件标签的概率 | 默认 `0.1`，即 10% |

**初始化过程**

`super().__init__()` 调用 `nn.Module` 的初始化方法，
使后续赋给 `self` 的子模块及其参数能够被正确注册，参与设备迁移和训练/评估状态切换。

随后创建三个子模块：

| 子模块 | 作用 | 输入 → 输出形状 | 是否有可训练参数 |
| --- | --- | --- | --- |
| `self.time_proj` | 使用固定的正弦/余弦函数编码时间步 | `(N,) → (N, 256)` | 否 |
| `self.timestep_embedder` | 将时间步编码映射到模型的隐藏维度 | `(N, 256) → (N, D)` | 是 |
| `self.class_embedder` | 根据类别编号查询嵌入表 | `(N,) → (N, D)` | 是 |

**`time_proj` 的编码参数**

| 参数 | 含义 |
| --- | --- |
| `num_channels=256` | 输出 256 个特征：128 个余弦特征和 128 个正弦特征 |
| `flip_sin_to_cos=True` | 将输出排列为前半部分余弦、后半部分正弦 |
| `downscale_freq_shift=1` | 调整编码频率的间隔；当前频率公式的分母为 `256 / 2 - 1 = 127` |

具体地，对 `i = 0, ..., 127`，频率为 `exp(-log(10000) * i / 127)`，
输出按顺序拼接 `cos(t * frequency)` 和 `sin(t * frequency)`。
`downscale_freq_shift` 调整的是频率计算，并不将输入时间步减 1。

**`timestep_embedder` 的网络结构**

这里使用默认的 SiLU 激活函数，结构为：

```text
Linear(256, D) → SiLU → Linear(D, D)
```

它通过可训练的线性层，将固定的时间步编码转换为适合模型使用的条件特征。

**`class_embedder` 的嵌入表与标签丢弃**

当 `class_dropout_prob > 0` 时，嵌入表额外预留一个无条件标签，
所以当前嵌入表形状为 `(1001, 1152)`：编号 `0–999` 表示实际类别，编号 `1000` 表示无条件标签。

训练时，以指定概率将整个类别编号替换为无条件标签，
用于训练无分类器引导（Classifier-Free Guidance，CFG）所需的无条件分支。
这里的标签丢弃不是将嵌入向量中的部分元素置零。

### 2. `forward`：计算条件向量

前向方法使用初始化时创建的三个子模块，依次计算时间步嵌入、类别嵌入，并融合两者。

```python
    def forward(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        class_labels = self.class_embedder(class_labels)  # (N, D)

        conditioning = timesteps_emb + class_labels  # (N, D)

        return conditioning
```

**输入与返回值**

| 名称 | 含义 | 形状或类型 |
| --- | --- | --- |
| `timestep` | 每个样本对应的扩散时间步 | `(N,)` |
| `class_labels` | 每个样本的整数类别编号 | `(N,)`，整数张量 |
| `hidden_dtype` | 时间编码送入嵌入网络前采用的数据类型，用于匹配模型精度；默认 `None` 表示不显式转换类型 | 如 `torch.float16` |
| 返回值 `conditioning` | 融合时间步和类别信息的条件向量 | `(N, D)` |

**计算过程**

1. **编码时间步：`timesteps_proj = self.time_proj(timestep)`**

   使用固定的正弦/余弦编码，将 `(N,)` 的时间步转换为 `(N, 256)`。
   该实现使用 `float32` 计算并输出时间编码。

2. **映射时间步特征：`timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))`**

   先转换时间编码的数据类型，再经过两层线性层和 SiLU，将 `(N, 256)` 映射为 `(N, D)`。
   例如模型使用 `float16` 且调用方传入该类型时，会先将编码转为 `float16`，以匹配线性层权重精度。
   此处 `.to()` 只指定数据类型，不负责切换 CPU/GPU。

3. **查询类别嵌入：`class_labels = self.class_embedder(class_labels)`**

   根据类别编号查询嵌入表，得到 `(N, D)` 的类别向量。
   这里复用了变量名：赋值前 `class_labels` 是整数编号，赋值后是浮点嵌入向量。
   当前分析脚本已调用 `.eval()`，且这里没有传入强制丢弃标签的参数，因此不会随机丢弃类别标签。

4. **融合条件：`conditioning = timesteps_emb + class_labels`**

   将两个形状均为 `(N, D)` 的向量逐元素相加，结果仍为 `(N, D)`。
   这里没有进行向量拼接，因此输出维度不会变成 `2D`。

5. **返回结果：`return conditioning`**

   当前分析脚本中，返回值形状为 `(1, 1152)`。
   调用它的 `AdaLayerNormZero` 随后通过 SiLU 和线性层，将该向量映射为六组 `D` 维调制参数，
   用于控制 Attention 和前馈网络分支的缩放、偏移及残差门控。

**前向数据流**

```text
timestep (N,)
  → Timesteps：固定正弦/余弦编码 (N, 256)
  → 转换 dtype → TimestepEmbedding：可训练网络 (N, D) ─┐
                                                      ├─ 逐元素相加 → conditioning (N, D)
class_labels (N,) → LabelEmbedding：可训练查表 (N, D) ─┘
```
