# TAAC-KDD-2026-UNI-REC
tencent uni--rec challenge
algo0425是baseline
algo8236是score=0.8236的版本
revise 是在看了很多大佬的赛后分享后做的修正版，后面还会继续更新，只是比赛结束了，遗憾不能去验证这些修改是否有效。

`algo0425` 是一个比较完整的 HyFormer 序列建模方案； algo8236 是在它基础上改成了另一种“NS token 主导”的架构。

**核心差异**
| 模块 | `algo0425`  baseline | algo8236 |
|---|---|---|
| 序列交互方式 | 每个序列域 A/B/C/D 单独生成 Query，再分别 cross-attention 自己的序列 | 把所有序列拼成一个大 KV，让 user/item NS token 去统一 attend |
| Query 来源 | `MultiSeqQueryGenerator(ns_tokens + seq_pool)`，每个 domain 有自己的 Q | 没有独立 domain Q，直接用 user/item/dense NS tokens 做 query |
| 多序列融合 | 每层里：序列演化 -> domain Q attend domain seq -> Q+NS 用 RankMixer 融合 | 先各 domain 编码，再拼接全部序列，NS query 统一解码 |
| 最终输出 | 只用所有 domain 的 Q tokens concat 后分类 | 池化 user/item NS 输出，拼接后过 DCNv2 再分类 |
| 显式交叉 | 主要靠 HyFormer block / RankMixer | 加了 `CrossNetV2`，更偏 CTR 特征交叉 |
| 时间特征 | 只有 time bucket | algo8236 加了 cyclic time、UTC+8 等 |
| 语义规则 | 没有 | algo8236 加了 predict-time rule、history match/OOV 相关逻辑 |
| 正则 | dropout 默认较低，日志里有每轮 embedding reinit | algo8236 dropout 更高，OOV、reinit、full_train、后处理更多 |

**最关键的结构差异**

`algo0425` 的主路径大概是：

```text
user/item sparse + dense -> NS tokens

每个序列域 seq_a/b/c/d:
  seq embedding -> seq encoder
  生成 domain-specific Q
  Q attend 自己的序列

所有 Q + NS -> RankMixer 融合
最终只拿 Q tokens -> MLP 分类
```

algo8236 的主路径大概是：

```text
user/item sparse + dense -> NS tokens

每个序列域先编码
所有序列拼成一个大 KV
NS tokens 作为 query 统一 attend 全部序列

输出 user/item token 池化
user_out + item_out -> DCNv2 -> MLP 分类
```
**这对“语义做不进去”的影响**

问题可能就在这里：`algo0425` 的架构更尊重 domain 序列语义，每个 domain 有自己的 Q，最后 Q token 直接决定分类；而algo8236 把所有序列拼成一个大池子，再让 NS token 去 attend，最后又池化成 user/item 两个向量。这个结构很强，但也容易把加的显式语义信号“揉碎”。

所以algo8236 更像：

```text
强 ID + 强序列注意力 + DCN 交叉模型
```

它能拿到高分，但很“自我”。新加的规则、history match、dense pair 进去后，可能被 attention/pooling/DCN 稀释掉。

`algo0425` 更像：

```text
domain-aware sequence model
```

它天然保留 A/B/C/D 行为域差异，但缺少algo8236 后面那些 OOV、full_train、predict-time rerank、DCNv2 之类提分手段。

**结论**

algo8236 不是架构差，它线上比原始 baseline 强很多；但它确实不适合“后期硬塞语义特征”。它的主干会把新增语义当成普通输入，不会给它单独决策权。

如果重做，我会选一个折中架构：

```text
保留 algo0425 的 domain-specific Q 机制
加上algo8236 的 OOV / full_train / reinit / UTC+8
再单独加 Rule Expert / History Match Expert
最后用 gate 融合 backbone score 和 expert score
```

这比继续往algo8236 的 pooled NS + DCN 里塞语义，成功率会高很多。


**revise版本的修改**
已在 `algo8236` 完成这版升级，走的是“先把主结构做对，DIN/semi-local 留开关”的方案。

改动概要：

- [model.py]
  - 新增 `FieldAwareNSTokenizer`：按 fid 整字段分桶，不再像 RankMixer 那样按维度切碎字段 embedding。
  - 新增 `--use_abs_time_ns`：样本级绝对时间周期特征注入 NS token。
  - 新增 `--use_session_crossday_time`：序列侧注入 `session_bucket` 和 `cross_day_bucket`。
  - `DenseIntPairProjector` 改为先做 `sign(x) * log1p(abs(x))`，再做 pair gating。
  - 新增可开关轻 DIN 分支：`--use_light_din_branch`。
  - 新增可开关 semi-local causal mask：`--seq_semilocal_causal_mask --seq_semilocal_window 128`。
  - EMA 局部同步 reinit embedding。 `reinit_high_cardinality_params()` 现在返回两份信息：
      - `reinit_ptrs`：继续给 Adagrad 用，恢复低基数 optimizer state。
      - `reinit_names`：被重置的 embedding 参数名，给 EMA 精准同步用。

- [dataset.py]
  - 每个序列域新增输出：
    - `{domain}_session_bucket`
    - `{domain}_cross_day_bucket`
  - session 按相邻行为超过 30 分钟切分。
  - cross-day 按当前样本时间与历史行为时间的天级差分桶。

- [train.py])、[trainer.py]、[infer.py]
  - [trainer.py]：新增 `_sync_ema_reinitialized_params()`，只把这些高基数 embedding 从主模型复制到 `ema_model`，不重置其他 EMA 权重。
  - [trainer.py]：每轮 reinit 后立即调用 EMA 局部同步。

  - 新参数已接入训练和推理。
  - DIN 支路支持可选辅助 loss：`--din_branch_loss_weight`。
  - 推理会从 `train_config.json` 自动恢复这些结构参数。

- [run.sh]
  - 当前激活配置已改成保守版：

```bash
--ns_tokenizer_type fieldaware
--user_ns_tokens 3
--item_ns_tokens 4
--loss_type focal
--focal_alpha 0.75
--focal_gamma 1.5
--label_smoothing 0
--use_dense_int_pair
--use_abs_time_ns
--use_session_crossday_time
```

DIN 和 semi-local 默认没打开。跑 DIN 可追加：

```bash
--use_light_din_branch --din_branch_loss_weight 0.1
```

跑 semi-local 可追加：

```bash
--seq_semilocal_causal_mask --seq_semilocal_window 128
```

