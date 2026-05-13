# Qwen3-ASR Segment Aggregation Notes

## 结论

Qwen3-ASR 本身会输出带标点的 `result.text`，但 `Qwen3-ForcedAligner` 返回的是字/词级 `time_stamps`，中文场景下基本是字级，而且不带标点。

因此当前字幕策略不能只按 `time_stamps` 的字流硬切。现在的实现改成：

1. 以 Qwen3-ASR 的 `result.text` 作为带标点的权威文本。
2. 以 ForcedAligner 的 `time_stamps` 作为时间轴。
3. 把 `result.text` 里的标点投影回 timestamp 字流。
4. 聚合时优先在标点边界切分，只有没有合适标点时才用硬时长/字数上限兜底。

这个策略的目标是避免把“阿莫西林”“阿司匹林”“布洛芬”这类完整名词从中间切开。

## 代码位置

- 聚合入口：`src/services/qwen_asr_service.py`
- 主函数：`QwenAsrService._aggregate_subtitle_segments()`
- 标点投影：`QwenAsrService._project_punctuation_onto_timestamps()`
- 元数据：`QwenAsrService._aggregation_metadata()`
- 当前默认参数：
  - `QWEN_SUBTITLE_MAX_SECONDS=1.6`
  - `QWEN_SUBTITLE_MAX_CHARS=12`
  - `QWEN_SUBTITLE_GAP_SECONDS=0.35`

## 输入是什么

Qwen3-ASR 一次转写会返回两层信息：

```python
result.text
result.time_stamps
```

`result.text` 带标点，例如：

```text
五块，这不是特例哈。阿莫西林从六毛多一票降到两毛，阿司匹林最便宜只要三分钱一票，
```

`result.time_stamps` 带时间但不带标点，例如：

```python
ForcedAlignItem(text="阿", start_time=6.64, end_time=6.72)
ForcedAlignItem(text="莫", start_time=6.72, end_time=6.80)
ForcedAlignItem(text="西", start_time=6.80, end_time=6.96)
ForcedAlignItem(text="林", start_time=6.96, end_time=7.20)
```

在 Apple Podcast 90 秒样本里，Qwen3 原始 forced-align segment 是 `549` 条。直接拿它做字幕会太碎，所以需要聚合。

## 标点投影

`_project_punctuation_onto_timestamps()` 会按顺序对齐两条文本流：

- 左边：ForcedAligner timestamp text，去掉空白和标点后作为时间轴字符流。
- 右边：Qwen3-ASR `result.text`，保留标点，但对齐时跳过标点和空白。

当它在 `result.text` 中遇到标点时，会把标点挂到前一个 timestamp token 上。这样：

```text
哈 + 。 + 阿
```

会变成：

```python
{"text": "哈。", "start": 6.40, "end": 6.56}
{"text": "阿", "start": 6.64, "end": 6.72}
```

后续聚合看到当前字幕以 `。` 结尾，就会在“阿莫西林”之前切开，而不是在“阿莫/西林”之间硬切。

如果标点投影发现 `result.text` 和 timestamp 字流对不上，会打印 warning 并回退到原始 timestamp 文本，避免错误投影。

## 当前怎么合并

投影后，聚合器按时间排序，从左到右维护一个 `current_text`，不断尝试把下一个 timestamp piece 加进来。

切分优先级是：

1. 停顿超过 `QWEN_SUBTITLE_GAP_SECONDS`，并且当前字幕不是过短的半词，切。
2. 当前字幕以句末标点结束，比如 `。！？.!?`，切。
3. 当前字幕以短句标点结束，比如 `，；：、,;:`，且已有一定长度或时长，切。
4. 如果没有标点可用，但合并后超过硬上限，切。

硬上限不是直接使用 `1.6s / 12字`，而是用更宽的兜底：

- `hard_max_seconds = max(max_seconds * 2, max_seconds + 1.2)`
- `hard_max_chars = max(max_chars * 2, max_chars + 8)`

这样做的原因是：`1.6s / 12字` 适合作为目标字幕节奏，但不适合作为“强制切断中文词”的硬边界。现在会优先等标点，只有真的太长才硬切。

停顿切分也有短文本保护：如果当前字幕只有一两个字，且没有标点，普通短停顿不会立刻切断，避免把“似乎”切成“似 / 乎”。

## 例子

旧策略可能输出：

```srt
五块这不是特例哈阿莫
西林从六毛多一票降到
```

新策略会利用 `result.text` 里的标点，倾向输出：

```srt
五块，这不是特例哈。
阿莫西林从六毛多一票降到两毛，
阿司匹林最便宜只要三分钱一票，
```

这仍不是完整的中文分词器，但比纯字数/时长阈值更符合 Qwen3-ASR 的输出结构。

## 和 Whisper 的区别

Whisper 输出的 segment 本身就偏短句/短语级别，已经带有模型层面的语义分段。我们只是直接沿用 Whisper segment，所以看起来比较自然。

Qwen3-ASR 本地开源路径不同：ForcedAligner 提供字/词级时间戳，句级字幕需要服务层自己从 `result.text` 和 `time_stamps` 重建。

## 验收标准

自动检查继续保留：

- 所有核心用例 `processing_status=success`。
- Qwen 用例 `model_used` 以 `qwen3_asr:` 开头。
- `segments` 非空。
- 时间戳单调，无负时长；小于 `0.05s` 的 forced-align 浮点 overlap 可以容忍。
- Qwen segment 不应退化成字级字幕。

边界质量检查新增：

- `阿莫西林` 不被切成 `阿莫` / `西林`。
- `阿司匹林` 不被切开。
- `布洛芬` 不被切开。
- 标点应保留在字幕文本里。
- 字幕应优先在 `。！？；，` 这类标点后切分。

## 当前状态

当前实现已经从旧的“硬阈值贪心切分”升级为“标点投影 + 标点优先切分”。它仍然不是完整中文分词器，但已经利用了 Qwen3-ASR 自身的标点输出，能解决最明显的药品名被硬切问题。

下一步需要用 Apple Podcast 90s 和 300s 样本重新跑 GPU direct test，确认真实 SRT 中药品名和标点边界符合预期。
