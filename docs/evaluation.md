# 评测说明

RecResearcher 的轻量评测同时观察工作流可靠性、报告结构和检索相关性。运行质量指标
可以从系统产物确定性计算；Recall@K 和 MRR 属于相关性指标，必须依赖人工提供的
`gold_source_ids`。二者不能混用。

## Case 与汇总

Benchmark 输入是 JSONL，每一行至少包含稳定 `id` 和非空 `question`：

```json
{"id":"example","question":"研究问题","gold_source_ids":["source-a"]}
```

每个 case 独立执行并写入 `cases/<case-id>.json`。异常或失败运行保存
`failure_reason`，不会取消其他 case。`summary.json` 保存 case 数量、成功/失败数量、
每个 case 的完整结果及成功 case 的算术均值。

## 指标定义

### task_success_rate

成功完成的 `TaskResult` 数除以任务总数：

```text
completed_tasks / all_tasks
```

没有任务时返回 `0.0`。该指标反映编排执行情况，不衡量回答正确性。

### citation_coverage

直接使用本地 `CitationVerifier` 的覆盖率：有足够长度的事实性 prose paragraph 中，
包含合法引用标记的段落占比。标题和列表不会被当作 prose paragraph。该指标衡量报告
引用覆盖结构，不证明来源本身可靠，也不证明论断蕴含关系成立。

### valid_url_rate

具有 `http` 或 `https` scheme 且包含 host 的来源 URL 数除以来源总数。没有来源时
返回 `0.0`。`SourceRecord` 已使用 Pydantic `HttpUrl` 做输入校验，此指标仍保留为运行
产物的显式健康检查。

### source_diversity

唯一来源域名数除以来源总数：

```text
unique URL hosts / all sources
```

没有来源时返回 `0.0`。它只能观察域名层面的分散程度，不代表编辑独立性、观点多样性
或来源质量。

### report_section_completeness

报告中实际出现的必需二级标题数除以必需标题总数。当前必需章节为：

- 论文与代码对照
- 数据集与指标
- 复现难度分析
- 三天复现建议

空的自定义必需章节集合定义为 `1.0`。章节存在不等于章节内容正确或充分。

### average_latency

成功 case 的 `latency_seconds` 算术平均值。没有成功 case 时返回 `0.0`。这是当前执行
环境中的墙钟时间，不适合跨机器直接比较。

### provider_failure_rate

Runner 以未完成的研究任务作为 provider/task 失败尝试，计算：

```text
failed_or_non-completed_tasks / all_task_attempts
```

没有任务尝试时返回 `0.0`。计数为负数或失败数大于总数会抛出清晰的 `ValueError`。
当前粒度不能区分搜索、LLM、Embedding 和 Reranker 的独立失败率，这是后续可观测性
建设的一部分。

### Recall@K

只有 case 提供非空人工 `gold_source_ids` 时才计算：

```text
|top-K retrieved source IDs ∩ gold source IDs| / |gold source IDs|
```

`gold_source_ids` 缺失或为空时返回 JSON `null`。系统绝不会用“返回了任意 URL”、
“URL 格式有效”或“来源数量大于零”冒充 Recall@K。

### MRR

只有 case 提供非空人工 `gold_source_ids` 时，才根据第一个相关来源的排名计算：

```text
MRR(case) = 1 / rank(first relevant source)
```

有 gold 但没有召回相关来源时为 `0.0`；没有 gold relevance 时为 JSON `null`。
汇总 Recall@K/MRR 仅对有 gold 且成功执行的 case 求均值。如果整个 benchmark 都没有
gold，这两个汇总值也为 `null`。

## 确定性边界

Mock provider 使用固定虚构语料和确定性逻辑，Mock benchmark 适合发现代码回归、
并发隔离失败、Schema 变化、引用结构变化和指标计算错误。它不代表真实网页研究质量，
也不能证明系统对真实论文的覆盖率或结论准确率。

实时网页搜索不具有完全确定性。即使查询文本相同，搜索索引更新、网页删除或修改、
地域差异、服务端排序、限流、网络故障以及模型版本变化都可能改变来源和报告。因此：

- Real benchmark 不应被描述为逐字节可复现；
- 应保存运行时间、provider、来源 ID/URL、失败原因和配置版本；
- 跨时间比较时应使用足够多的人工标注 case，并报告方差或置信区间；
- 没有人工 relevance judgment 时，只能报告运行质量指标，不能报告检索 Recall/MRR。

默认测试通过 Pytest marker 排除 `network` 测试，因而不需要互联网或 API Key。真实
网络测试必须显式选择，例如：

```bash
uv run pytest -m network
```
