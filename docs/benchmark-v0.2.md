# Benchmark v0.2：人工相关性标注与检索消融

v0.2 是一个版本化的检索评测格式。`examples/bench/recsys10.template.jsonl`
只提供十个候选问题；其中 `gold_sources` 均为空。模板中的 `TODO: human
annotator` 表明尚未完成标注，不能把它当作可报告相关性指标的 benchmark。

## Schema

每行是一个 JSON 对象，包含：

- `id`：稳定且唯一的 case 标识；
- `question`、`category`；
- `gold_sources`：人工确认的来源列表；每项必须有 `title`、`url`、等级为
  1/2/3 的 `relevance_grade`，以及解释判断的 `annotation_note`；
- 可选的 `expected_entities`；
- `annotation_version`、`annotated_by`、`annotated_at`：标注批次、标注者和
  ISO 8601 时间。

等级含义应在一个标注版本内固定：3 表示直接且充分回答问题，2 表示提供
重要的部分答案，1 表示相关但只提供辅助信息。没有人工确认来源时，必须保留
空数组；评测器会将 Recall@3、Recall@5、MRR 和 nDCG@5 输出为 `null`。

## 人工标注流程

1. 从模板复制出新的、带版本号的数据文件，不覆盖历史版本。
2. 冻结问题文本和标注规范，再由人工检索并阅读来源正文。搜索返回仅是候选集，
   不能由程序自动升级为 gold。
3. 标注者逐项确认标题和最终 URL，赋 1/2/3 等级，并在
   `annotation_note` 中记录该来源为何相关。不要只依据标题或 snippet 判断。
4. 第二位人工复核有分歧或高等级来源；按团队预先约定的仲裁方式处理分歧。
5. 完成后更新 `annotation_version`、真实 `annotated_by` 和 `annotated_at`，运行
   schema 校验与离线测试。新增或修改 gold 必须产生新版本。

## 防止数据泄漏

- gold 文件只用于评测，不得进入查询生成、提示词、召回、重排或答案生成上下文。
- 开发阶段使用无标签模板或独立 train/dev case；冻结测试集后再运行最终比较。
- 不从被评系统的搜索结果自动建立 gold，也不根据单个消融的失败结果补标签。
- 保存问题版本、标注版本和消融名称，使结果可追溯；若问题或 gold 改变，不与旧
  版本分数直接混合。
- 报告未标注 case 的运行质量时只使用操作性指标，不能用 0 替代未知相关性。

## 指标与消融

相关性指标包括 Recall@3、Recall@5、MRR 和采用 `2^grade - 1` 增益的
nDCG@5。操作性指标包括来源域名多样性、规范化 URL 重复率、引用覆盖率、
case 延迟和 provider 调用数。均值逐指标忽略 `null`，失败 case 不参与均值但会
保留在产物中。

支持的消融名称为：`snippet`、`bm25_only`、`dense_only`、`hybrid_rrf`、
`hybrid_rerank`、`hybrid_rerank_mmr`。每个名称都映射到明确的 BM25、dense、
RRF、rerank、MMR 阶段开关。当前默认 runner 只允许 `mock` 模式，因而默认测试
不需要 API key，也不会访问网络。真实 provider 的消融执行需要后续单独、显式的
网络入口；不能通过默认测试隐式开启。

## 输出

每次运行写出：

- `cases/<case-id>.json`：单 case 结果或隔离后的失败原因；
- `summary.json`：整体的 null-aware 均值和 case 列表；
- `per_category.json`：按 category 聚合的结果；
- `comparison.md`：以 Markdown 表格列出 Recall@5、MRR、nDCG@5、延迟和
  API 调用数。一次运行生成当前消融的一行；多个消融汇总可传给
`comparison_markdown` 生成多行比较表。

## Real 模式、超时与断点续跑

真实 benchmark 应显式区分单次 provider 请求、单任务、完整 case 和报告生成
的期限。例如：

```bash
uv run rec-researcher benchmark examples/bench/recsys10.v0.2.jsonl \
  --mode real --retrieval-mode hybrid --vector-store memory \
  --request-timeout 120 --task-timeout 180 --case-timeout 600 \
  --report-timeout 120 --max-retries 1 \
  --max-concurrency 1 --retrieval-concurrency 1 --fetch-concurrency 1
```

case 超时一定产生失败状态，并直接生成确定性 fallback 报告，不再调用外部 LLM。
所有 case 完成后会关闭 LLM、搜索、embedding、reranker、fetcher 和根向量索引
资源。只要仍有失败，CLI 就以非零状态退出，但已完成的产物仍然保留。

首次用修复后的版本完成运行后，可在相同命令末尾添加 `--resume`。runner 会根据
benchmark 内容和不含密钥的执行配置生成指纹，只跳过指纹一致的成功 case，并
重跑失败或缺失 case。旧版产物没有配置指纹，或模型、超时、检索参数发生变化时，
会拒绝复用，防止把不可比较的结果合并到同一份 summary。
