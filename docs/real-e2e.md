# 真实网络端到端测试

完整 hybrid 测试会依次调用 Tavily Search、网页抓取、分块、BM25、
SiliconFlow Embedding、临时 Milvus Lite、RRF、SiliconFlow Reranker、MMR，
最后生成 Evidence、Report 并验证引用。该测试会产生真实、可能计费的 API 调用，
不会被默认的 `uv run pytest` 执行。

## 配置

在运行测试的 shell 中通过环境变量配置以下值；不要把密钥提交到版本库：

```bash
export REC_LLM_BASE_URL="https://your-openai-compatible-service.example/v1"
export REC_LLM_API_KEY="..."
export REC_LLM_MODEL="your-chat-model"
export REC_TAVILY_API_KEY="..."
export REC_SILICONFLOW_API_KEY="..."
export REC_EMBEDDING_MODEL="your-siliconflow-embedding-model"
export REC_RERANKER_MODEL="your-siliconflow-reranker-model"
```

Tavily 与 SiliconFlow 的 base URL 已有官方端点默认值，需要代理或兼容端点时可另设
`REC_TAVILY_BASE_URL` 和 `REC_SILICONFLOW_BASE_URL`。

## 运行

仅运行这个真实 E2E 测试：

```bash
uv run pytest -q -m network_e2e tests/integration/test_hybrid_network_e2e.py
```

缺少任一 API Key、LLM 端点或模型配置时，测试显示为 skipped，不会失败。测试限制为
3 个 planner task、每个查询 2 个来源、总计最多 5 个页面、并发 2，并限制 LLM
输出 token。每个 task 最终最多保留 5 个 passage。

## 代理与超时排查

`httpx` 默认读取 `HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY`。若连接失败，先确认
代理能访问 Tavily、LLM 服务、SiliconFlow 以及搜索结果页面，并检查企业代理的 CA
证书配置。超时时可先单独验证各端点连通性，再按需提高测试里的
`request_timeout_seconds` 或 orchestrator 总超时；429 通常表示限流或额度不足。

## 临时结果清理

pytest 的 `tmp_path` 同时存放 Milvus Lite 数据库和五类运行产物；正常情况下 pytest
会管理这些临时目录。需要手动清理时，可先用 `pytest --basetemp=/tmp/rec-e2e`
指定一个独立目录，测试结束且没有进程占用数据库后删除 `/tmp/rec-e2e`。测试不会写入
项目的 `data/` 或 `outputs/`。
