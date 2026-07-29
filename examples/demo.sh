#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uv run rec-researcher doctor
uv run ruff check .
uv run pytest
uv run rec-researcher run "生成式推荐与传统双塔有什么区别？" --mode mock

latest_report="$(ls -t outputs/*/report.md | sed -n '1p')"
echo "Latest report: $latest_report"
