"""hello-rocm 2.0 自动化质量门控与单元测试套件 (Test-First)

覆盖范围：
1. 第 4 章「知识管理 MCP Server」核心逻辑（正常流程、边界条件、目录穿越防御异常情况）
2. 第 5 章「中文通知 JSON 提取与人设破功词评测脚本」核心评分逻辑
3. 第 6 章「AMD Quark 单模型量化前后对比测速报告生成」核心计算与边界保护
4. 第 2 章「已验证客户端配置库 (src/configs/)」JSON/TOML 语法与关键避坑字段校验
5. 全站 VitePress config.mts 导航路由与新增章节 Markdown 相对链接/图片零死链校验
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestKnowledgeMCPServer(unittest.TestCase):
    """测试第 4 章个人知识库 MCP 服务（正常、边界、异常安全边界）"""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.vault_path = Path(self.tmp_dir.name)
        from src.main_assistant_pkg import load_knowledge_mcp

        self.km = load_knowledge_mcp(self.vault_path)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_create_and_search_note_normal(self) -> None:
        msg = self.km.create_note_card(
            title="ROCm显存三本账",
            content="4B Q4_K_M 权重约 2.6GB，加上 [[KV Cache]] 建议预留 4.5GB 显存。",
            tags=["rocm", "入门笔记"],
            folder="01-学习笔记",
        )
        self.assertIn("ROCm显存三本账.md", msg)

        results = self.km.search_notes("KV Cache", top_k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "ROCm显存三本账")
        self.assertIn("rocm", results[0]["tags"])

    def test_list_notes_by_tag_and_empty_boundary(self) -> None:
        self.km.create_note_card(
            title="WorkBuddy避坑",
            content="models.json 必须开启 32768 上下文。 #workbuddy",
            tags=["办公"],
        )
        # 既支持 YAML frontmatter 标签，也支持正文 #标签
        self.assertEqual(len(self.km.list_notes_by_tag("办公")), 1)
        self.assertEqual(len(self.km.list_notes_by_tag("workbuddy")), 1)
        # 边界条件：空查询或不存在标签返回空列表而非报错
        self.assertEqual(self.km.search_notes("   "), [])
        self.assertEqual(self.km.list_notes_by_tag("不存在的标签"), [])

    def test_path_traversal_and_invalid_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.km.create_note_card(
                title="hack",
                content="bad",
                folder="../../etc",
            )
        with self.assertRaises(ValueError):
            self.km.create_note_card(title="   ", content="空标题应拒绝")


class TestChineseFineTuneEval(unittest.TestCase):
    """测试第 5 章自建中文测试集微调前后客观评估逻辑"""

    def test_json_extraction_and_persona_metrics(self) -> None:
        from src.main_assistant_pkg import load_eval_zh

        ev = load_eval_zh()
        # 支持从带 Markdown 代码块的回答中提取 JSON 并核对字段
        raw_pred = '好的，提取结果如下：\n```json\n{"事项": "组会汇报", "截止时间": "周五14:00", "负责人": "张三"}\n```'
        gold = {"事项": "组会汇报", "截止时间": "周五14:00", "负责人": "张三"}
        score = ev.evaluate_json_sample(raw_pred, gold)
        self.assertTrue(score["json_valid"])
        self.assertAlmostEqual(score["field_accuracy"], 1.0)

        # 异常情况：JSON 破损时优雅降级返回 0 分而非崩溃
        broken = ev.evaluate_json_sample('{"事项": "漏了右括号"', gold)
        self.assertFalse(broken["json_valid"])
        self.assertEqual(broken["field_accuracy"], 0.0)

        # 网感人设破功词检测（检测“作为一个人工智能”等客服腔）
        self.assertTrue(ev.has_ai_cliche("作为一个人工智能语言模型，我无法体会困倦。"))
        self.assertFalse(ev.has_ai_cliche("宝子们谁懂啊！这杯冰美式直接救我狗命！"))


class TestQuarkBeforeAfterBench(unittest.TestCase):
    """测试第 6 章 AMD Quark 量化前后体积/显存/速度对比报告生成"""

    def test_comparison_report_and_zero_guard(self) -> None:
        from src.main_assistant_pkg import load_quark_bench

        qb = load_quark_bench()
        summary = qb.compute_comparison(
            before={"name": "Qwen3-4B (BF16)", "size_gb": 8.0, "vram_gb": 9.6, "tps": 38.0, "ttft_ms": 420.0},
            after={"name": "Qwen3-4B (Quark INT4-AWQ)", "size_gb": 2.6, "vram_gb": 3.8, "tps": 89.3, "ttft_ms": 190.0},
        )
        self.assertGreater(summary["size_saving_pct"], 65.0)
        self.assertGreater(summary["tps_speedup"], 2.0)
        md_table = qb.format_markdown_report(summary)
        self.assertIn("Qwen3-4B (Quark INT4-AWQ)", md_table)
        self.assertIn("tok/s", md_table)

        # 边界异常保护：分母为 0 时抛出清晰 ValueError
        with self.assertRaises(ValueError):
            qb.compute_comparison(
                before={"name": "bad", "size_gb": 0.0, "vram_gb": 1.0, "tps": 10.0, "ttft_ms": 100.0},
                after={"name": "after", "size_gb": 1.0, "vram_gb": 1.0, "tps": 10.0, "ttft_ms": 100.0},
            )


class TestVerifiedConfigsAndDocsLinks(unittest.TestCase):
    """测试第 2 章配置文件契约与全站 0~9 章路由/相对链接完整性"""

    def test_configs_valid_and_safe(self) -> None:
        wb_path = REPO_ROOT / "src/configs/workbuddy-models.json"
        qoder_path = REPO_ROOT / "src/configs/qoder-openai-compatible.json"
        codex_path = REPO_ROOT / "src/configs/codex-config.toml"

        self.assertTrue(wb_path.exists(), "缺少 workbuddy-models.json")
        wb = json.loads(wb_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(wb.get("models", [])), 3)
        for m in wb["models"]:
            self.assertTrue(m["url"].endswith("/v1/chat/completions"), f"WorkBuddy models.json url 必须以 /v1/chat/completions 结尾: {m['url']}")
            self.assertTrue(bool(m.get("apiKey", "").strip()), "WorkBuddy apiKey 不可为空字符串")
            self.assertGreaterEqual(m.get("maxContextTokens", 0), 32768, "WorkBuddy 上下文必须 >= 32768 防工具说明截断")
            self.assertTrue(m.get("supportsToolCall", False))

        self.assertTrue(qoder_path.exists(), "缺少 qoder-openai-compatible.json")
        qoder = json.loads(qoder_path.read_text(encoding="utf-8"))
        self.assertIn("providers", qoder)

        self.assertTrue(codex_path.exists(), "缺少 codex-config.toml")
        codex = tomllib.loads(codex_path.read_text(encoding="utf-8"))
        self.assertIn("model_providers", codex)

    def test_vitepress_zh_routes_and_new_markdown_links(self) -> None:
        config_text = (REPO_ROOT / "docs/.vitepress/config.mts").read_text(encoding="utf-8")
        # 检查 config.mts 中所有 /zh/... 路由在 docs/ 下均真实存在
        zh_links = set(re.findall(r"link:\s*'(/zh/[^']*)'", config_text))
        self.assertGreater(len(zh_links), 10)
        for route in sorted(zh_links):
            clean = route.split("#")[0]
            if clean.endswith("/"):
                candidate = REPO_ROOT / "docs" / clean.lstrip("/") / "index.md"
            else:
                candidate = REPO_ROOT / "docs" / f"{clean.lstrip('/')}.md"
                if not candidate.exists():
                    candidate = REPO_ROOT / "docs" / clean.lstrip("/") / "index.md"
            self.assertTrue(candidate.exists(), f"config.mts 存在失效中文路由: {route} -> 找不到 {candidate}")

        # 检查新增 00~09 章节内所有 Markdown 相对链接与图片真实存在
        chapter_dirs = [
            "00-guide",
            "01-first-model",
            "02-connect-tools",
            "03-prompt-and-rag",
            "04-mcp-and-skills",
            "05-fine-tune",
            "06-optimize-speed",
            "07-portfolio-resume",
            "08-fun-cases",
            "09-amd-ecosystem",
        ]
        md_link_re = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
        for ch in chapter_dirs:
            ch_dir = REPO_ROOT / "docs/zh" / ch
            self.assertTrue(ch_dir.exists(), f"章节目录尚未创建: {ch_dir}")
            for md_file in ch_dir.rglob("*.md"):
                # 忽略代码块中的示例链接
                content_no_fence = re.sub(r"```.*?```", "", md_file.read_text(encoding="utf-8"), flags=re.S)
                for raw_target in md_link_re.findall(content_no_fence):
                    target = raw_target.strip().split()[0].split("#")[0]
                    if not target or target.startswith(("http://", "https://", "mailto:")):
                        continue
                    if target.startswith("/zh/"):
                        # VitePress 站内根路径
                        p1 = REPO_ROOT / "docs" / target.lstrip("/") / "index.md"
                        p2 = REPO_ROOT / "docs" / f"{target.lstrip('/')}.md"
                        self.assertTrue(p1.exists() or p2.exists(), f"{md_file} 含失效站内链接: {raw_target}")
                    elif target.startswith("/"):
                        p_pub = REPO_ROOT / "docs/public" / target.lstrip("/")
                        self.assertTrue(p_pub.exists(), f"{md_file} 含失效静态资源路径: {raw_target}")
                    else:
                        resolved = (md_file.parent / target).resolve()
                        self.assertTrue(
                            resolved.is_relative_to((REPO_ROOT / "docs").resolve()),
                            f"VitePress 禁止相对链接越出 docs/ 目录: {md_file} -> {raw_target}",
                        )
                        if not resolved.exists() and not resolved.suffix:
                            resolved_md = Path(f"{resolved}.md")
                            resolved_idx = resolved / "index.md"
                            self.assertTrue(
                                resolved_md.exists() or resolved_idx.exists(),
                                f"{md_file} 含失效相对链接: {raw_target}",
                            )
                        else:
                            self.assertTrue(resolved.exists(), f"{md_file} 含失效相对路径: {raw_target} -> {resolved}")

        # 同时校验根目录 README.md 所有本地相对链接真实存在
        readme_no_fence = re.sub(r"```.*?```", "", (REPO_ROOT / "README.md").read_text(encoding="utf-8"), flags=re.S)
        for raw_target in md_link_re.findall(readme_no_fence):
            target = raw_target.strip().split()[0].split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (REPO_ROOT / target).resolve()
            self.assertTrue(resolved.exists(), f"README.md 含失效相对链接: {raw_target} -> {resolved}")


if __name__ == "__main__":
    unittest.main()
