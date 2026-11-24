"""第 4 章实战源码：个人知识库 MCP 服务 (Knowledge Management MCP Server)

功能：将第 3 章搭建的 WorkBuddy + Obsidian 本地 Markdown 笔记库封装成标准 MCP 服务，
既可在自己的 WorkBuddy、Qoder、Codex 中跨软件调用，也可打包分享给队友调用。

暴露的三个标准 MCP 工具：
1. search_notes(query, top_k=5): 按关键词检索笔记标题与正文，返回出处路径与匹配摘要
2. create_note_card(title, content, tags=None, folder="00-收件箱"): 安全创建带 YAML 头与双链的笔记卡片
3. list_notes_by_tag(tag): 盘点包含指定标签（YAML tags 或正文 #标签）的全部笔记
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Any


class KnowledgeVault:
    """带沙盒目录边界校验的本地 Obsidian / Markdown 知识库管理器"""

    def __init__(self, vault_dir: str | Path) -> None:
        self.vault_root = Path(vault_dir).expanduser().resolve()
        self.vault_root.mkdir(parents=True, exist_ok=True)

    def _safe_resolve(self, relative_path: str | Path) -> Path:
        candidate = (self.vault_root / relative_path).resolve()
        if not candidate.is_relative_to(self.vault_root):
            raise ValueError(f"安全拦截：禁止访问知识库沙盒目录之外的路径 -> {relative_path}")
        return candidate

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        tags: set[str] = set()
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
        if fm_match:
            fm_block = fm_match.group(1)
            inline = re.search(r"tags:\s*\[([^\]]*)\]", fm_block)
            if inline:
                for item in inline.group(1).split(","):
                    cleaned = item.strip().strip("'\"")
                    if cleaned:
                        tags.add(cleaned)
            for bullet in re.findall(r"^\s*-\s*([^\n#]+)", fm_block, flags=re.M):
                cleaned = bullet.strip().strip("'\"")
                if cleaned:
                    tags.add(cleaned)
        for hashtag in re.findall(r"(?<!\w)#([\w\u4e00-\u9fff-]+)", text):
            tags.add(hashtag)
        return sorted(tags)

    def search_notes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """在个人知识库中检索与 query 最相关的 Markdown 笔记及出处片段"""
        keywords = [kw.lower() for kw in query.strip().split() if kw.strip()]
        if not keywords:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for md_path in sorted(self.vault_root.rglob("*.md")):
            text = md_path.read_text(encoding="utf-8", errors="ignore")
            haystack = f"{md_path.stem}\n{text}".lower()
            hits = sum(haystack.count(kw) for kw in keywords)
            if hits <= 0:
                continue
            # 截取第一处命中上下文作为带出处预览
            first_pos = min((haystack.find(kw) for kw in keywords if kw in haystack), default=0)
            snippet_start = max(0, first_pos - 40)
            snippet = re.sub(r"\s+", " ", text[snippet_start : snippet_start + 160]).strip()
            scored.append(
                (
                    hits,
                    {
                        "title": md_path.stem,
                        "source": str(md_path.relative_to(self.vault_root)),
                        "tags": self._extract_tags(text),
                        "snippet": snippet,
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[: max(1, top_k)]]

    def create_note_card(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        folder: str = "00-收件箱",
    ) -> str:
        """向知识库写入一张带 YAML 元数据与创建日期的 Markdown 知识卡片"""
        clean_title = re.sub(r"[\\/:*?\"<>|]", "-", title.strip())
        if not clean_title:
            raise ValueError("笔记标题不能为空")
        if not content.strip():
            raise ValueError("笔记内容不能为空")

        target_dir = self._safe_resolve(folder)
        target_dir.mkdir(parents=True, exist_ok=True)
        note_path = self._safe_resolve(Path(folder) / f"{clean_title}.md")

        tag_list = ", ".join(tags or ["AI沉淀"])
        frontmatter = (
            f"---\n"
            f"title: {clean_title}\n"
            f"created: {date.today().isoformat()}\n"
            f"tags: [{tag_list}]\n"
            f"---\n\n"
        )
        note_path.write_text(f"{frontmatter}# {clean_title}\n\n{content.strip()}\n", encoding="utf-8")
        return f"已保存笔记卡片：{note_path.relative_to(self.vault_root)}"

    def list_notes_by_tag(self, tag: str) -> list[dict[str, str]]:
        """按标签盘点知识库中所有匹配的笔记"""
        target = tag.strip().lstrip("#").lower()
        if not target:
            return []
        matched: list[dict[str, str]] = []
        for md_path in sorted(self.vault_root.rglob("*.md")):
            text = md_path.read_text(encoding="utf-8", errors="ignore")
            tags_lower = {t.lower() for t in self._extract_tags(text)}
            if target in tags_lower:
                matched.append(
                    {
                        "title": md_path.stem,
                        "source": str(md_path.relative_to(self.vault_root)),
                    }
                )
        return matched


def build_mcp_server(vault_dir: str | Path | None = None):
    """构建标准 FastMCP Server 实例，供 stdio / SSE 启动"""
    from mcp.server.fastmcp import FastMCP

    root = vault_dir or os.environ.get("OBSIDIAN_VAULT_PATH", "./my-obsidian-vault")
    vault = KnowledgeVault(root)
    mcp = FastMCP("hello-rocm-knowledge-mcp")

    @mcp.tool()
    def search_notes(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索个人 Obsidian 知识库中的笔记卡片并返回出处路径与摘要"""
        return vault.search_notes(query=query, top_k=top_k)

    @mcp.tool()
    def create_note_card(
        title: str,
        content: str,
        tags: list[str] | None = None,
        folder: str = "00-收件箱",
    ) -> str:
        """在个人知识库指定分类文件夹下新建一张支持 [[双向链接]] 的 Markdown 笔记卡片"""
        return vault.create_note_card(title=title, content=content, tags=tags, folder=folder)

    @mcp.tool()
    def list_notes_by_tag(tag: str) -> list[dict[str, str]]:
        """按标签名称盘点知识库中所有关联的笔记清单"""
        return vault.list_notes_by_tag(tag=tag)

    return mcp


if __name__ == "__main__":
    server = build_mcp_server()
    server.run()
