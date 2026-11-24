# 3.3 & 3.4 私人第二大脑实战：用 `WorkBuddy + Obsidian` 搭建双链开卷笔记库

相比动辄要额外部署复杂数据库的重型方案，**`WorkBuddy + Obsidian`** 是最适合个人学习者与职场人的零代码黄金组合：
- 你的所有课件、读书笔记、实验记录全部是本地纯文本 `.md` 文件，永久掌握在自己手里；
- 挂载 Obsidian 官方开源的 **[`obsidianmd/obsidian-skills`](https://github.com/obsidianmd/obsidian-skills)** 后，WorkBuddy 不仅能精准检索本地笔记回答问题，还能在帮你沉淀新笔记时自动保留 `[[双向链接]]`、YAML 标签与高亮卡片！

---

## 3.3 三步搭建 `WorkBuddy + Obsidian` 私人开卷问答库

### Step 1：创建你的本地 Obsidian 知识库目录

1. 下载并打开免费的 [Obsidian](https://obsidian.md/)，新建一个独立仓库（Vault），例如 `D:\AI_Workspace\my-obsidian-vault`（或 `~/ai_workspace/my-obsidian-vault`）。
2. 在里面建立清晰的分类文件夹：`00-收件箱`、`01-课程讲义`、`02-论文精读`、`03-日常复盘`，放入几篇你自己的 Markdown 笔记（记得按 3.2 节经验用好 `##` 标题分节！）。

### Step 2：为 WorkBuddy 装上官方 `obsidian-skills`

在你的知识库目录终端中拉取 Obsidian 官方技能包到 `.codebuddy/skills/`（或 `.agents/skills/`）：

```bash
git clone https://github.com/obsidianmd/obsidian-skills.git /tmp/obsidian-skills
mkdir -p .codebuddy/skills
cp -r /tmp/obsidian-skills/skills/* .codebuddy/skills/
```

### Step 3：带出处角标与 `[[双向链接]]` 的开卷问答实战

在 WorkBuddy 中打开你的 `my-obsidian-vault` 工作区，输入以下融合了“第 6 关防幻觉锁”的开卷提问：

```text
请检索当前 Obsidian 笔记库中关于 ROCm 显存计算与模型精度的笔记，回答以下问题：
1. 为什么 4B Q4_K_M 模型跑起来占用的显存大于权重文件本身？
2. 回答中每一条关键结论末尾，必须用 [[笔记文件名#章节名]] 标注精确出处角标；
3. 将本次问答整理成一张新的复习卡片存入 `00-收件箱/显存复习卡.md`，自动打上 YAML 标签并关联原有笔记双链。
```

打开 Obsidian 的 **关系图谱（Graph View）**，你会惊喜地看到 AI 新生成的复习卡片已经自动与你原有的课程笔记连成了知识网络！

---

## 3.4 知识库防幻觉自测：10 道“开卷考题”验收法

怎么科学验证你的本地知识库真的靠谱（这也是第 7 章写进简历的核心评测亮点）？用你自己的笔记库出 **10 道标准自测题**：

- **6 道单篇细节题**：答案明确写在某篇笔记的某一段里（验收标准：回答正确且 `[[出处角标]]` 100% 指对文件）。
- **2 道跨篇综合题**：需要同时结合两篇不同笔记才能答全（验收标准：两边出处均被召回引用）。
- **2 道超纲陷阱题**：故意问一个笔记库里**根本没写过**的细节（验收标准：模型老老实实回答“当前笔记库未提及该信息”，**绝不脑补瞎编**！）。

---

## 🎯 第 3 章实战产出与进阶导流

- **本章产出**：「个人学习助手 v1.5（结构化提示词模板 + `WorkBuddy + Obsidian` 双链私人问答库 + 10 题防幻觉自测记录）」。
- 🔗 **课程引用与进阶导流**：当你的文档规模从几十篇笔记扩大到上万份复杂多栏 PDF、扫描件与企业级知识图谱时，欢迎深入学习本章联动的 Datawhale 专精课程 —— **[`all-in-rag`（大模型应用开发实战：RAG 技术全栈指南）](https://github.com/datawhalechina/all-in-rag)**，系统掌握多模态文档解析、向量+关键词混合检索、Rerank 重排与 GraphRAG！

> 💡 **新的想法诞生了**：现在你的 Obsidian 知识库已经在 WorkBuddy 里很好用了，但如果我在 `Qoder` 写代码、在 `Codex` 跑终端，或者**我想把自己的专业知识库做成一个标准服务给队友调用**，该怎么做？

👉 马上进入 **[第 4 章｜让助手长出手脚：把个人知识库制作成可共享的「知识管理 MCP」与自定义 Skills](../04-mcp-and-skills/index.md)**！
