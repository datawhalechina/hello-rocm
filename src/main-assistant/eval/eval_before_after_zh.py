"""第 5 章实战源码：自建中文测试集微调前后效果客观评估脚本

用途：
1. 评估主线任务（中文群通知提取结构化 JSON）：统计 JSON 合法率与关键字段精确匹配率
2. 评估选修任务（WeClone 微信分身 / 小红书种草体）：统计机械“AI 客服腔”破功词出现率
支持无缝请求本地三大标准引擎：llama-server (8080) / lemonade-server (13305) / vLLM (8000)
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from typing import Any

AI_CLICHE_KEYWORDS = (
    "作为一个人工智能",
    "作为AI语言模型",
    "作为一名AI",
    "我没有个人情感",
    "我无法体会",
    "希望以上建议对您有所帮助",
    "总而言之，综上所述",
)


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """从模型输出（纯文本或 ```json 代码块）中稳健解析 JSON 字典"""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.S)
    candidate = fenced.group(1) if fenced else raw_text.strip()
    if not candidate.startswith("{"):
        brace_match = re.search(r"\{.*\}", candidate, flags=re.S)
        if not brace_match:
            return None
        candidate = brace_match.group(0)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def evaluate_json_sample(pred_text: str, gold: dict[str, Any]) -> dict[str, Any]:
    """评估单条中文通知抽取样本的 JSON 合法性与字段准确率"""
    parsed = extract_json_object(pred_text)
    if parsed is None:
        return {"json_valid": False, "field_accuracy": 0.0}
    if not gold:
        return {"json_valid": True, "field_accuracy": 1.0}
    matched = sum(1 for k, v in gold.items() if str(parsed.get(k, "")).strip() == str(v).strip())
    return {
        "json_valid": True,
        "field_accuracy": matched / len(gold),
    }


def has_ai_cliche(reply_text: str) -> bool:
    """检测回复中是否出现典型的机械 AI 客服腔破功词"""
    normalized = reply_text.replace(" ", "")
    return any(kw.replace(" ", "") in normalized for kw in AI_CLICHE_KEYWORDS)


def call_local_openai(base_url: str, model: str, prompt: str, api_key: str = "rocm-local") -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


BUILTIN_DEMO_TESTSET = [
    {
        "prompt": "请从以下群通知提取 JSON（包含字段：事项、截止时间、负责人）：\n【课程群通知】请各组同学注意，期末AI作品集PPT务必要在本周五18:00前发给学委李明汇总！",
        "gold": {"事项": "提交期末AI作品集PPT", "截止时间": "本周五18:00", "负责人": "李明"},
    },
    {
        "prompt": "请从以下群通知提取 JSON（包含字段：事项、截止时间、负责人）：\n@全体成员 3月差旅报销单请于周三中午12:00前交到行政王芳处，逾期顺延下月。",
        "gold": {"事项": "提交3月差旅报销单", "截止时间": "周三中午12:00", "负责人": "王芳"},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="中文微调前后效果评估脚本")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="本地服务 /v1 地址")
    parser.add_argument("--model", default="qwen3.5-4b-q4_k_m", help="模型名称")
    parser.add_argument("--testset", default="", help="可选：自定义 JSONL 测试集路径")
    args = parser.parse_args()

    samples = BUILTIN_DEMO_TESTSET
    if args.testset:
        with open(args.testset, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]

    valid_cnt, acc_sum = 0, 0.0
    for item in samples:
        reply = call_local_openai(args.base_url, args.model, item["prompt"])
        res = evaluate_json_sample(reply, item["gold"])
        valid_cnt += int(res["json_valid"])
        acc_sum += res["field_accuracy"]

    total = max(1, len(samples))
    print(f"评测样本数: {len(samples)}")
    print(f"JSON 格式合法率: {valid_cnt / total * 100:.1f}%")
    print(f"关键字段准确率: {acc_sum / total * 100:.1f}%")


if __name__ == "__main__":
    main()
