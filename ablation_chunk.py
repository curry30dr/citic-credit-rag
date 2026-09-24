# -*- coding: utf-8 -*-
"""chunk-size 消融实验：测不同切块大小对 Recall@4 的影响"""
import json, os, re, math
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi

BASE = r"D:\project\citic_credit_rag"

# 1) 读原始知识库，按 source 拼回完整文本
raw = [json.loads(l) for l in open(os.path.join(BASE, "knowledge_base.jsonl"), encoding="utf-8")]
by_source = defaultdict(list)
for c in raw:
    by_source[c["source"]].append(c)

# 拼成完整文档（按原来的顺序）
docs = []
for src, chunks in by_source.items():
    chunks.sort(key=lambda x: x["chunk_id"])
    full_text = "\n".join(c["text"] for c in chunks)
    docs.append({"source": src, "text": full_text})

print(f"原始文档数: {len(docs)}")
print(f"总字数: {sum(len(d['text']) for d in docs)}")

# 2) 简单分词（中文按字+英文按词）
def tokenize(text):
    text = text.lower()
    # 英文/数字按词，中文按字
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    return tokens

# 3) 按 chunk_size 切块（带 overlap）
def split_chunks(docs, chunk_size, overlap):
    chunks = []
    cid = 0
    for d in docs:
        text = d["text"]
        if len(text) <= chunk_size:
            chunks.append({"chunk_id": cid, "source": d["source"], "text": text})
            cid += 1
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # 尽量在句号/换行处切
            if end < len(text):
                for sep in ["。", "！", "？", "\n"]:
                    p = text.rfind(sep, start, end)
                    if p > start + chunk_size * 0.5:
                        end = p + 1
                        break
            chunks.append({"chunk_id": cid, "source": d["source"], "text": text[start:end]})
            cid += 1
            if end >= len(text):
                break
            start = end - overlap
            if start < 0 or start >= len(text):
                break
    return chunks

# 4) 测试题 + 正确答案的关键词（用于判断是否召回正确）
# 格式：(问题, 正确答案应该包含的关键词)
test_cases = [
    ("信用卡挂失手续费多少？", ["挂失", "40"]),
    ("境外取现限额多少？", ["境外取现", "1万"]),
    ("最低还款有利息吗？", ["最低还款", "免息"]),
    ("违约金怎么收？", ["违约金", "5%"]),
    ("免息还款期最长多少天？", ["免息", "50"]),
    ("信用卡激活怎么弄？", ["激活"]),
    ("补卡费多少？", ["补卡"]),
    ("分期手续费怎么算？", ["分期"]),
    ("取现手续费多少？", ["取现", "2%"]),
    ("信用卡有效期多久？", ["有效期"]),
    ("溢缴款领回收费吗？", ["溢缴款"]),
    ("短信提醒收费吗？", ["短信"]),
    ("境外紧急补卡多少钱？", ["紧急补卡"]),
    ("附属卡年费多少？", ["附属卡", "年费"]),
    ("年费怎么收？", ["年费"]),
    ("账单日怎么查？", ["账单日"]),
    ("信用卡能取现吗？", ["取现"]),
    ("逾期还款有什么后果？", ["逾期"]),
    ("信用卡额度怎么调整？", ["额度"]),
    ("信用卡怎么注销？", ["注销"]),
    ("电子卡和实体卡区别？", ["电子卡"]),
    ("积分怎么累积？", ["积分"]),
    ("信用卡挂失后怎么补卡？", ["挂失", "补卡"]),
    ("最低还款额怎么算？", ["最低还款"]),
    ("外币消费怎么还款？", ["外汇"]),
    ("对账单多久寄一次？", ["对账单"]),
    ("信用卡密码忘了怎么办？", ["密码"]),
    ("什么是循环信用？", ["循环信用"]),
    ("信用卡取现利息怎么算？", ["取现", "利息"]),
    ("新卡多久能收到？", ["邮寄"]),
    ("信用卡能网上支付吗？", ["支付"]),
    ("什么是到期还款日？", ["到期还款日"]),
]

# 5) 跑一组配置
def eval_config(chunk_size, overlap):
    chunks = split_chunks(docs, chunk_size, overlap)
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    hit = 0
    for q, keywords in test_cases:
        scores = bm25.get_scores(tokenize(q))
        top4 = np.argsort(scores)[::-1][:4]
        # 检查Top4里有没有包含所有关键词的块
        found = False
        for idx in top4:
            text = chunks[idx]["text"]
            if all(kw in text for kw in keywords):
                found = True
                break
        if found:
            hit += 1
    recall = hit / len(test_cases)
    return len(chunks), recall

# 6) 测不同配置
configs = [
    (500, 0),
    (500, 20),
    (500, 40),
    (500, 60),
    (500, 80),
    (500, 100),
    (500, 120),
    (500, 140),
    (500, 160),
]

print(f"\n{'chunk_size':<12}{'overlap':<10}{'块数':<8}{'Recall@4':<10}")
print("-" * 40)
for cs, ov in configs:
    n, r = eval_config(cs, ov)
    print(f"{cs:<12}{ov:<10}{n:<8}{r:.0%}")








