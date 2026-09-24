# -*- coding: utf-8 -*-
"""
消融实验脚本：对比 纯向量 / 纯BM25 / 混合RRF / 混合+Rerank 四种检索方案
指标：Recall@4 = 标准答案关键词被 Top4 召回文档覆盖的比例
"""
import os, json, numpy as np, pickle, faiss
import rag_assistant as R

# ============ 20 道模拟测试题（答案关键词用于判断召回是否命中）============
TESTS = [
    {"q": "信用卡怎么激活？", "keys": ["动卡空间", "40088"]},
    {"q": "境外取现金有什么限额？", "keys": ["1万", "10万"]},
    {"q": "取现手续费怎么收？", "keys": ["2%", "20"]},
    {"q": "信用卡透支日利率是多少？", "keys": ["0.05%"]},
    {"q": "免息还款期最长多少天？", "keys": ["50天"]},
    {"q": "只还最低还款额有没有利息？", "keys": ["免息", "利息"]},
    {"q": "最低还款没还要收什么费用？", "keys": ["5%", "违约金"]},
    {"q": "普卡年费多少钱？", "keys": ["100"]},
    {"q": "卡片挂失手续费多少？", "keys": ["40"]},
    {"q": "ATM机取现每卡每日限额多少？", "keys": ["1万"]},
    {"q": "柜台取现每卡每日限额多少？", "keys": ["5万"]},
    {"q": "境外消费兑换手续费多少？", "keys": ["1.5%"]},
    {"q": "循环信用利息按什么利率算？", "keys": ["万分之五", "0.05%"]},
    {"q": "超信用额度用卡一个账单周期能用几次？", "keys": ["一次", "一个账单周期"]},
    {"q": "信用卡有效期最长多少年？", "keys": ["20年"]},
    {"q": "调单费收多少？", "keys": ["20"]},
    {"q": "补制对账单有免费额度吗？", "keys": ["12个月", "免费一次"]},
    {"q": "信用卡短信宝怎么收费？", "keys": ["12", "季度"]},
    {"q": "提前结束分期收违约金吗？", "keys": ["0-3%", "违约金"]},
    {"q": "还最低还款额不享免息期是什么意思？", "keys": ["免息", "记账日"]},
]

chunks = R.CHUNKS
texts = [c["text"] for c in chunks]

# ============ 四种检索 ============
def vec_top4(q):
    qe = R.emb_local(["为这个句子生成表示以用于检索相关文章：" + q])
    D, I = R.FAISS_LOCAL.search(np.array(qe, dtype="float32"), 4)
    return [int(i) for i in I[0]]

def bm25_top4(q):
    sc = R.BM25.get_scores(R._tok(q))
    return list(np.argsort(sc)[::-1][:4])

def hybrid_rrf_top4(q):
    qe = R.emb_local(["为这个句子生成表示以用于检索相关文章：" + q])
    D, I = R.FAISS_LOCAL.search(np.array(qe, dtype="float32"), 10)
    vr = {int(i): r+1 for r, i in enumerate(I[0])}
    sc = R.BM25.get_scores(R._tok(q))
    ti = np.argsort(sc)[::-1][:10]
    br = {int(i): r+1 for r, i in enumerate(ti)}
    cands = set(vr) | set(br); rrf = {}
    for c in cands:
        s = 1/(60+vr[c]) if c in vr else 0
        if c in br: s += 1/(60+br[c])
        rrf[c] = s
    return sorted(rrf, key=lambda c: -rrf[c])[:4]

def hybrid_rerank_top4(q):
    ctx = R.retrieve(q, "local")
    return [chunks.index(c) for c in ctx]

def recall(idx_list, keys):
    """召回的4篇文档是否覆盖答案关键词（命中一半以上算召回成功）"""
    docs = " ".join(texts[i] for i in idx_list)
    hit = sum(1 for k in keys if k in docs)
    return hit / len(keys)

# ============ 跑评测 ============
methods = {
    "A 纯向量": vec_top4,
    "B 纯BM25": bm25_top4,
    "C 混合RRF": hybrid_rrf_top4,
    "D 混合+Rerank": hybrid_rerank_top4,
}
results = {m: [] for m in methods}
for t in TESTS:
    for m, fn in methods.items():
        try:
            idx = fn(t["q"])
            results[m].append(recall(idx, t["keys"]))
        except Exception as e:
            print("ERR", m, t["q"], e)
            results[m].append(0)

print("\n" + "="*55)
print("消融实验结果（Recall@4，越高越好）")
print("="*55)
print(f"{'方案':<18}{'平均Recall@4':<15}{'逐题命中(20题)'}")
print("-"*55)
# 逐题明细
detail = {m: [1 if x>=0.5 else 0 for x in results[m]] for m in methods}
for m in methods:
    avg = np.mean(results[m])
    hitcount = sum(detail[m])
    print(f"{m:<18}{avg*100:>6.1f}%   {hitcount}/20 题命中")
print("-"*55)
print("\n逐题命中明细（●=命中 ○=未命中）：")
header = "题号  " + "".join([f"{i+1:>4}" for i in range(20)])
print(header)
for m in methods:
    row = "     ".join("●" if x else "○" for x in detail[m])
    print(f"{m:<14}{row}")
