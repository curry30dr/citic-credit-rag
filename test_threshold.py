# -*- coding: utf-8 -*-
"""实测rerank分数分布，找最优阈值"""
import sys, os
sys.path.insert(0, r"D:\project\citic_credit_rag")
os.chdir(r"D:\project\citic_credit_rag")
import rag_assistant as R

# 几道测试题（涵盖费用、规则、超纲）
test_questions = [
    "信用卡挂失手续费多少？",
    "境外取现限额多少？",
    "最低还款有利息吗？",
    "信用卡能买股票吗？",
    "年费怎么免？",
    "违约金怎么收？",
    "免息还款期最长多少天？",
    "信用卡激活怎么弄？",
    "补卡费多少？",
    "分期手续费怎么算？",
    "信用卡积分能干嘛？",
    "超限费收多少？",
    "取现手续费多少？",
    "信用卡有效期多久？",
    "账单日怎么查？",
    "溢缴款领回收费吗？",
    "短信提醒收费吗？",
    "境外紧急补卡多少钱？",
    "信用卡能转账吗？",
    "附属卡年费多少？",
]

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    from rag_assistant import emb_online, FAISS_ONLINE, BM25, CHUNKS, _tok, RECALL_K, rerank_online
    import numpy as np
    qv = emb_online([q])
    D, I = FAISS_ONLINE.search(qv, RECALL_K)
    vec_rank = {int(i): r+1 for r, i in enumerate(I[0])}
    scores = BM25.get_scores(_tok(q))
    top_idx = np.argsort(scores)[::-1][:RECALL_K]
    bm_rank = {int(i): r+1 for r, i in enumerate(top_idx)}
    cands = set(vec_rank) | set(bm_rank)
    rrf = {}
    for c in cands:
        s = 1/(60+vec_rank[c]) if c in vec_rank else 0
        if c in bm_rank:
            s += 1/(60+bm_rank[c])
        rrf[c] = s
    fused = sorted(rrf, key=lambda c: -rrf[c])[:16]
    docs = [CHUNKS[c]["text"] for c in fused]
    rr = rerank_online(q, docs, 16)
    print(f"  {'排名':<4}{'分数':<8}{'主题':<16}{'内容前40字'}")
    for rank, (idx, score) in enumerate(rr, 1):
        c = fused[idx]
        preview = CHUNKS[c]["text"][:40].replace("\n", " ")
        print(f"  {rank:<4}{score:<8.4f}{CHUNKS[c]['topic']:<16}{preview}")
