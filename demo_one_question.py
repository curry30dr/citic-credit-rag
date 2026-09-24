# -*- coding: utf-8 -*-
"""现场演示：一道题在四个版本下分别召回了什么"""
import numpy as np, faiss
import rag_assistant as R

q = "取现手续费怎么收？"
print("提问：", q)
print("="*60)

def show(name, idx):
    print(f"\n【{name}】找回来的前4篇：")
    for rank, i in enumerate(idx, 1):
        c = R.CHUNKS[i]
        print(f"  {rank}. [{c['topic']}] {c['text'][:55]}...")

# A 纯向量
qe = R.emb_local(["为这个句子生成表示以用于检索相关文章："+q])
D, I = R.FAISS_LOCAL.search(np.array(qe,dtype="float32"), 4)
show("A 纯向量", [int(x) for x in I[0]])

# B 纯BM25
sc = R.BM25.get_scores(R._tok(q))
bi = list(np.argsort(sc)[::-1][:4])
show("B 纯BM25", [int(x) for x in bi])

# C 混合RRF
D, I = R.FAISS_LOCAL.search(np.array(qe,dtype="float32"), 10)
vr = {int(i):r+1 for r,i in enumerate(I[0])}
ti = np.argsort(R.BM25.get_scores(R._tok(q)))[::-1][:10]
br = {int(i):r+1 for r,i in enumerate(ti)}
cands = set(vr)|set(br); rrf={}
for c in cands:
    s = 1/(60+vr[c]) if c in vr else 0
    if c in br: s += 1/(60+br[c])
    rrf[c]=s
ci = sorted(rrf, key=lambda x:-rrf[x])[:4]
show("C 混合RRF", ci)

# D 混合+Rerank
ctx = R.retrieve(q, "local")
di = [R.CHUNKS.index(c) for c in ctx]
show("D 混合+Rerank", di)
