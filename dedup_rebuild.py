# -*- coding: utf-8 -*-
"""去重 knowledge_base.jsonl，按 text 去重后重建索引。"""
import os, json, pickle, numpy as np, faiss
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks = [json.loads(l) for l in open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), encoding="utf-8")]
print("去重前:", len(chunks))

seen = set()
uniq = []
for c in chunks:
    key = c["text"][:50]
    if key in seen:
        continue
    seen.add(key)
    uniq.append(c)
# 重新编号
for i, c in enumerate(uniq, start=1):
    c["chunk_id"] = i
print("去重后:", len(uniq))

with open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), "w", encoding="utf-8") as f:
    for c in uniq:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

def _tok(s): return list(s.replace(" ", ""))
bm25 = BM25Okapi([_tok(c["text"]) for c in uniq])
with open(os.path.join(BASE_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump({"chunks": uniq, "bm25": bm25}, f)

from sentence_transformers import SentenceTransformer
bge = SentenceTransformer(r"D:\project\models\bge-small-zh-v1.5")
texts = [c["text"] for c in uniq]
v = bge.encode(texts, normalize_embeddings=True).astype("float32")
idx = faiss.IndexFlatIP(v.shape[1]); idx.add(v)
faiss.write_index(idx, os.path.join(BASE_DIR, "faiss.index"))
print("本地faiss:", idx.ntotal)

import urllib.request
KEY = "sk-ws-H.PHEPREI.wzWE.MEQCIFbO9FpQzB_5mbvsyhs6-bJeEIh83CP8bqgZyU95NLJVAiBwKTHfLrH8Ip-FVyg-RECAUHnwKIIeWwuR03veHJYSMA"
outs = []
for t in texts:
    body = json.dumps({"model": "text-embedding-v3", "input": [t]}).encode()
    req = urllib.request.Request("https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        data=body, headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    outs.append(d["data"][0]["embedding"])
v2 = np.array(outs, dtype="float32")
v2 /= np.linalg.norm(v2, axis=1, keepdims=True).reshape(-1, 1)
idx2 = faiss.IndexFlatIP(v2.shape[1]); idx2.add(v2.astype("float32"))
faiss.write_index(idx2, os.path.join(BASE_DIR, "faiss_online.index"))
print("在线faiss:", idx2.ntotal)
print("完成")
