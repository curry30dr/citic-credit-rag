# -*- coding: utf-8 -*-
"""从当前86块追加3个海报块，重建索引。"""
import os, json, pickle, numpy as np, faiss
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks = [json.loads(l) for l in open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), encoding="utf-8")]
print("当前块:", len(chunks))

new_blocks = [
    {"source": "活动海报-爱吃版", "topic": "返现规则-餐饮",
     "text": "中信银行信运无界信用卡「爱吃版」返现规则：全年最高返现1200元，每月最高返现100元。微信/支付宝/云闪付消费返现1%；指定美食平台返现15%；指定美食商户返现15%。适用场景：吃饭、餐饮。首年免年费，交易12笔免次年年费。"},
    {"source": "活动海报-爱行版", "topic": "返现规则-出行",
     "text": "中信银行信运无界信用卡「爱行版」返现规则：全年最高返现1200元，每月最高返现100元。微信/支付宝/云闪付消费返现1%；指定出行商户（如网约车、打车）返现15%。适用场景：打车出行。首年免年费，交易12笔免次年年费。"},
    {"source": "活动海报-爱家版", "topic": "返现规则-生活缴费",
     "text": "中信银行信运无界信用卡「爱家版」返现规则：全年最高返现1200元，每月最高返现100元。微信/支付宝/云闪付消费返现1%；指定商户缴纳物业费、电费、燃气费、水费、网费、话费等返现15%。适用场景：物业、水电燃气、生活缴费。首年免年费，交易12笔免次年年费。"},
]
for i, b in enumerate(new_blocks, start=len(chunks)+1):
    b["chunk_id"] = i
    chunks.append(b)
print("追加后:", len(chunks))

with open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

def _tok(s): return list(s.replace(" ", ""))
bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])
with open(os.path.join(BASE_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump({"chunks": chunks, "bm25": bm25}, f)

from sentence_transformers import SentenceTransformer
bge = SentenceTransformer(r"D:\project\models\bge-small-zh-v1.5")
texts = [c["text"] for c in chunks]
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
