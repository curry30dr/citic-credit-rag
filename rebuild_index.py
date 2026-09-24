# -*- coding: utf-8 -*-
"""把活动海报的返现规则追加进知识库，重建 BM25 + 双 FAISS 索引。"""
import os, json, pickle, numpy as np, faiss
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) 读现有知识库
chunks = [json.loads(l) for l in open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), encoding="utf-8")]
print("现有知识块:", len(chunks))

# 2) 追加活动海报知识块（来自"2026年活动海报"图片OCR）
new_blocks = [
    {
        "source": "活动海报-爱吃版",
        "topic": "返现规则-餐饮",
        "text": (
            "中信银行信运无界信用卡「爱吃版」返现规则：\n"
            "全年最高返现1200元，每月最高返现100元。\n"
            "微信/支付宝/云闪付消费返现1%；指定美食平台返现15%；指定美食商户返现15%。\n"
            "适用场景：吃饭、餐饮。首年免年费，交易12笔免次年年费。"
        )
    },
    {
        "source": "活动海报-爱行版",
        "topic": "返现规则-出行",
        "text": (
            "中信银行信运无界信用卡「爱行版」返现规则：\n"
            "全年最高返现1200元，每月最高返现100元。\n"
            "微信/支付宝/云闪付消费返现1%；指定出行商户（如网约车、打车）返现15%。\n"
            "适用场景：打车出行。首年免年费，交易12笔免次年年费。"
        )
    },
    {
        "source": "活动海报-爱家版",
        "topic": "返现规则-生活缴费",
        "text": (
            "中信银行信运无界信用卡「爱家版」返现规则：\n"
            "全年最高返现1200元，每月最高返现100元。\n"
            "微信/支付宝/云闪付消费返现1%；指定商户缴纳物业费、电费、燃气费、水费、网费、话费等返现15%。\n"
            "适用场景：物业、水电燃气、生活缴费。首年免年费，交易12笔免次年年费。"
        )
    },
]
for i, b in enumerate(new_blocks, start=len(chunks)+1):
    b["chunk_id"] = i
    chunks.append(b)
print("追加后知识块:", len(chunks))

# 3) 写回 knowledge_base.jsonl
with open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print("knowledge_base.jsonl 已更新")

# 4) 建 BM25
def _tok(s):
    return list(s.replace(" ", ""))
tokenized = [_tok(c["text"]) for c in chunks]
bm25 = BM25Okapi(tokenized)
with open(os.path.join(BASE_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump({"chunks": chunks, "bm25": bm25}, f)
print("bm25.pkl 已重建")

# 5) 建本地 FAISS（bge-small-zh 512维）
from sentence_transformers import SentenceTransformer
EMB_LOCAL = r"D:\project\models\bge-small-zh-v1.5"
bge = SentenceTransformer(EMB_LOCAL)
texts = [c["text"] for c in chunks]
emb_local = bge.encode(texts, normalize_embeddings=True).astype("float32")
idx_local = faiss.IndexFlatIP(emb_local.shape[1])
idx_local.add(emb_local)
faiss.write_index(idx_local, os.path.join(BASE_DIR, "faiss.index"))
print("faiss.index 已重建，dim=", emb_local.shape[1], "count=", idx_local.ntotal)

# 6) 建在线 FAISS（text-embedding-v3 1024维）
import urllib.request
DASHSCOPE_KEY = "sk-ws-H.PHEPREI.wzWE.MEQCIFbO9FpQzB_5mbvsyhs6-bJeEIh83CP8bqgZyU95NLJVAiBwKTHfLrH8Ip-FVyg-RECAUHnwKIIeWwuR03veHJYSMA"
def emb_online(texts):
    out = []
    for t in texts:
        body = json.dumps({"model": "text-embedding-v3", "input": [t]}).encode()
        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            data=body, headers={"Authorization": "Bearer " + DASHSCOPE_KEY,
                                 "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode())
            out.append(d["data"][0]["embedding"])
        except Exception as e:
            print("embedding失败:", t[:30], e)
            out.append([0.0]*1024)
    return np.array(out, dtype="float32")
emb_on = emb_online(texts)
emb_on /= np.linalg.norm(emb_on, axis=1, keepdims=True).reshape(-1, 1)
idx_on = faiss.IndexFlatIP(emb_on.shape[1])
idx_on.add(emb_on.astype("float32"))
faiss.write_index(idx_on, os.path.join(BASE_DIR, "faiss_online.index"))
print("faiss_online.index 已重建，dim=", emb_on.shape[1], "count=", idx_on.ntotal)
print("完成！")
