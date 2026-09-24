# -*- coding: utf-8 -*-
"""
中信银行信用卡智能咨询助手 - Hugging Face Spaces 线上版
全部走阿里百炼在线API，无需本地模型。
API Key 通过 HF Secrets 环境变量 DASHSCOPE_KEY 注入。
"""
import os, json, numpy as np, faiss, urllib.request
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_KEY", "")
LLM_MODEL = "qwen-plus"
TOP_N = 4
RECALL_K = 10

# ============ 加载知识库 ============
CHUNKS = [json.loads(l) for l in open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), encoding="utf-8")]
FAISS_ONLINE = faiss.read_index(os.path.join(BASE_DIR, "faiss_online.index"))

# 运行时用 jieba 重建 BM25（86条，秒级）
import jieba
from rank_bm25 import BM25Okapi
def _tok(s):
    return [w for w in jieba.lcut(s) if w.strip()]
BM25 = BM25Okapi([_tok(c["text"]) for c in CHUNKS])

# ============ 在线 Embedding ============
def emb_online(texts):
    body = json.dumps({"model": "text-embedding-v3", "input": texts}).encode()
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        data=body, headers={"Authorization": "Bearer " + DASHSCOPE_KEY,
                            "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    arr = sorted(d["data"], key=lambda x: x["index"])
    v = np.array([x["embedding"] for x in arr], dtype="float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

# ============ 在线 Rerank ============
def rerank_online(query, docs, top_n):
    body = json.dumps({"model": "gte-rerank-v2",
                       "input": {"query": query, "documents": docs},
                       "parameters": {"top_n": top_n}}).encode()
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        data=body, headers={"Authorization": "Bearer " + DASHSCOPE_KEY,
                            "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    return [(x["index"], float(x["relevance_score"])) for x in d["output"]["results"]]

# ============ 混合检索 ============
def retrieve(query):
    qv = emb_online([query])
    D, I = FAISS_ONLINE.search(qv, RECALL_K)
    vec_rank = {int(i): r + 1 for r, i in enumerate(I[0])}
    scores = BM25.get_scores(_tok(query))
    top_idx = np.argsort(scores)[::-1][:RECALL_K]
    bm_rank = {int(i): r + 1 for r, i in enumerate(top_idx)}
    cands = set(vec_rank) | set(bm_rank)
    rrf = {}
    for c in cands:
        s = 1 / (60 + vec_rank[c]) if c in vec_rank else 0
        if c in bm_rank:
            s += 1 / (60 + bm_rank[c])
        rrf[c] = s
    fused = sorted(rrf, key=lambda c: -rrf[c])[:TOP_N * 2]
    docs = [CHUNKS[c]["text"] for c in fused]
    rr = rerank_online(query, docs, TOP_N)
    picked = [fused[i] for i, _ in rr]
    return [CHUNKS[c] for c in picked]

# ============ LLM 生成 ============
SYSTEM_PROMPT = (
    "你是中信银行信用卡智能咨询助手。请只根据下面提供的【业务资料】回答用户问题。"
    "要求：1.金额、利率、费用、期限等数字必须原样引用资料，不得编造；"
    "2.若资料中没有相关信息，明确说“根据现有资料无法回答，建议拨打客服热线4008895558”；"
    "3.回答简洁清晰，分点说明。")

def answer(query):
    ctx = retrieve(query)
    ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{query}"}
        ],
        "max_tokens": 700, "temperature": 0.2}).encode()
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body, headers={"Authorization": "Bearer " + DASHSCOPE_KEY,
                            "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.loads(r.read().decode())["choices"][0]["message"]["content"]
    return resp, ctx

# ============ Web ============
app = Flask(__name__, static_folder="static", static_url_path="")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/chat.html")
def chat():
    return send_from_directory("static", "chat.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True)
    q = (body.get("question") or "").strip()
    if not q:
        return jsonify({"error": "问题不能为空"}), 400
    try:
        resp, ctx = answer(q)
        return jsonify({
            "answer": resp,
            "sources": [{"topic": c["topic"], "source": c["source"], "text": c["text"]} for c in ctx]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
