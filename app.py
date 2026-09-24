# -*- coding: utf-8 -*-
"""Streamlit Cloud 部署版：中信银行信用卡智能咨询助手"""
import os, json, numpy as np, faiss, urllib.request
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st

DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    return emb, reranker

@st.cache_data
def load_index():
    chunks = []
    with open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    texts = [c["text"] for c in chunks]
    emb, reranker = load_models()
    vecs = emb.encode(texts, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs.astype("float32"))
    bm25 = BM25Okapi([t.split() for t in texts])
    return chunks, idx, bm25, emb, reranker

def retrieve(q):
    chunks, idx, bm25, emb, reranker = load_index()
    qv = emb.encode([q], normalize_embeddings=True).astype("float32")
    _, vi = idx.search(qv, 10)
    bv = bm25.get_scores(q.split())
    br = np.argsort(bv)[::-1][:10]
    rrf = {}
    for rank, i in enumerate(vi[0]):
        rrf[i] = rrf.get(i, 0) + 1/(60+rank)
    for rank, i in enumerate(br):
        rrf[i] = rrf.get(i, 0) + 1/(60+rank)
    cands = sorted(rrf.items(), key=lambda x: -x[1])[:10]
    pairs = [(q, chunks[i]["text"]) for i, _ in cands]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1]
    return [chunks[cands[i][0]] for i in order[:4] if scores[i] > 0.20]

def llm(q, ctx):
    ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])
    sys_prompt = "你是中信银行信用卡智能咨询助手。只依据提供的业务资料回答，数字必须原样引用，资料没有就说不清楚并引导拨打4008895558。"
    body = json.dumps({
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{q}"}
        ],
        "temperature": 0
    }).encode()
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + DASHSCOPE_KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]

st.set_page_config(page_title="中信信用卡智能咨询", page_icon="💳", layout="centered")
st.title("💳 中信银行信用卡智能咨询助手")
st.caption("仅依据《领用合约》《收费价格表》等业务资料回答，数字有据可查")

q = st.chat_input("请输入您的问题，如：信用卡挂失手续费多少？")
if q:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        ctx = retrieve(q)
        ans = llm(q, ctx)
        st.write(ans)
        with st.expander("查看召回来源"):
            for i, c in enumerate(ctx):
                st.write(f"{i+1}. [{c.get('topic','')}] {c['text'][:100]}...")
