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

# 中信红品牌风格
st.markdown("""
<style>
:root { --brand: #e60012; }
.stApp { background: #fafafa; }
h1 { color: #e60012 !important; }
.stChatInput > div > div > input:focus { border-color: #e60012 !important; }
.stButton > button { background: #e60012; color: white; border: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#e60012,#ff4444);padding:24px;border-radius:12px;color:white;margin-bottom:16px">
<h2 style="color:white;margin:0">中信银行信用卡智能咨询助手</h2>
<p style="opacity:0.9;margin:8px 0 0">智能客服 · 数字有据可查 · 7×24小时服务</p>
</div>
""", unsafe_allow_html=True)

st.caption("仅依据《领用合约》《收费价格表》等业务资料整理，具体以中信银行官方公告为准")

# 快捷问题
st.markdown("**常见问题：**")
quick = ["信用卡挂失手续费多少？", "境外取现限额多少？", "最低还款有利息吗？", "违约金怎么收？", "溢缴款领回收费吗？"]
cols = st.columns(3)
for i, qq in enumerate(quick):
    if cols[i%3].button(qq, key=f"q{i}"):
        st.session_state["quick"] = qq

q = st.chat_input("请输入您的问题")
if "quick" in st.session_state:
    q = st.session_state.pop("quick")
if q:
    with st.chat_message("user", avatar="👤"):
        st.write(q)
    with st.chat_message("assistant", avatar="🤖"):
        ctx = retrieve(q)
        ans = llm(q, ctx)
        st.write(ans)
        with st.expander("查看召回来源"):
            for i, c in enumerate(ctx):
                st.write(f"{i+1}. [{c.get('topic','')}] {c['text'][:100]}...")

st.markdown("---")
st.caption("如需人工服务，请拨打中信银行信用卡客服热线 4008895558")
