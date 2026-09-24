# -*- coding: utf-8 -*-
"""Streamlit Cloud 部署版：中信银行信用卡智能咨询助手（完整版）"""
import os, json, urllib.request
from rank_bm25 import BM25Okapi
import streamlit as st

DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def llm_chat(messages):
    body = json.dumps({"model": "qwen-plus", "messages": messages, "temperature": 0}).encode()
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + DASHSCOPE_KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]

@st.cache_resource
def load_bm25():
    chunks = []
    with open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    texts = [c["text"] for c in chunks]
    bm25 = BM25Okapi([t.split() for t in texts])
    return chunks, bm25

def retrieve(q):
    chunks, bm25 = load_bm25()
    scores = bm25.get_scores(q.split())
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:4]
    return [chunks[i] for i in top]

st.set_page_config(page_title="中信信用卡智能咨询", page_icon="💳", layout="centered")

# 品牌风格
st.markdown("""
<style>
.stApp { background: #fafafa; }
.main .block-container { padding-top: 2rem; }
.stButton > button { background: #e60012; color: white; border: none; border-radius: 8px; }
.stButton > button:hover { background: #cc0010; }
[data-testid="stChatInput"] input:focus { border-color: #e60012 !important; }
</style>
""", unsafe_allow_html=True)

# 顶栏
st.markdown("""
<div style="background:linear-gradient(135deg,#e60012,#ff4444);padding:24px;border-radius:12px;color:white;margin-bottom:16px">
<h2 style="color:white;margin:0">中信银行信用卡智能咨询助手</h2>
<p style="opacity:0.9;margin:8px 0 0">智能客服 · 数字有据可查 · 7×24小时服务</p>
</div>
""", unsafe_allow_html=True)

# 清空按钮
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🗑 清空对话"):
        st.session_state.chat_history = []
        st.rerun()

st.caption("仅依据《领用合约》《收费价格表》等业务资料整理，具体以中信银行官方公告为准")

# 快捷问题
st.markdown("**常见问题：**")
quick = ["信用卡挂失手续费多少？", "境外取现限额多少？", "最低还款有利息吗？", "违约金怎么收？", "溢缴款领回收费吗？"]
cols = st.columns(3)
q = None
for i, qq in enumerate(quick):
    if cols[i%3].button(qq, key=f"q{i}"):
        q = qq

# 对话历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🤖"):
        st.write(msg["content"])

if not q:
    q = st.chat_input("请输入您的问题")

if q:
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user", avatar="👤"):
        st.write(q)
    with st.chat_message("assistant", avatar="🤖"):
        ctx = retrieve(q)
        ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])
        # 多轮对话：带上最近2轮历史
        history_msgs = st.session_state.chat_history[-4:-1]  # 最近2轮
        msgs = [
            {"role": "system", "content": "你是中信银行信用卡智能咨询助手。只依据提供的业务资料回答，数字必须原样引用，资料没有就说不清楚并引导拨打4008895558。"}
        ] + history_msgs + [
            {"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{q}"}
        ]
        ans = llm_chat(msgs)
        st.write(ans)
        with st.expander("查看召回来源"):
            for i, c in enumerate(ctx):
                st.write(f"{i+1}. [{c.get('topic','')}] {c['text'][:100]}...")
        st.session_state.chat_history.append({"role": "assistant", "content": ans})

st.markdown("---")
st.caption("如需人工服务，请拨打中信银行信用卡客服热线 4008895558")
