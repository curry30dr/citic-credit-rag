# -*- coding: utf-8 -*-
"""Streamlit 完整版：中信银行信用卡智能咨询助手"""
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

def ask(q):
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user", avatar="👤"):
        st.write(q)
    with st.chat_message("assistant", avatar="🤖"):
        ctx = retrieve(q)
        ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])
        history_msgs = st.session_state.chat_history[-4:-1]
        msgs = [{"role": "system", "content": "你是中信银行信用卡智能咨询助手。只依据提供的业务资料回答，数字必须原样引用，资料没有就说不清楚并引导拨打4008895558。"}] + history_msgs + [{"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{q}"}]
        ans = llm_chat(msgs)
        st.write(ans)
        with st.expander("查看召回来源"):
            for i, c in enumerate(ctx):
                st.write(f"{i+1}. [{c.get('topic','')}] {c['text'][:100]}...")
        st.session_state.chat_history.append({"role": "assistant", "content": ans})

st.set_page_config(page_title="中信信用卡智能咨询", page_icon="💳", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f5f5; }
section[data-testid="stSidebar"] { background: #e60012; }
section[data-testid="stSidebar"] * { color: white !important; }
.stButton > button { background: white; color: #333; border: none; border-radius: 12px; text-align: center; }
.stButton > button:hover { background: #fff5f5; color: #e60012; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:30px">
        <div style="width:40px;height:40px;background:white;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#e60012;font-weight:bold;font-size:18px">中</div>
        <div>
            <div style="font-weight:bold;font-size:16px">中信银行</div>
            <div style="font-size:11px;opacity:0.8">CHINA CITIC BANK</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("24小时客服热线")
    st.markdown("**4008895558**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None

if st.session_state.pending_q:
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    ask(q)

if not st.session_state.chat_history:
    st.markdown("""
    <div style="background:#e60012;padding:10px 20px;border-radius:24px;margin-bottom:20px">
        <span style="color:white;opacity:0.9">💬 点击开始咨询信用卡问题 →</span>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 8])
    with col1:
        st.markdown("""
        <div style="width:60px;height:60px;background:linear-gradient(135deg,#e60012,#ff4444);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:32px;box-shadow:0 4px 12px rgba(230,0,18,0.3)">🤖</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("## 您好，我是中信银行 <span style='color:#e60012'>智能客服</span>", unsafe_allow_html=True)
        st.caption("我可以为您解答信用卡相关问题，依据领用合约与收费价格表，数字有据可查")

    cards = [
        ("💳", "取现手续费", "境内外取现费率", "信用卡取现手续费多少？"),
        ("🌍", "境外取现", "每日/年度限额", "境外取现限额多少？"),
        ("✨", "卡怎么激活", "快速激活流程", "信用卡怎么激活？"),
        ("💰", "还款指南", "最低还款/免息期", "最低还款有利息吗？"),
        ("💴", "年费怎么收", "年费标准/免年费政策", "年费怎么免？"),
    ]
    cols = st.columns(5)
    for i, (icon, title, desc, q_text) in enumerate(cards):
        with cols[i]:
            if st.button(f"{icon} {title}\n{desc}", key=f"card{i}"):
                st.session_state.pending_q = q_text
                st.rerun()

    st.markdown("**常见问题**")
    faqs = ["如何申请信用卡", "账单日和还款日", "逾期后果", "挂失手续费", "最低还款额怎么算", "优惠活动"]
    cols = st.columns(6)
    for i, faq in enumerate(faqs):
        if cols[i].button(faq, key=f"faq{i}"):
            st.session_state.pending_q = faq
            st.rerun()

else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🤖"):
            st.write(msg["content"])

    q = st.chat_input("请输入您的问题")
    if q:
        ask(q)

    if st.button("🗑 返回首页/清空对话"):
        st.session_state.chat_history = []
        st.rerun()
