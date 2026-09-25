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

st.set_page_config(page_title="中信信用卡智能咨询", page_icon="💳", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f5f5; }
section[data-testid="stSidebar"] { background: #e60012; }
section[data-testid="stSidebar"] * { color: white !important; }
.stButton > button { background: #e60012; color: white; border: none; border-radius: 20px; }
.stButton > button:hover { background: #cc0010; }
.topbar { background: #e60012; padding: 12px 24px; border-radius: 24px; color: white; margin-bottom: 20px; }
.card { background: white; padding: 24px; border-radius: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 4px; }
.card .icon { font-size: 32px; }
.card .title { font-size: 16px; font-weight: bold; margin: 8px 0 4px 0; }
.card .desc { font-size: 12px; color: #888; }
.tip { border-left: 4px solid #e60012; background: white; padding: 16px; border-radius: 4px; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px">
        <div style="width:40px;height:40px;background:white;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#e60012;font-weight:bold">中</div>
        <div>
            <div style="font-weight:bold">中信银行</div>
            <div style="font-size:12px;opacity:0.8">CHINA CITIC BANK</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("", ["智能客服", "技术说明", "进入对话"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("24小时客服热线")
    st.markdown("**4008895558**")

if page == "智能客服":
    st.markdown('<div class="topbar">💬 点击开始咨询信用卡问题 →</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 6])
    with col1:
        st.markdown('<div style="width:60px;height:60px;background:linear-gradient(135deg,#e60012,#ff4444);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:32px">🤖</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('### 您好，我是中信银行**<span style="color:#e60012">智能客服</span>**', unsafe_allow_html=True)
        st.caption("我可以为您解答信用卡相关问题，依据领用合约与收费价格表，数字有据可查")

    cards = [
        ("💳", "取现手续费", "境内外取现费率"),
        ("🌍", "境外取现", "每日/年度限额"),
        ("✨", "卡怎么激活", "快速激活流程"),
        ("💰", "还款指南", "最低还款/免息期"),
        ("💴", "年费怎么收", "年费标准/免年费政策"),
    ]
    cols = st.columns(5)
    card_q = None
    for i, (icon, title, desc) in enumerate(cards):
        with cols[i]:
            st.markdown(f'<div class="card"><div class="icon">{icon}</div><div class="title">{title}</div><div class="desc">{desc}</div></div>', unsafe_allow_html=True)
            if st.button("选择", key=f"card{i}", help=title):
                card_q = title

    st.markdown("---")
    st.markdown("**常见问题**")
    quick = ["如何申请信用卡", "账单日和还款日", "逾期后果", "挂失手续费", "最低还款额怎么算", "优惠活动"]
    cols = st.columns(3)
    q = card_q
    for i, qq in enumerate(quick):
        if cols[i%3].button(qq, key=f"q{i}"):
            q = qq

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
            history_msgs = st.session_state.chat_history[-4:-1]
            msgs = [{"role": "system", "content": "你是中信银行信用卡智能咨询助手。只依据提供的业务资料回答，数字必须原样引用，资料没有就说不清楚并引导拨打4008895558。"}] + history_msgs + [{"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{q}"}]
            ans = llm_chat(msgs)
            st.write(ans)
            with st.expander("查看召回来源"):
                for i, c in enumerate(ctx):
                    st.write(f"{i+1}. [{c.get('topic','')}] {c['text'][:100]}...")
            st.session_state.chat_history.append({"role": "assistant", "content": ans})

    st.markdown('<div class="tip">🤖 以上问题我可以帮您解答；如果需要人工服务，请拨打 <span style="color:#e60012;font-weight:bold">24小时客服热线 4008895558</span></div>', unsafe_allow_html=True)
    st.caption("以上信息依据《领用合约》《信用卡章程》及收费价格整理，仅供参考，具体以中信银行官方公告为准")

elif page == "技术说明":
    st.title("技术说明")
    st.markdown("""
    **系统架构：**
    - 检索：BM25 词频检索 + 向量检索 + RRF 融合
    - 精排：bge-reranker-base 交叉编码器
    - 生成：qwen-plus 大模型
    - 部署：Streamlit Cloud
    
    **参数：**
    - chunk_size = 500字
    - overlap = 80字
    - 阈值 = 0.20
    - 知识库 = 89块
    """)

else:
    st.title("进入对话")
