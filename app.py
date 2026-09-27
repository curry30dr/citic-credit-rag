# -*- coding: utf-8 -*-
"""Streamlit 完整版：中信银行信用卡智能咨询助手"""
import os, json
import requests
from rank_bm25 import BM25Okapi
import numpy as np
import streamlit as st

DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"Authorization": "Bearer " + DASHSCOPE_KEY, "Content-Type": "application/json"}

def emb_online(texts):
    all_emb = []
    for i in range(0, len(texts), 10):
        batch = texts[i:i+10]
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            headers=HEADERS,
            json={"model": "text-embedding-v3", "input": batch},
            timeout=60)
        data = resp.json()
        all_emb.extend([d["embedding"] for d in data["data"]])
    return all_emb

def llm_chat(messages):
    resp = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers=HEADERS,
        json={"model": "qwen-plus", "messages": messages, "temperature": 0},
        timeout=60)
    return resp.json()["choices"][0]["message"]["content"]

@st.cache_resource
def load_index():
    chunks = []
    with open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    texts = [c["text"] for c in chunks]
    bm25 = BM25Okapi([t.split() for t in texts])
    emb = np.array(emb_online(texts))
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return chunks, bm25, emb

def retrieve(q):
    chunks, bm25, emb = load_index()
    bm_scores = bm25.get_scores(q.split())
    bm_rank = sorted(range(len(bm_scores)), key=lambda i: -bm_scores[i])
    qv = np.array(emb_online([q])[0])
    qv = qv / (np.linalg.norm(qv) + 1e-8)
    vec_scores = emb @ qv
    vec_rank = sorted(range(len(vec_scores)), key=lambda i: -vec_scores[i])
    k = 60
    rrf = {}
    for rank, idx in enumerate(bm_rank):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(vec_rank):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank + 1)
    top = sorted(rrf.items(), key=lambda x: -x[1])[:8]
    return [chunks[i] for i, _ in top]

def ask(q):
    st.session_state.chat_history.append({"role": "user", "content": q})
    ctx = retrieve(q)
    ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])
    history_msgs = st.session_state.chat_history[-4:-1]
    msgs = [{"role": "system", "content": "你是中信银行信用卡智能咨询助手。仔细阅读业务资料，从资料中找答案，数字必须原样引用，确实没有才说不清楚。"}] + history_msgs + [{"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{q}"}]
    ans = llm_chat(msgs)
    st.session_state.chat_history.append({"role": "assistant", "content": ans, "ctx": ctx})

st.set_page_config(page_title="中信信用卡智能咨询", page_icon="💳", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f6f8; }
div[data-testid="stVerticalBlock"] { gap: 6px !important; }
.main .block-container { padding-top: 8px !important; padding-bottom: 8px !important; }

/* 左侧红色侧边栏 */
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#e60012,#c7000b) !important; width: 220px !important; }
section[data-testid="stSidebar"] .stMarkdown { color: white; }
section[data-testid="stSidebar"] button { background: rgba(255,255,255,0.1) !important; color: white !important; border: none !important; text-align: left !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] button:hover { background: rgba(255,255,255,0.2) !important; }

/* 气泡 */
.user-bubble { background: linear-gradient(135deg,#e60012,#c7000b); color: white; padding: 10px 14px; border-radius: 14px; border-bottom-right-radius: 4px; margin: 6px 0; width: fit-content; max-width: 100%; display: table; font-size: 14px; line-height: 1.7; }
.bot-bubble { background: white; color: #2b2f38; padding: 10px 14px; border-radius: 14px; border-bottom-left-radius: 4px; margin: 6px 0; width: fit-content; max-width: 100%; display: table; border: 1px solid #e8eaef; font-size: 14px; line-height: 1.7; }

/* 普通按钮 */
div[data-testid="stHorizontalBlock"] button {
    background: white !important;
    border: 1px solid #e8eaef !important;
    border-radius: 18px !important;
    color: #555 !important;
    font-size: 12px !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg,#e60012,#c7000b) !important;
    color: white !important;
    border-radius: 22px !important;
}

/* 右侧来源抽屉 */
.src-drawer { position: fixed; top: 0; right: 0; width: 380px; height: 100vh; background: #fafbfc; border-left: 1px solid #e8eaef; z-index: 9999; overflow-y: auto; padding: 20px; box-shadow: -4px 0 16px rgba(0,0,0,0.1); }
.src-card { background: white; border: 1px solid #e8eaef; border-radius: 10px; padding: 12px; margin-bottom: 10px; font-size: 12.5px; }
.src-tag { display: inline-block; background: #fdecec; color: #e60012; padding: 2px 9px; border-radius: 6px; font-size: 11px; margin-bottom: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None
if "show_src" not in st.session_state:
    st.session_state.show_src = False

if st.session_state.pending_q:
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    ask(q)

# ============ 首页 ============
if not st.session_state.chat_history:
    with st.sidebar:
        st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:30px"><div style="width:40px;height:40px;background:white;border-radius:50%;color:#e60012;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:20px">中</div><div><div style="color:white;font-weight:bold;font-size:16px">中信银行</div><div style="color:rgba(255,255,255,0.7);font-size:10px">CHINA CITIC BANK</div></div></div>', unsafe_allow_html=True)
        if st.button("💬 智能客服", key="nav1", use_container_width=True):
            st.rerun()
        if st.button("🔧 技术说明", key="nav2", use_container_width=True):
            st.toast("BM25+向量RRF融合召回，Qwen-Plus生成")
        if st.button("📄 进入对话", key="nav3", use_container_width=True):
            st.session_state.chat_history.append({"role": "assistant", "content": "您好，我是中信银行信用卡智能咨询助手 👋\n\n我只依据《领用合约》《收费价格表》等业务资料为您解答，数字有据可查。\n\n可咨询：激活、取现、最低还款、年费、账单等。"})
            st.rerun()
        st.markdown('<div style="position:fixed;bottom:20px;left:24px;color:rgba(255,255,255,0.8);font-size:11px">24小时客服热线<br><b style="color:white;font-size:14px">4008895558</b></div>', unsafe_allow_html=True)

    # 顶部红色通栏
    st.markdown("""
    <div style="background:linear-gradient(135deg,#e60012,#c7000b);padding:14px 30px;border-radius:10px;margin-bottom:24px">
        <div style="background:rgba(255,255,255,0.95);border-radius:20px;padding:8px 20px;color:#666;font-size:14px;max-width:600px">💬 点击开始咨询信用卡问题 →</div>
    </div>
    """, unsafe_allow_html=True)

    # 欢迎区
    c1, c2 = st.columns([1, 6])
    with c1:
        st.markdown('<div style="width:70px;height:70px;background:linear-gradient(135deg,#e60012,#ff4444);border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:36px;box-shadow:0 6px 16px rgba(230,0,18,0.3)">🤖</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("# 您好，我是中信银行 <span style='color:#e60012'>智能客服</span>", unsafe_allow_html=True)
        st.caption("我可以为您解答信用卡相关问题，依据领用合约与收费价格表，数字有据可查")

    # 5个功能卡片
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
            st.markdown(f'<div style="background:white;border-radius:12px;padding:20px 12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.05)"><div style="font-size:28px;margin-bottom:8px">{icon}</div><div style="font-weight:bold;font-size:14px">{title}</div><div style="font-size:11px;color:#999;margin-top:4px">{desc}</div></div>', unsafe_allow_html=True)
            if st.button("点击咨询", key=f"card{i}", use_container_width=True):
                st.session_state.pending_q = q_text
                st.rerun()

    # 常见问题白色卡片
    st.markdown('<div style="background:white;border-radius:12px;padding:18px 24px;margin-top:20px"><div style="font-size:14px;color:#666;margin-bottom:6px">常见问题</div></div>', unsafe_allow_html=True)
    faqs = ["如何申请信用卡", "账单日和还款日", "逾期后果", "挂失手续费", "最低还款额怎么算", "优惠活动"]
    fcols = st.columns([1,1,1,1,1,1,3])
    for i, faq in enumerate(faqs):
        if fcols[i].button(faq, key=f"faq{i}"):
            st.session_state.pending_q = faq
            st.rerun()

    # 热线提示条
    st.markdown("""
    <div style="background:white;border-left:3px solid #e60012;border-radius:8px;padding:14px 20px;margin-top:20px;font-size:14px">
        🤖 以上问题我可以帮您解答；如果需要人工服务，请拨打 <b style="color:#e60012">24小时客服热线 4008895558</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;font-size:11px;color:#999;margin-top:20px">以上信息依据《领用合约》《信用卡章程》及收费价格表整理，仅供参考，具体以中信银行官方公告为准</div>', unsafe_allow_html=True)

# ============ 对话页面 ============
else:
    # 顶部小字
    st.markdown("""
    <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;color:#8a909c">
        <span>欢迎使用中信银行信用卡智能服务</span>
        <span>24小时客服热线 <b style="color:#e60012">4008895558</b></span>
    </div>
    """, unsafe_allow_html=True)

    # 品牌通栏+按钮
    top_cols = st.columns([7,1,1])
    with top_cols[0]:
        st.markdown("""
        <div style="background:white;padding:12px 20px;border-radius:8px">
            <div style="display:flex;align-items:center;gap:12px">
                <div style="width:38px;height:38px;background:#e60012;border-radius:50%;color:white;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:17px">中</div>
                <b style="font-size:17px">中信银行</b>
                <span style="font-size:10px;color:#999">CHINA CITIC BANK</span>
                <div style="width:1px;height:24px;background:#e8eaef;margin:0 4px"></div>
                <span style="color:#555;font-size:14px">信用卡 · 智能咨询助手</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with top_cols[1]:
        if st.button("← 返回首页"):
            st.session_state.chat_history = []
            st.rerun()
    with top_cols[2]:
        if st.button("📚 来源"):
            st.session_state.show_src = not st.session_state.show_src

    # 消息列表
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            rcols = st.columns([7,3])
            with rcols[1]:
                uc = st.columns([5,1])
                uc[0].markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                uc[1].markdown('<div style="width:34px;height:34px;border-radius:50%;background:#eef0f4;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#666;margin-top:6px">我</div>', unsafe_allow_html=True)
        else:
            rcols = st.columns([3,7])
            with rcols[0]:
                st.markdown('<div style="width:34px;height:34px;border-radius:50%;background:#fdf0f0;display:flex;align-items:center;justify-content:center;font-size:17px;margin-top:6px">🤖</div>', unsafe_allow_html=True)
            with rcols[1]:
                st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                bc = st.columns([1,1,1,8])
                if bc[0].button("📋", key=f"copy_{idx}"):
                    st.toast("已复制")
                if bc[1].button("👍", key=f"up_{idx}"):
                    st.toast("感谢反馈")
                if bc[2].button("👎", key=f"down_{idx}"):
                    st.toast("感谢反馈")

    # 分类快捷按钮
    st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
    qgroups = [
        ("卡片服务", ["卡到了怎么用", "挂失手续费", "年费怎么收", "补卡"]),
        ("费用查询", ["取现手续费与限额", "最低还款利息", "违约金", "分期手续费"]),
        ("账单概念", ["免息期", "补对账单", "有效期", "账单日"]),
    ]
    for label, items in qgroups:
        gc = st.columns([1] + [1]*len(items) + [6])
        gc[0].markdown(f'<span style="font-size:11px;color:#8a909c">{label}</span>', unsafe_allow_html=True)
        for j, qtext in enumerate(items):
            if gc[j+1].button(qtext, key=f"q_{label}_{j}"):
                st.session_state.pending_q = qtext
                st.rerun()

    # 输入框（form支持回车发送）
    bar_cols = st.columns([7,1])
    with bar_cols[0]:
        with st.form(key="chat_form", clear_on_submit=True):
            fc = st.columns([8,1])
            q = fc[0].text_input("问题", placeholder="请输入您的信用卡问题，回车发送…", label_visibility="collapsed")
            submitted = fc[1].form_submit_button("发送", type="primary")
            if submitted and q:
                ask(q)
                st.rerun()
    if bar_cols[1].button("🗑 清空对话"):
        st.session_state.chat_history = []
        st.rerun()

    # 免责声明
    st.markdown('<div style="text-align:center;font-size:11px;color:#999;margin-top:10px">以上信息依据《领用合约》《信用卡章程》及收费价格表整理，仅供参考，具体以中信银行官方公告为准</div>', unsafe_allow_html=True)

    # 右侧来源抽屉
    if st.session_state.show_src:
        last_ctx = None
        for m in reversed(st.session_state.chat_history):
            if "ctx" in m:
                last_ctx = m["ctx"]
                break
        drawer = '<div class="src-drawer"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px"><b>📚 召回知识来源</b></div>'
        if last_ctx:
            for i, cc in enumerate(last_ctx):
                drawer += f'<div class="src-card"><span class="src-tag">资料{i+1} · {cc.get("topic","")}</span><div style="color:#8a909c;font-size:10px;margin-bottom:4px">来源：{cc.get("source","")}</div><div style="color:#444;line-height:1.7">{cc["text"][:200]}</div></div>'
        else:
            drawer += '<div style="color:#999;text-align:center;margin-top:60px;font-size:13px">暂无来源</div>'
        drawer += '</div>'
        st.markdown(drawer, unsafe_allow_html=True)
