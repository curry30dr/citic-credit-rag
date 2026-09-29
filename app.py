# -*- coding: utf-8 -*-
"""Streamlit 完整版：中信银行信用卡智能咨询助手"""
import os, json, html
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

/* 顶部全宽红色通栏容器 */
/* 主内容区左右padding归零，使红条可填满到边缘 */
.stMainBlockContainer.block-container { padding: 3.5rem 0 0 0 !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0 !important; margin-top: 0 !important; }
.stSidebar .stSidebarContent { padding-top: 0 !important; }
.stApp section[data-testid="stSidebar"], section[data-testid="stSidebar"] { top: 3.5rem !important; height: calc(100vh - 3.5rem) !important; margin-top: 0 !important; }

/* 首页顶部全宽红条（width100%，无负margin） */
.st-key-topbanner {
    background: linear-gradient(135deg,#e60012,#c7000b) !important;
    width: 100% !important;
    margin: -6px 0 0 0 !important;
    padding: 15px clamp(1rem,4vw,3.5rem) !important;
    border-radius: 0 !important;
    position: static !important;
    left: auto !important;
    box-sizing: border-box !important;
}
.st-key-topbanner .stButton { text-align: left !important; width: 100% !important; }
.st-key-topbanner .stButton button {
    background: white !important;
    color: #555 !important;
    border-radius: 22px !important;
    border: none !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    width: auto !important;
    min-width: 400px !important;
    height: auto !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}
.st-key-topbanner .stButton button p { margin: 0 !important; text-align: left !important; }
.st-key-topbanner button * { justify-content: flex-start !important; }
.st-key-topbanner .stButton button:hover { background: #fff5f5 !important; color: #e60012 !important; }

/* 内层内容容器：恢复响应式左右内边距 */
.st-key-contentpad { padding: 1rem clamp(1rem,4vw,3.5rem) 2rem; }

/* 功能卡片行：整卡可点 */
.st-key-cardrow button {
    background: linear-gradient(160deg,#ffffff,#f7f5fc) !important;
    border: 1px solid #f0f1f4 !important;
    border-radius: 12px !important;
    height: 135px !important;
    white-space: pre-wrap !important;
    color: #2b2f38 !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    padding: 16px 8px !important;
}
.st-key-cardrow button[key="card4"] { background: linear-gradient(160deg,#ffffff,#fff0f1) !important; }
.st-key-cardrow button:hover { border-color: #e60012 !important; color: #e60012 !important; }
.st-key-cardrow button p { white-space: pre-wrap !important; text-align: center !important; margin: 0 !important; line-height: 1.9 !important; }
.st-key-cardrow button div { justify-content: center !important; }

/* 常见问题：纯白无边框卡片 */
.st-key-faqbox { background: white; border-radius: 12px; padding: 16px 22px 20px; border: none; }
.st-key-faqbox button { background: white !important; border: 1px solid #eceef2 !important; border-radius: 18px !important; color: #555 !important; font-size: 12px !important; white-space: nowrap !important; padding: 6px 14px !important; }
.st-key-faqbox button:hover { border-color: #e60012 !important; color: #e60012 !important; }
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

/* ===== 对话页（对齐F版） ===== */
.s-topbar { background:#fff; border-bottom:1px solid #e8eaef; padding:11px 32px; font-size:12.5px; color:#8a909c; display:flex; justify-content:space-between; }
.st-key-s-header { background:#fff; border-bottom:1px solid #e8eaef; padding:10px 24px; }
.s-brand { display:flex; align-items:center; gap:14px; }
.s-logo { width:44px;height:44px;border-radius:50%;background:#fff;border:1px solid #f0e0e0;box-shadow:0 2px 8px rgba(230,0,18,.15);color:#e60012;font-weight:800;font-size:19px;display:flex;align-items:center;justify-content:center; }
.s-bank { font-size:20px;font-weight:700;color:#222;letter-spacing:1px;line-height:1.1; }
.s-sub { font-size:11.5px;color:#8a909c; }
.s-divider { width:1px;height:28px;background:#e8eaef;margin:0 4px; }
.s-cardlabel { font-size:14.5px;color:#555;font-weight:600; }
.st-key-go_home button { background:#fff !important; border:none !important; color:#666 !important; font-weight:600 !important; height:40px !important; }
.st-key-go_src button { background:#f2f3f6 !important; border:1px solid #e8eaef !important; color:#555 !important; font-weight:600 !important; height:40px !important; border-radius:18px !important; }
.st-key-s-msgs { background:#f5f6f8; padding:24px 32px; min-height:180px; }
.b-row { display:flex; gap:12px; max-width:84%; align-items:flex-start; margin-bottom:4px; }
.b-avatar { width:40px;height:40px;border-radius:50%;background:#fdf0f0;flex:none;display:flex;align-items:center;justify-content:center;font-size:20px; }
.b-bubble { background:#fff;border:1px solid #e8eaef;border-radius:14px;border-bottom-left-radius:4px;padding:12px 16px;width:fit-content;max-width:100%;font-size:14.5px;line-height:1.8;white-space:pre-wrap;word-break:break-word;box-shadow:0 1px 3px rgba(0,0,0,.04); }
.u-row { display:flex;gap:12px;flex-direction:row-reverse;max-width:84%;margin:0 0 4px auto;align-items:flex-start; }
.u-avatar { width:40px;height:40px;border-radius:50%;background:#eef0f4;flex:none;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#666; }
.u-bubble { background:linear-gradient(135deg,#e60012,#c7000b);color:#fff;border-radius:14px;border-bottom-right-radius:4px;padding:12px 16px;width:fit-content;max-width:100%;font-size:14.5px;line-height:1.8;white-space:pre-wrap;word-break:break-word; }
[class*="st-key-act"] { padding-left:52px; margin-bottom:6px; }
[class*="st-key-act"] button { background:#fff !important;border:1px solid #e8eaef !important;color:#999 !important;border-radius:6px !important;padding:3px 10px !important;font-size:11.5px !important;height:auto !important;min-height:0 !important;width:auto !important; }
.st-key-s-chips { background:#fff; padding:10px 32px 2px; }
.g-label { font-size:11.5px;color:#8a909c; }
.st-key-s-chips button { width:auto !important;white-space:nowrap !important;padding:4px 12px !important;font-size:12px !important;border-radius:18px !important;background:#fff !important;border:1px solid #e8eaef !important;color:#555 !important;height:auto !important;min-height:0 !important; }
.st-key-s-chips button:hover { color:#e60012 !important;border-color:#e60012 !important; }
.st-key-s-input { background:#fff; padding:10px 32px 16px; border-top:1px solid #e8eaef; }
.st-key-s-input input { background:#f5f6f8 !important; border-radius:22px !important; }
.st-key-s-input .stButton button { border-radius:22px !important; height:42px !important; padding:0 18px !important; }
.st-key-s-input button[kind="primaryFormSubmit"] { background:linear-gradient(135deg,#e60012,#c7000b) !important; color:#fff !important; border:none !important; font-weight:700 !important; }
.s-disclaimer { background:#fff; border-top:1px solid #e8eaef; padding:9px 32px; font-size:11px; color:#8a909c; text-align:center; }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None
if "show_src" not in st.session_state:
    st.session_state.show_src = False
if "in_chat" not in st.session_state:
    st.session_state.in_chat = False

if st.session_state.pending_q:
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    st.session_state.in_chat = True
    ask(q)

# ============ 首页 ============
if not st.session_state.in_chat:
    with st.sidebar:
        st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:30px"><div style="width:40px;height:40px;background:white;border-radius:50%;color:#e60012;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:20px">中</div><div><div style="color:white;font-weight:bold;font-size:16px">中信银行</div><div style="color:rgba(255,255,255,0.7);font-size:10px">CHINA CITIC BANK</div></div></div>', unsafe_allow_html=True)
        if st.button("💬 智能客服", key="nav1", use_container_width=True):
            st.rerun()
        if st.button("🔧 技术说明", key="nav2", use_container_width=True):
            st.toast("BM25+向量RRF融合召回，Qwen-Plus生成")
        if st.button("📄 进入对话", key="nav3", use_container_width=True):
            st.session_state.in_chat = True
            st.rerun()
        st.markdown('<div style="position:fixed;bottom:20px;left:24px;color:rgba(255,255,255,0.8);font-size:11px">24小时客服热线<br><b style="color:white;font-size:14px">4008895558</b></div>', unsafe_allow_html=True)

    # 顶部全宽红色通栏（真实容器，可点击）
    with st.container(key="topbanner"):
        if st.button("💬  点击开始咨询信用卡问题  →", key="start_chat"):
            st.session_state.in_chat = True
            st.rerun()

    with st.container(key="contentpad"):
        # 欢迎区
        c1, c2 = st.columns([0.75, 10], gap="small")
        with c1:
            st.markdown("""<div style="width:72px;height:72px;background:linear-gradient(135deg,#e60012,#ff3344);border-radius:20px;display:flex;align-items:center;justify-content:center;box-shadow:0 8px 20px rgba(230,0,18,0.35)"><svg width="46" height="46" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="hg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#dcd3f6"/></linearGradient><linearGradient id="fg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7c6fe0"/><stop offset="1" stop-color="#4a409f"/></linearGradient><radialGradient id="bl" cx="0.35" cy="0.35" r="0.85"><stop offset="0" stop-color="#ffe27a"/><stop offset="1" stop-color="#f5a623"/></radialGradient><linearGradient id="eg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#eae5fb"/><stop offset="1" stop-color="#c6bcef"/></linearGradient></defs><rect x="7" y="29" width="7" height="13" rx="3.5" fill="url(#eg)"/><rect x="50" y="29" width="7" height="13" rx="3.5" fill="url(#eg)"/><rect x="30.5" y="9" width="3" height="9" rx="1.5" fill="#e8e2f7"/><circle cx="32" cy="8" r="4.2" fill="url(#bl)"/><rect x="14" y="19" width="36" height="31" rx="9" fill="url(#hg)" stroke="#c4b9ee" stroke-width="0.8"/><rect x="19" y="25" width="26" height="17" rx="6" fill="url(#fg)"/><circle cx="26.5" cy="33.5" r="3.2" fill="#8fd3ff"/><circle cx="37.5" cy="33.5" r="3.2" fill="#8fd3ff"/><circle cx="27.4" cy="32.6" r="1" fill="#eafaff"/><circle cx="38.4" cy="32.6" r="1" fill="#eafaff"/><rect x="27" y="45" width="10" height="3" rx="1.5" fill="#a78bfa"/></svg></div>""", unsafe_allow_html=True)
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
        with st.container(key="cardrow"):
            cols = st.columns(5)
            for i, (icon, title, desc, q_text) in enumerate(cards):
                with cols[i]:
                    if st.button(f"{icon}\n**{title}**\n{desc}", key=f"card{i}", use_container_width=True):
                        st.session_state.pending_q = q_text
                        st.session_state.in_chat = True
                        st.rerun()

        # 常见问题纯白无边框卡片（pills自适应宽度，不截断）
        with st.container(key="faqbox"):
            faqs = ["如何申请信用卡", "账单日和还款日", "逾期后果", "挂失手续费", "最低还款额怎么算", "优惠活动"]
            picked = st.pills("常见问题", faqs, key="faq_pills")
            if picked:
                st.session_state.pending_q = picked
                st.session_state.in_chat = True
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
    # 顶部小字栏（全宽白底）
    st.markdown("""
    <div class="s-topbar">
      <span>欢迎使用中信银行信用卡智能服务</span>
      <span>24小时客服热线 <b style="color:#e60012">4008895558</b></span>
    </div>
    """, unsafe_allow_html=True)

    # 品牌栏：左品牌 右返回首页+来源（紧挨）
    with st.container(key="s-header"):
        hc = st.columns([11, 1.1, 1.3])
        with hc[0]:
            st.markdown("""
            <div class="s-brand">
              <div class="s-logo">中</div>
              <div><div class="s-bank">中信银行</div><div class="s-sub">CHINA CITIC BANK</div></div>
              <div class="s-divider"></div>
              <div class="s-cardlabel">信用卡 · 智能咨询助手</div>
            </div>
            """, unsafe_allow_html=True)
        with hc[1]:
            if st.button("← 返回首页", key="go_home"):
                st.session_state.in_chat = False
                st.session_state.chat_history = []
                st.rerun()
        with hc[2]:
            if st.button("📚 来源", key="go_src"):
                st.session_state.show_src = not st.session_state.show_src
                st.rerun()

    # 消息区（浅灰底，气泡fit-content左右分置）
    with st.container(key="s-msgs"):
        st.markdown('<div class="b-row"><div class="b-avatar">🤖</div><div class="b-bubble">您好，我是中信银行信用卡智能咨询助手 👋\n我只依据《领用合约》《收费价格表》等业务资料为您解答，数字有据可查。\n可咨询：激活、取现、最低还款、年费、账单等。</div></div>', unsafe_allow_html=True)
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f'<div class="u-row"><div class="u-avatar">我</div><div class="u-bubble">{html.escape(msg["content"])}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="b-row"><div class="b-avatar">🤖</div><div class="b-bubble">{html.escape(msg["content"])}</div></div>', unsafe_allow_html=True)
                with st.container(key=f"act{idx}"):
                    ac = st.columns([0.8, 0.8, 0.8, 12])
                    if ac[0].button("📋 复制", key=f"cp{idx}"):
                        st.toast("已复制到剪贴板")
                    if ac[1].button("👍 赞", key=f"up{idx}"):
                        st.toast("感谢反馈")
                    if ac[2].button("👎 踩", key=f"dn{idx}"):
                        st.toast("感谢反馈")

    # 分类快捷标签（白底，小标签不截断）
    with st.container(key="s-chips"):
        qgroups = [
            ("卡片服务", [("卡到了怎么用", "信用卡收到后怎么激活使用？"),
                         ("挂失手续费", "信用卡挂失手续费多少？"),
                         ("年费怎么收", "信用卡年费怎么收？怎么免年费？")]),
            ("费用查询", [("取现手续费与限额", "境外取现金手续费怎么收？有限额吗？"),
                         ("最低还款利息", "只还最低还款额有利息吗？"),
                         ("违约金", "最低还款没还要收违约金吗？")]),
            ("账单概念", [("免息期", "免息还款期最长多少天？"),
                         ("补对账单", "补制对账单收费吗？"),
                         ("有效期", "信用卡有效期多久？")]),
        ]
        for label, items in qgroups:
            cc = st.columns([1] + [1.5]*len(items) + [5])
            cc[0].markdown(f'<span class="g-label">{label}</span>', unsafe_allow_html=True)
            for j, (disp, real) in enumerate(items):
                if cc[j+1].button(disp, key=f"chip_{label}_{j}"):
                    st.session_state.pending_q = real
                    st.rerun()

    # 输入区：输入框 | 清空 | 红色发送（同一form同一行，回车发送+自动清空）
    with st.container(key="s-input"):
        with st.form(key="chat_form", clear_on_submit=True):
            ic = st.columns([12, 1.5, 1.7])
            q = ic[0].text_input("问题", placeholder="请输入您的信用卡问题，回车发送…", label_visibility="collapsed")
            clear_clicked = ic[1].form_submit_button("🗑 清空")
            send_clicked = ic[2].form_submit_button("发送", type="primary")
            if send_clicked and q:
                ask(q)
                st.rerun()
            if clear_clicked:
                st.session_state.chat_history = []
                st.rerun()

    # 免责声明（全宽白底）
    st.markdown('<div class="s-disclaimer">以上信息依据《领用合约》《信用卡章程》及收费价格表整理，仅供参考，具体以中信银行官方公告为准。</div>', unsafe_allow_html=True)

    # 右侧来源抽屉（fixed）
    if st.session_state.show_src:
        last_ctx = None
        for m in reversed(st.session_state.chat_history):
            if "ctx" in m:
                last_ctx = m["ctx"]
                break
        drawer = '<div class="src-drawer"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px"><b>📚 召回知识来源</b></div>'
        if last_ctx:
            for i, cc in enumerate(last_ctx):
                drawer += f'<div class="src-card"><span class="src-tag">资料{i+1} · {html.escape(str(cc.get("topic","")))}</span><div style="color:#8a909c;font-size:10px;margin-bottom:4px">来源：{html.escape(str(cc.get("source","")))}</div><div style="color:#444;line-height:1.7">{html.escape(str(cc["text"][:200]))}</div></div>'
        else:
            drawer += '<div style="color:#999;text-align:center;margin-top:60px;font-size:13px">暂无来源</div>'
        drawer += '</div>'
        st.markdown(drawer, unsafe_allow_html=True)




















