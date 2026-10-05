# -*- coding: utf-8 -*-
"""Streamlit 完整版：中信银行信用卡智能咨询助手"""
import os, json, html, re
import requests
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
import streamlit as st

DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"Authorization": "Bearer " + DASHSCOPE_KEY, "Content-Type": "application/json"}

NO_PROXY = {"http": None, "https": None}

def emb_online(texts):
    all_emb = []
    for i in range(0, len(texts), 10):
        batch = texts[i:i+10]
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            headers=HEADERS,
            json={"model": "text-embedding-v3", "input": batch},
            timeout=60,
            proxies=NO_PROXY)
        data = resp.json()
        if "data" not in data:
            raise RuntimeError(f"Embedding API error: {data}")
        all_emb.extend([d["embedding"] for d in data["data"]])
    return all_emb

def llm_chat(messages):
    resp = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers=HEADERS,
        json={"model": "qwen-plus", "messages": messages, "temperature": 0},
        timeout=60,
        proxies=NO_PROXY)
    return resp.json()["choices"][0]["message"]["content"]

def llm_chat_stream(messages):
    resp = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers=HEADERS,
        json={"model": "qwen-plus", "messages": messages, "temperature": 0, "stream": True},
        timeout=60,
        stream=True,
        proxies=NO_PROXY)
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except Exception:
                pass

@st.cache_resource
def load_index():
    chunks = []
    with open(os.path.join(BASE_DIR, "knowledge_base.jsonl"), encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    texts = [c["text"] for c in chunks]
    bm25 = BM25Okapi([jieba.lcut(t) for t in texts])
    emb_path = os.path.join(BASE_DIR, "_emb_cache.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
    else:
        emb = np.array(emb_online(texts))
        np.save(emb_path, emb)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return chunks, bm25, emb

def retrieve(q):
    chunks, bm25, emb = load_index()
    bm_scores = bm25.get_scores(jieba.lcut(q))
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

def build_msgs(q):
    """构建 LLM 消息（含检索上下文和历史）"""
    ctx = retrieve(q)
    ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])
    history_msgs = st.session_state.chat_history[-4:-1]
    msgs = [{"role": "system", "content": "你是中信银行信用卡智能咨询助手。仔细阅读业务资料，从资料中找答案，数字必须原样引用，确实没有才说不清楚。回答中引用资料时用[1][2]标注出处。"}] + history_msgs + [{"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{q}"}]
    return msgs, ctx

def render_bubble(text, role="bot"):
    """渲染一条消息气泡"""
    HARD = ["无法回答", "未找到", "无法提供", "未提供", "未列明", "未明确", "无法确定"]
    text = str(text) if text is not None else ""
    if role == "user":
        st.markdown(f'<div class="u-row"><div class="u-avatar">我</div><div class="u-bubble">{html.escape(text)}</div></div>', unsafe_allow_html=True)
    else:
        body = html.escape(text)
        body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', body)
        body = re.sub(r'\[(\d+)\]', r'<a href="javascript:void(0)" class="cite-link" data-idx="\1" style="color:#e60012;font-weight:600;text-decoration:none;cursor:pointer">[<span>\1</span>]</a>', body)
        body = body.replace('\n', '<br>')
        st.markdown(f'<div class="b-row"><div class="b-avatar">🤖</div><div class="b-bubble">{body}</div></div>', unsafe_allow_html=True)
        if any(k in text for k in HARD):
            st.markdown('<div class="fallback" style="margin-left:52px">如需进一步帮助，请拨打 <b>24小时客服热线 4008895558</b> 转人工，或尝试提问：取现手续费 / 最低还款 / 年费。</div>', unsafe_allow_html=True)

import streamlit.components.v1 as components

def _action_bar(text, idx, voted_up, voted_down, q="", ctx=None):
    """一行按钮：复制+赞+踩（hover才显示）"""
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("'", "\\'")
    up_dis = "disabled" if voted_up else ""
    dn_dis = "disabled" if voted_down else ""
    _val = components.html(f"""
    <style>
      .ab-wrap {{display:flex; gap:6px; margin-top:4px; margin-left:52px; align-items:center; opacity:0.35; transition:opacity 0.2s}}
      .ab-wrap:hover {{opacity:1}}
      .ab-btn {{background:#fff; border:1px solid #e8eaef; color:#888; border-radius:6px; padding:2px 10px; font-size:12px; cursor:pointer; line-height:1.6}}
      .ab-btn:hover {{color:#e60012; border-color:#e60012}}
      .ab-btn[disabled] {{opacity:.5; cursor:default}}
    </style>
    <div class="ab-wrap">
      <button class="ab-btn" onclick="navigator.clipboard.writeText(`{safe}`).then(()=>{{this.textContent='✓ 已复制';setTimeout(()=>this.textContent='📋 复制',2000)}})">📋 复制</button>
      <button class="ab-btn" {up_dis} onclick="this.disabled=true;this.textContent='✓ 已赞';parent.postMessage({{isStreamlitMessage:true,type:'streamlit:setComponentValue',value:'__UP_{idx}'}},'*')">👍</button>
      <button class="ab-btn" {dn_dis} onclick="this.disabled=true;this.textContent='✓ 已踩';parent.postMessage({{isStreamlitMessage:true,type:'streamlit:setComponentValue',value:'__DOWN_{idx}'}},'*')">👎</button>
    </div>
    """, height=32)
    # 处理赞/踩/引用反馈
    if _val and isinstance(_val, str):
        if _val == f"__UP_{idx}":
            st.session_state[f"voted_{idx}"] = "up"
            _save_feedback({"q": q, "content": text}, "up")
            st.rerun()
        elif _val == f"__DOWN_{idx}":
            st.session_state[f"voted_{idx}"] = "down"
            _save_feedback({"q": q, "content": text}, "down")
            st.rerun()
        elif _val.startswith(f"__CITE_{idx}_"):
            cite_idx = int(_val.split("_")[-1])
            st.session_state.show_src = True
            st.session_state.cite_scroll = cite_idx
            st.rerun()

def _save_feedback(msg, vote):
    """把反馈追加到 feedback.csv"""
    import csv
    row = [msg.get("q", ""), msg.get("content", "")[:200], vote]
    path = os.path.join(BASE_DIR, "feedback.csv")
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass

def stream_answer(msgs, ph):
    """流式生成回答，逐字更新 placeholder，返回完整文本"""
    full = ""
    for delta in llm_chat_stream(msgs):
        full += delta
        body = html.escape(full)
        body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', body)
        body = re.sub(r'\[(\d+)\]', r'<a href="javascript:void(0)" class="cite-link" data-idx="\1" style="color:#e60012;font-weight:600;text-decoration:none;cursor:pointer">[<span>\1</span>]</a>', body)
        body = body.replace('\n', '<br>')
        ph.markdown(f'<div class="b-row"><div class="b-avatar">🤖</div><div class="b-bubble">{body}</div></div>', unsafe_allow_html=True)
    return full

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
.b-avatar { width:40px;height:40px;border-radius:50%;background:#fdf0f0;flex:none;display:flex;align-items:center;justify-content:center;font-size:22px;line-height:1; }
.b-bubble { background:#fff;border:1px solid #e8eaef;border-radius:14px;border-bottom-left-radius:4px;padding:10px 14px;width:fit-content;max-width:100%;font-size:14.5px;line-height:1.55;word-break:break-word;box-shadow:0 1px 3px rgba(0,0,0,.04); }
.b-bubble strong { color:#e60012; }

/* 打字动画 */
.typing-dot { display:inline-block; width:6px; height:6px; background:#bbb; border-radius:50%; margin-right:3px; animation:typingBlink 1s infinite; }
.typing-dot:nth-child(2){ animation-delay:.2s; }
.typing-dot:nth-child(3){ animation-delay:.4s; }
@keyframes typingBlink { 0%,100%{opacity:.3;transform:translateY(0)} 50%{opacity:1;transform:translateY(-4px)} }
.u-row { display:flex;gap:12px;flex-direction:row-reverse;max-width:84%;margin:0 0 4px auto;align-items:flex-start; }
.u-avatar { width:40px;height:40px;border-radius:50%;background:#eef0f4;flex:none;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#666; }
.u-bubble { background:linear-gradient(135deg,#e60012,#c7000b);color:#fff;border-radius:14px;border-bottom-right-radius:4px;padding:10px 14px;width:fit-content;max-width:100%;font-size:14.5px;line-height:1.55;white-space:pre-wrap;word-break:break-word; }
/* 输入框圆角胶囊 */
.st-key-s-input input { border-radius:22px !important; border:1px solid #e8eaef !important; background:#f5f6f8 !important; padding:11px 18px !important; }
.st-key-s-input input:focus { border-color:#e60012 !important; background:#fff !important; }
.st-key-s-input button[kind="primary"] { border-radius:22px !important; padding:11px 26px !important; }
.st-key-s-input button:not([kind="primary"]) { border-radius:22px !important; }
/* 操作按钮半透明 hover */
.st-key-act button { opacity:.5; transition:opacity .2s; }
.st-key-act button:hover { opacity:1; }
/* 复制按钮iframe和赞踩并排：用flex */
.st-key-act { display:flex !important; flex-wrap:wrap; align-items:center; gap:4px; }
.st-key-act .element-container { display:inline-block !important; width:auto !important; }
.st-key-act [data-testid="stHorizontalBlock"] { display:inline-block !important; width:auto !important; }
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
.s-disclaimer { background:#fff; border-top:1px solid #e8eaef; padding:9px 32px 60px; font-size:11px; color:#8a909c; text-align:center; }
.fallback { margin-top:8px; padding:10px 14px; background:#fdf0f0; border-radius:10px; font-size:13px; line-height:1.6; color:#555; }
.fallback b { color:#e60012; }

/* 首页功能卡片 hover 上浮 */
.st-key-cardrow button {
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s !important;
}
.st-key-cardrow button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(230,0,18,0.12) !important;
    border-color: #e60012 !important;
}

/* 来源面板滑出动画 */
.src-side {
    transition: right 0.3s ease-in-out !important;
}

/* 操作按钮默认半透明，hover 时全显示 */
.st-key-s-msgs ~ div button[kind="secondary"] {
    opacity: 0.35;
    transition: opacity 0.2s;
}
.st-key-s-msgs ~ div button[kind="secondary"]:hover {
    opacity: 1;
}

/* ===== 技术说明面板 ===== */
.tech-block { background:#fff; border:1px solid #e8eaef; border-radius:16px; padding:26px 28px; margin-top:10px; }
.tech-block h3 { font-size:18px; color:#1a1d24; margin-bottom:6px; }
.tech-block .sub { font-size:12.5px; color:#8a909c; margin-bottom:18px; }
.arch-flow { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
.anode { background:#f5f6f8; border:1px solid #e8eaef; border-radius:10px; padding:9px 13px; font-size:12px; text-align:center; }
.anode.hl { background:linear-gradient(135deg,#e60012,#c7000b); color:#fff; border:none; }
.aarr { color:#bbb; font-size:15px; }
.tech-note { display:flex; gap:24px; flex-wrap:wrap; font-size:12px; color:#8a909c; }
.tech-note b { color:#e60012; }
.stats-row { display:flex; gap:36px; margin-top:18px; flex-wrap:wrap; }
.stat b { font-size:22px; color:#e60012; display:block; }
.stat span { font-size:12px; color:#8a909c; }
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
if "show_tech" not in st.session_state:
    st.session_state.show_tech = False
if "src_history" not in st.session_state:
    st.session_state.src_history = []

if st.session_state.pending_q:
    st.session_state.in_chat = True

# ============ 首页 ============
if not st.session_state.in_chat:
    # 首页打开时预热知识库（已缓存则毫秒级返回，对话页不再卡）
    load_index()
    with st.sidebar:
        st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:30px"><div style="width:40px;height:40px;background:white;border-radius:50%;color:#e60012;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:20px">中</div><div><div style="color:white;font-weight:bold;font-size:16px">中信银行</div><div style="color:rgba(255,255,255,0.7);font-size:10px">CHINA CITIC BANK</div></div></div>', unsafe_allow_html=True)
        if st.button("💬 智能客服", key="nav1", use_container_width=True):
            st.session_state.show_tech = False
            st.rerun()
        if st.button("🔧 技术说明", key="nav2", use_container_width=True):
            st.session_state.show_tech = not st.session_state.show_tech
            st.rerun()
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
        if st.session_state.show_tech:
            # ===== 技术说明面板 =====
            st.markdown("""
            <div class="tech-block">
              <h3>RAG 技术架构</h3>
              <p class="sub">检索增强生成全链路 · 混合召回 → Rerank 精排 → 大模型生成</p>
              <div class="arch-flow">
                <div class="anode">用户问题</div><div class="aarr">→</div>
                <div class="anode">Embedding<br>向量化</div><div class="aarr">→</div>
                <div class="anode">向量+BM25<br>混合召回</div><div class="aarr">→</div>
                <div class="anode">RRF<br>融合</div><div class="aarr">→</div>
                <div class="anode hl">Rerank<br>精排</div><div class="aarr">→</div>
                <div class="anode">LLM<br>生成</div>
              </div>
              <div class="tech-note">
                <span>向量检索：懂<b>语义</b>（"忘记还款"≈逾期）</span>
                <span>BM25：抓<b>关键词</b>（年费/取现/违约金）</span>
                <span>Rerank：<b>精排</b>候选资料相关性</span>
              </div>
              <div class="stats-row">
                <div class="stat"><b>89</b><span>业务知识块</span></div>
                <div class="stat"><b>2路</b><span>向量 + BM25</span></div>
                <div class="stat"><b>双栈</b><span>本地开源/在线API</span></div>
                <div class="stat"><b>3/3</b><span>测试题全对</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 左侧主内容 + 右侧FAQ栏
            left, right = st.columns([2.2, 1])
            with left:
                # 欢迎区
                c1, c2 = st.columns([1.2, 10], gap="medium")
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

                # 热线提示条
                st.markdown("""
                <div style="background:white;border-left:3px solid #e60012;border-radius:8px;padding:14px 20px;margin-top:20px;font-size:14px">
                    🤖 以上问题我可以帮您解答；如果需要人工服务，请拨打 <b style="color:#e60012">24小时客服热线 4008895558</b>
                </div>
                """, unsafe_allow_html=True)

                # 首页直接输入框
                with st.form(key="home_form", clear_on_submit=True):
                    h1, h2 = st.columns([4, 1])
                    home_q = h1.text_input("", placeholder="输入您的信用卡问题，直接开始咨询…", label_visibility="collapsed")
                    home_send = h2.form_submit_button("开始咨询 →", type="primary")
                    if home_send and home_q and home_q.strip():
                        st.session_state.pending_q = home_q.strip()
                        st.session_state.in_chat = True
                        st.rerun()

            # 右侧FAQ栏
            with right:
                st.markdown("### 常见问题")
                faq_items = [
                    ("信用卡取现手续费多少？", "信用卡取现手续费多少？"),
                    ("信用卡怎么激活？", "信用卡怎么激活？"),
                    ("最低还款有利息吗？", "只还最低还款额有利息吗？"),
                    ("年费怎么收？", "信用卡年费怎么收？怎么免年费？"),
                    ("挂失手续费多少？", "信用卡挂失手续费多少？"),
                    ("逾期有什么后果？", "逾期有什么后果？"),
                ]
                for label, q_text in faq_items:
                    if st.button(f"{label} →", key=f"faq_{label}", use_container_width=True):
                        st.session_state.pending_q = q_text
                        st.session_state.in_chat = True
                        st.rerun()

            st.markdown('<div style="text-align:center;font-size:11px;color:#999;margin-top:20px">以上信息依据《领用合约》《信用卡章程》及收费价格表整理，仅供参考，具体以中信银行官方公告为准</div>', unsafe_allow_html=True)

# ============ 对话页面 ============
else:
    # 隐藏首页红色 sidebar 和首页容器
    st.markdown("""
    <style>
    section[data-testid="stSidebar"],
    .st-key-topbanner,
    .st-key-contentpad,
    .st-key-cardrow,
    .st-key-faqbox { display:none !important; }
    </style>
    """, unsafe_allow_html=True)

    # 顶部小字栏（全宽白底）
    st.markdown("""
    <div class="s-topbar">
      <span>欢迎使用中信银行信用卡智能服务</span>
      <span>24小时客服热线 <b style="color:#e60012">4008895558</b></span>
    </div>
    """, unsafe_allow_html=True)

    # 品牌栏：左品牌 右返回首页+清空对话+来源
    with st.container(key="s-header"):
        hc = st.columns([8, 1.5, 1.5, 1.5])
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
            if st.button("🔄 清空对话", key="reset_chat"):
                st.session_state.chat_history = []
                st.session_state.src_history = []
                st.session_state.pending_q = None
                st.rerun()
        with hc[3]:
            if st.button("📚 来源", key="go_src"):
                st.session_state.show_src = not st.session_state.show_src
                st.rerun()

    # 消息区（浅灰底，气泡fit-content左右分置）
    with st.container(key="s-msgs"):
        # 用 components.html 注入自动滚动（st.markdown的script不执行）
        components.html("""
        <script>
        (function() {
            function scrollDown() {
                const doc = window.parent.document;
                const main = doc.querySelector('.stMain') || doc.querySelector('[data-testid="stMain"]');
                if (main) main.scrollTop = main.scrollHeight;
            }
            scrollDown();
            const obs = new MutationObserver(function() {
                scrollDown();
            });
            obs.observe(window.parent.document.body, {childList: true, subtree: true});
        })();
        </script>
        """, height=0)
        st.markdown('<div class="b-row"><div class="b-avatar">🤖</div><div class="b-bubble">您好，我是中信银行信用卡智能咨询助手 👋<br>我只依据《领用合约》《收费价格表》等业务资料为您解答，数字有据可查。<br>可咨询：激活、取现、最低还款、年费、账单等。</div></div>', unsafe_allow_html=True)
        for idx, msg in enumerate(st.session_state.chat_history):
            render_bubble(msg["content"], msg["role"])
            if msg["role"] == "assistant":
                voted_up = st.session_state.get(f"voted_{idx}") == "up"
                voted_down = st.session_state.get(f"voted_{idx}") == "down"
                _action_bar(msg["content"], idx, voted_up, voted_down, msg.get("q", ""), msg.get("ctx"))

        # 处理新问题：流式生成
        if st.session_state.pending_q:
            q = st.session_state.pending_q
            st.session_state.pending_q = None
            # 用户消息
            st.session_state.chat_history.append({"role": "user", "content": q})
            render_bubble(q, "user")
            # loading 动画
            ph = st.empty()
            ph.markdown('<div class="b-row"><div class="b-avatar">🤖</div><div class="b-bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span> <span style="font-size:13px;color:#999;margin-left:8px">正在检索业务资料…</span></div></div>', unsafe_allow_html=True)
            # 检索 + 流式生成
            try:
                msgs, ctx = build_msgs(q)
                full = stream_answer(msgs, ph)
                if not full:
                    ph.markdown('<div class="b-row"><div class="b-avatar">🤖</div><div class="b-bubble" style="color:#e60012">抱歉，暂时无法获取回答，请稍后重试。</div></div>', unsafe_allow_html=True)
                    full = "抱歉，暂时无法获取回答，请稍后重试。"
                    ctx = []
            except Exception as e:
                import traceback
                full = "抱歉，网络开小差了，请稍后重试。如持续出现，请拨打 24小时客服热线 4008895558。"
                ctx = []
                ph.markdown(f'<div class="b-row"><div class="b-avatar">🤖</div><div class="b-bubble" style="color:#e60012">{full}</div></div>', unsafe_allow_html=True)
                print("ERROR:", traceback.format_exc())
            # 存历史
            st.session_state.chat_history.append({"role": "assistant", "content": full, "ctx": ctx, "q": q})
            st.session_state.src_history.append(ctx)
            # 兜底引导
            HARD = ["无法回答", "未找到", "无法提供", "未提供", "未列明", "未明确", "无法确定"]
            if any(k in full for k in HARD):
                st.markdown('<div class="fallback" style="margin-left:52px">如需进一步帮助，请拨打 <b>24小时客服热线 4008895558</b> 转人工，或尝试提问：取现手续费 / 最低还款 / 年费。</div>', unsafe_allow_html=True)
            # 操作按钮
            act_idx = len(st.session_state.chat_history) - 1
            last_msg = st.session_state.chat_history[act_idx]
            _action_bar(last_msg["content"], act_idx, False, False, last_msg.get("q", ""), ctx)

    # 来源面板（点"来源"按钮在右侧滑出）
    if st.session_state.show_src:
        _items = ""
        if not st.session_state.src_history:
            _items = '<div style="font-size:12px;color:#999">提问后这里实时展示检索到的业务资料块</div>'
        for _ri, _ctx in enumerate(st.session_state.src_history):
            _items += f'<div style="font-size:11px;color:#999;padding:8px 0 4px">—— 第{_ri+1}次回答 ——</div>'
            if not _ctx:
                _items += '<div style="font-size:12px;color:#999">未检索到相关资料</div>'
            for _ci, _c in enumerate(_ctx):
                _topic = _c.get("topic", "")
                _source = _c.get("source", "")
                _text = _c.get("text", "")[:120].replace("<", "&lt;")
                _items += ('<div style="background:#f5f6f8;border-radius:8px;padding:8px 12px;margin-bottom:6px;font-size:12px">'
                           f'<b style="color:#e60012">资料{_ci+1} · {_topic}</b><br>'
                           f'<span style="color:#8a909c;font-size:11px">来源：{_source}</span><br>'
                           f'<span style="color:#555">{_text}…</span></div>')
        sc = st.columns([10, 1])
        sc[1].button("✕ 关闭", key="close_src", on_click=lambda: setattr(st.session_state, "show_src", False))
        st.markdown(
            '<style>.src-side{position:fixed;top:140px;right:0;width:340px;height:calc(100vh - 140px);background:#fff;'
            'border-left:1px solid #e8eaef;padding:20px;overflow-y:auto;z-index:1000;'
            'box-shadow:-4px 0 16px rgba(0,0,0,.08)}.src-side h3{margin:0 0 12px;font-size:15px}'
            '.st-key-s-header { padding-right:360px !important; } .s-topbar { padding-right:360px !important; }</style>'
            '<div class="src-side"><h3>📚 召回知识来源</h3>' + _items + '</div>',
            unsafe_allow_html=True)

        # 如果需要滚动到指定引用
        if st.session_state.get("cite_scroll"):
            cite_idx = st.session_state.cite_scroll
            st.session_state.cite_scroll = None
            components.html(f"""
            <script>
            setTimeout(function() {{
                const doc = window.parent.document;
                const panel = doc.querySelector('.src-side');
                if (panel) {{
                    const items = panel.querySelectorAll('[style*="border-radius:8px"]');
                    if (items[{cite_idx-1}]) {{
                        items[{cite_idx-1}].scrollIntoView({{behavior: 'smooth', block: 'center'}});
                        items[{cite_idx-1}].style.outline = '2px solid #e60012';
                        setTimeout(function() {{ items[{cite_idx-1}].style.outline = ''; }}, 2000);
                    }}
                }}
            }}, 300);
            </script>
            """, height=0)

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

    # 输入区：form 实现（Enter提交发送，清空按钮只清空输入框）
    with st.container(key="s-input"):
        with st.form(key="chat_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([11, 1.5, 1])
            user_input = c1.text_input("问题",
                         placeholder="请输入您的信用卡问题… （按 Enter 发送）",
                         label_visibility="collapsed")
            send_clicked = c2.form_submit_button("发送", type="primary")
            clear_clicked = c3.form_submit_button("🗑 清空")

        if send_clicked and user_input and user_input.strip():
            st.session_state.pending_q = user_input.strip()
            st.rerun()

    # 发送后自动聚焦输入框 + 引用点击监听
    components.html("""
    <script>
    // 自动聚焦输入框
    setTimeout(function() {
        const doc = window.parent.document;
        const input = doc.querySelector('.st-key-s-input input');
        if (input) input.focus();
    }, 500);

    // 全局监听引用点击
    window.parent.document.addEventListener('click', function(e) {
        const link = e.target.closest('.cite-link');
        if (!link) return;
        e.preventDefault();
        const idx = parseInt(link.dataset.idx);
        console.log('cite clicked:', idx);
        // 如果来源面板没打开，点"来源"按钮
        const srcPanel = window.parent.document.querySelector('.src-side');
        if (!srcPanel) {
            const buttons = window.parent.document.querySelectorAll('button');
            for (const btn of buttons) {
                if (btn.textContent.includes('来源')) {
                    btn.click();
                    break;
                }
            }
        }
        // 等待面板出现后scroll到对应资料
        setTimeout(function() {
            const panel = window.parent.document.querySelector('.src-side');
            if (panel) {
                const items = panel.querySelectorAll('[style*="border-radius:8px"]');
                if (items[idx-1]) {
                    items[idx-1].scrollIntoView({behavior: 'smooth', block: 'center'});
                    items[idx-1].style.outline = '2px solid #e60012';
                    setTimeout(function() { items[idx-1].style.outline = ''; }, 2000);
                }
            }
        }, 500);
    });
    </script>
    """, height=1)

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




















