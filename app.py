# -*- coding: utf-8 -*-
"""
Hugging Face Spaces 部署版：中信银行信用卡智能咨询助手
- 检索：本地 bge-small-zh-v1.5 向量 + BM25 + RRF + bge-reranker-base
- 生成：阿里百炼 qwen-plus（Key 从环境变量 DASHSCOPE_API_KEY 读）
"""
import os, json, numpy as np, faiss, urllib.request
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import gradio as gr

# ============ 配置 ============
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY",
    "sk-ws-H.PHEPREI.wzWE.MEQCIFbO9FpQzB_5mbvsyhs6-bJeEIh83CP8bqgZyU95NLJVAiBwKTHfLrH8Ip-FVyg-RECAUHnwKIIeWwuR03veHJYSMA")
_LOCAL_EMB = r"D:\project\models\bge-small-zh-v1.5"
_LOCAL_RR = r"D:\project\models\bge-reranker-base"
EMB_MODEL = _LOCAL_EMB if os.path.isdir(_LOCAL_EMB) else "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = _LOCAL_RR if os.path.isdir(_LOCAL_RR) else "BAAI/bge-reranker-base"
LLM_MODEL = "qwen-plus"
TOP_N = 4
RECALL_K = 10

# ============ 加载知识库 ============
here = os.path.dirname(os.path.abspath(__file__))
CHUNKS = [json.loads(l) for l in open(os.path.join(here, "knowledge_base.jsonl"), encoding="utf-8")]
texts = [c["text"] for c in CHUNKS]

# ============ 建索引（启动时一次）============
print("加载 embedding 模型...")
emb = SentenceTransformer(EMB_MODEL)
vecs = emb.encode(["为这个句子生成表示以用于检索相关文章：" + t for t in texts],
                  normalize_embeddings=True).astype("float32")
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)

print("建 BM25 索引...")
def tok(s): return list(s.replace(" ", ""))
BM25 = BM25Okapi([tok(t) for t in texts])

print("加载 reranker...")
reranker = CrossEncoder(RERANK_MODEL)

# ============ 检索 ============
def retrieve(query):
    qv = emb.encode(["为这个句子生成表示以用于检索相关文章：" + query],
                    normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, RECALL_K)
    vr = {int(i): r+1 for r, i in enumerate(I[0])}

    sc = BM25.get_scores(tok(query))
    ti = np.argsort(sc)[::-1][:RECALL_K]
    br = {int(i): r+1 for r, i in enumerate(ti)}

    cands = set(vr) | set(br); rrf = {}
    for c in cands:
        s = 1/(60+vr[c]) if c in vr else 0
        if c in br: s += 1/(60+br[c])
        rrf[c] = s
    fused = sorted(rrf, key=lambda c: -rrf[c])[:TOP_N*2]

    pairs = [[query, CHUNKS[c]["text"]] for c in fused]
    rsc = reranker.predict(pairs)
    order = np.argsort(rsc)[::-1][:TOP_N]
    return [CHUNKS[fused[i]] for i in order]

# ============ 生成 ============
SYS = ("你是中信银行信用卡智能咨询助手。请只根据下面提供的【业务资料】回答用户问题。"
       "要求：1.金额、利率、费用、期限等数字必须原样引用资料，不得编造；"
       "2.若资料中没有相关信息，明确说“根据现有资料无法回答，建议拨打客服热线4008895558”；"
       "3.回答简洁清晰，分点说明。")

def answer(query):
    if not DASHSCOPE_KEY:
        return "⚠️ 未配置 DASHSCOPE_API_KEY，请在 Space 的 Secrets 里设置。", []
    ctx = retrieve(query)
    ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{query}"}
        ], "max_tokens": 700, "temperature": 0.2}).encode()
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body, headers={"Authorization": "Bearer " + DASHSCOPE_KEY,
                            "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.loads(r.read().decode())["choices"][0]["message"]["content"]
    return resp, ctx

# ============ Gradio 界面 ============
def chat(q, history):
    if not q.strip(): return "", history
    ans, ctx = answer(q)
    src = "\n\n".join([f"【资料{i+1}·{c['topic']}】{c['text'][:150]}…" for i, c in enumerate(ctx)])
    history = history or []
    history.append((q, ans + "\n\n📚 召回来源：\n" + src))
    return "", history

css = """
.gradio-container{max-width:900px!important}
#title{color:#e60012;font-weight:700}
"""
with gr.Blocks(title="中信银行信用卡智能咨询助手", css=css) as demo:
    gr.Markdown("# 🇨🇳 中信银行信用卡智能咨询助手\n基于 RAG（向量+BM25混合检索 / Rerank精排）· 回答可溯源 · 客服热线 40088-95558")
    gr.ChatInterface(chat, examples=[
        "境外取现金手续费怎么收？有限额吗？",
        "只还最低还款额有没有利息？",
        "信用卡普卡年费多少钱？",
    ])

if __name__ == "__main__":
    # share=True 会自动生成 xxx.gradio.live 公网链接（有效期72小时）
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                share=True)
