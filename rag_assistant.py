# -*- coding: utf-8 -*-
"""
中信银行信用卡智能咨询助手（RAG 双栈版）
=================================================
用法：
  python rag_assistant.py --mode online "信用卡取现手续费怎么收？"
  python rag_assistant.py --mode local                # 交互模式
  python rag_assistant.py --mode online --compare "..."  # 两栈对比

架构：
  用户问题 -> Embedding(向量化) + BM25(关键词)
          -> 向量召回 + BM25召回 -> RRF 融合 -> Rerank 重排 -> Top-K
          -> qwen-plus 生成回答（只依据检索到的资料，防幻觉）

两种模式：
  local : BAAI/bge-small-zh-v1.5 (512维) + bge-reranker-base   【本地开源】
  online: 阿里 text-embedding-v3 (1024维) + gte-rerank-v2       【在线API】
生成模型统一用 qwen-plus，保证两栈对比公平。
"""
import os, json, argparse, pickle, numpy as np, faiss, urllib.request

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 从环境变量或 .env 文件读取 API Key，不要写在代码里
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_KEY", "")
if not DASHSCOPE_KEY:
    _env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(_env_path):
        for _line in open(_env_path, encoding="utf-8"):
            if "=" in _line and not _line.strip().startswith("#"):
                _k, _v = _line.strip().split("=", 1)
                if _k.strip() == "DASHSCOPE_KEY":
                    DASHSCOPE_KEY = _v.strip()
                    break
EMB_LOCAL = r"D:\project\models\bge-small-zh-v1.5"
RERANK_LOCAL = r"D:\project\models\bge-reranker-base"
LLM_MODEL = "qwen-plus"
TOP_N = 5          # 最终喂给 LLM 的资料条数
RECALL_K = 10      # 每路召回条数

# ============ 加载知识库与索引 ============
def load_index():
    data = pickle.load(open(os.path.join(BASE_DIR, "bm25.pkl"), "rb"))
    chunks, bm25 = data["chunks"], data["bm25"]
    faiss_local = faiss.read_index(os.path.join(BASE_DIR, "faiss.index"))
    faiss_online = faiss.read_index(os.path.join(BASE_DIR, "faiss_online.index"))
    return chunks, bm25, faiss_local, faiss_online

CHUNKS, BM25, FAISS_LOCAL, FAISS_ONLINE = load_index()

# ============ Embedding ============
_bge = None
def emb_local(texts):
    global _bge
    if _bge is None:
        from sentence_transformers import SentenceTransformer
        _bge = SentenceTransformer(EMB_LOCAL)
    # bge 中文检索建议给 query 加指令；这里统一对查询处理，文档已入库
    return _bge.encode(texts, normalize_embeddings=True).astype("float32")

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

# ============ Rerank ============
_rr = None
def rerank_local(query, pairs):
    """pairs: [[query, doc], ...] -> 分数列表"""
    global _rr
    if _rr is None:
        from sentence_transformers import CrossEncoder
        _rr = CrossEncoder(RERANK_LOCAL)
    return [float(s) for s in _rr.predict(pairs)]

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
def _tok(s):
    return list(s.replace(" ", ""))

def retrieve(query, mode="online"):
    # 1) 向量召回
    if mode == "local":
        qv = emb_local(["为这个句子生成表示以用于检索相关文章：" + query])
        D, I = FAISS_LOCAL.search(qv, RECALL_K)
    else:
        qv = emb_online([query])
        D, I = FAISS_ONLINE.search(qv, RECALL_K)
    vec_rank = {int(i): r + 1 for r, i in enumerate(I[0])}

    # 2) BM25 关键词召回
    scores = BM25.get_scores(_tok(query))
    top_idx = np.argsort(scores)[::-1][:RECALL_K]
    bm_rank = {int(i): r + 1 for r, i in enumerate(top_idx)}

    # 3) RRF 融合
    cands = set(vec_rank) | set(bm_rank)
    rrf = {}
    for c in cands:
        s = 1 / (60 + vec_rank[c]) if c in vec_rank else 0
        if c in bm_rank:
            s += 1 / (60 + bm_rank[c])
        rrf[c] = s
    fused = sorted(rrf, key=lambda c: -rrf[c])[:TOP_N * 2]

    # 4) Rerank + 阈值过滤
    RERANK_THRESHOLD = 0.20
    if mode == "local":
        pairs = [[query, CHUNKS[c]["text"]] for c in fused]
        sc = rerank_local(query, pairs)
        # 按分数排序，过滤低分
        scored = sorted(zip(fused, sc), key=lambda x: -x[1])
        picked = [c for c, s in scored if s >= RERANK_THRESHOLD][:TOP_N]
    else:
        docs = [CHUNKS[c]["text"] for c in fused]
        rr = rerank_online(query, docs, TOP_N * 2)
        picked = [fused[i] for i, s in rr if s >= RERANK_THRESHOLD][:TOP_N]
    return [CHUNKS[c] for c in picked]

# ============ LLM 生成 ============
SYSTEM_PROMPT = (
    "你是中信银行信用卡智能咨询助手。请只根据下面提供的【业务资料】回答用户问题。\n"
    "要求：\n"
    "1.金额、利率、费用、期限等数字必须原样引用资料，不得编造；\n"
    "2.客服热线统一写4008895558，不得加横杠、空格或换行；\n"
    "3.不得提及资料中未出现的卡种名称；\n"
    "4.先一句话给出结论，再用2-3点补充例外情况，总长度不超过300字，不复述资料原文；\n"
    "5.若资料中没有相关信息，明确说'根据现有资料无法回答，建议拨打客服热线4008895558'；\n"
    "6.涉及返现/优惠比例时，必须明确区分两档：普通微信/支付宝/云闪付支付的比例，和指定商户/指定平台的比例，不得混为一谈；\n"
    "7.计算返现时，每张卡独立计算月上限和年上限，不跨卡共享额度；公式：单卡返现=min(消费金额×返现比例, 该卡月上限)。例如消费600元返现15%月上限100元，则返min(90,100)=90元（不是100元）；最后把各卡返现相加。\n"
    "8.直接给出最终答案，不要展示任何计算过程、犹豫、自我纠正或矛盾推理，不要写'重新核对''最终答案''但需注意''不对'这类话。只输出最终结果，客户看到的是结论不是推理过程。\n"
    "9.用户说'分别选择适合这三类场景的卡'=三张卡各刷各的（爱吃版刷吃饭、爱行版刷打车、爱家版刷缴费），每张卡独立计算返现后相加，不是一张卡刷三笔。"
)

def answer(query, mode="online", history=None):
    ctx = retrieve(query, mode)
    ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{query}"})
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 800, "temperature": 0}).encode()
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body, headers={"Authorization": "Bearer " + DASHSCOPE_KEY,
                            "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.loads(r.read().decode())["choices"][0]["message"]["content"]
    return resp, ctx

def show(query, mode):
    print(f"\n{'='*60}\n[模式:{mode}] Q: {query}")
    resp, ctx = answer(query, mode)
    print(f"{'-'*60}\nA: {resp}")
    print(f"{'-'*60}\n召回来源:")
    for i, c in enumerate(ctx):
        print(f"  {i+1}. [{c['topic']}] {c['text'][:70]}...")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "online"], default="online")
    ap.add_argument("--compare", action="store_true", help="两栈对比")
    ap.add_argument("question", nargs="?", help="问题；不填则进入交互模式")
    args = ap.parse_args()

    if args.question:
        if args.compare:
            show(args.question, "local")
            show(args.question, "online")
        else:
            show(args.question, args.mode)
    else:
        print("中信银行信用卡智能咨询助手（输入 quit 退出）")
        while True:
            q = input("\n请输入问题> ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                show(q, args.mode)

