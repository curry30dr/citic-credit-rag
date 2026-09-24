# -*- coding: utf-8 -*-
"""网页演示后端：python app.py 后访问 http://127.0.0.1:5000"""
import os, json, csv, datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask import request as flask_request
import rag_assistant as R

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["SECRET_KEY"] = "citic-rag-demo"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")
LOG_FILE = os.path.join(BASE_DIR, "qa_log.jsonl")

# 每个用户独立对话历史，key=sid
HISTORIES = {}
MAX_HISTORY = 6

# 全局缓存：相同问题0秒返回
CACHE = {}

def get_history(sid):
    if sid not in HISTORIES:
        HISTORIES[sid] = []
    return HISTORIES[sid]

def log_qa(q, mode, sources, answer, cached):
    rec = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": q,
        "mode": mode,
        "cached": cached,
        "recalled": [{"topic": s["topic"], "source": s["source"]} for s in sources],
        "answer_len": len(answer),
        "answer_preview": answer[:200]
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/chat")
def chat():
    return send_from_directory("static", "chat.html")

@socketio.on("ask")
def handle_ask(data):
    from flask_socketio import request as socket_request
    sid = socket_request.sid
    q = (data.get("question") or "").strip()
    mode = data.get("mode", "online")
    if not q:
        emit("error", {"text": "问题不能为空"})
        return

    history = get_history(sid)

    # 先检索
    ctx = R.retrieve(q, mode)
    sources = [{"topic": c["topic"], "source": c["source"], "text": c["text"]} for c in ctx]
    ctx_text = "\n\n".join([f"[资料{i+1}] {c['text']}" for i, c in enumerate(ctx)])

    messages = [{"role": "system", "content": R.SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": f"【业务资料】\n{ctx_text}\n\n【用户问题】{q}"})

    body_json = json.dumps({
        "model": R.LLM_MODEL,
        "messages": messages,
        "max_tokens": 800, "temperature": 0,
        "stream": True
    }).encode()

    req = __import__("urllib.request").request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body_json,
        headers={"Authorization": "Bearer " + R.DASHSCOPE_KEY,
                 "Content-Type": "application/json"})

    # 先把sources推给当前用户
    emit("sources", {"sources": sources})
    full = ""
    try:
        with __import__("urllib.request").request.urlopen(req, timeout=90) as r:
            for line in r:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        full += delta
                        emit("token", {"text": delta})
                except:
                    pass
    except Exception as e:
        emit("error", {"text": str(e)})
        return

    # 更新当前用户历史和日志
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": full})
    if len(history) > MAX_HISTORY:
        del history[:len(history) - MAX_HISTORY]
    log_qa(q, mode, sources, full, False)
    emit("done")

@socketio.on("clear")
def handle_clear():
    from flask_socketio import request as socket_request
    sid = socket_request.sid
    if sid in HISTORIES:
        HISTORIES[sid].clear()
    emit("cleared")

@app.route("/api/feedback", methods=["POST"])
def feedback():
    body = request.get_json(force=True)
    q = (body.get("question") or "").strip()
    a = (body.get("answer") or "").strip()
    vote = body.get("vote")
    if not q or not vote:
        return jsonify({"error": "参数不完整"}), 400
    row = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), q, vote, a[:500]]
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["时间", "问题", "评价", "回答摘要"])
        w.writerow(row)
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("=" * 50)
    print(" 中信银行信用卡智能咨询助手 (WebSocket多用户版)")
    print(" 打开浏览器访问: http://127.0.0.1:5000")
    print("=" * 50)
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)
