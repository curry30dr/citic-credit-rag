# 中信银行信用卡智能咨询助手（RAG 双栈实践）

基于 RAG（检索增强生成）的信用卡智能客服系统。同一套知识库、同一套检索链路，
用 **本地开源** 与 **在线 API** 两种栈分别实现，便于对比技术选型与成本/隐私权衡。

---

## 一、快速开始

```bash
# 在线栈（text-embedding-v3 + gte-rerank-v2 + qwen-plus）
python rag_assistant.py --mode online "信用卡取现手续费怎么收？"

# 本地栈（bge-small-zh + bge-reranker-base）
python rag_assistant.py --mode local "信用卡怎么激活？"

# 两栈对比同一个问题
python rag_assistant.py --compare "还最低还款额有利息吗？"

# 交互模式
python rag_assistant.py --mode online

# 跑消融实验（20 题 × 4 配置，输出 Recall@4）
python ablation.py

# 启动本地网页演示（Flask，中信红品牌风）
python app.py
# 浏览器打开 http://127.0.0.1:5000
```

依赖：`pip install -r requirements.txt`
（本地模型首次运行会自动从 ModelScope 下载到 `~/.cache/modelscope`）
在线栈需要设置环境变量 `DASHSCOPE_API_KEY`。

## 二、文件说明

| 文件 | 说明 |
|---|---|
| `rag_assistant.py` | 主程序：双模式检索 + 生成，支持 local/online/compare/interactive |
| `ablation.py` | 消融实验：20 题 × 4 配置，输出 Recall@4 |
| `app.py` | Flask 后端：首页 + 对话页 + `/api/ask` 接口 |
| `app_hf.py` | Gradio 部署版（本地模型路径优先） |
| `knowledge_base.jsonl` | 知识库，89 个知识块（含来源、业务主题元数据） |
| `faiss.index` | 本地 bge-small-zh 向量索引（512 维） |
| `faiss_online.index` | 在线 text-embedding-v3 向量索引（1024 维） |
| `bm25.pkl` | BM25 索引 + 知识块 |
| `static/index.html` | 项目首页（中信红品牌风） |
| `static/chat.html` | 对话演示页（折叠侧边栏/清空/复制/赞踩/分组快捷问题/免责声明） |
| `requirements.txt` | Python 依赖 |
| `答辩PPT大纲.md` | 10 分钟答辩大纲 + 预判问答 |

## 三、系统架构

```
用户问题
   ├── Embedding 向量化 ──→ 向量召回(Top10)
   └── BM25 关键词召回(Top10)
            ↓
       RRF 倒数排名融合 (k=60)
            ↓
       Rerank 精排 → Top4
            ↓
       qwen-plus 生成（仅依据检索资料，防幻觉）
            ↓
         智能咨询回答 + 来源溯源
```

## 四、技术选型（双栈对比）

| 模块 | 本地开源栈 | 在线 API 栈 | 选择理由 |
|---|---|---|---|
| Embedding | BAAI/bge-small-zh-v1.5 | text-embedding-v3 | 均为中文优化；本地可离线、免费；在线语义更强、免维护 |
| 向量库 | FAISS（512/1024 维） | FAISS | 两栈共用同一召回框架 |
| 关键词 | rank_bm25 | rank_bm25 | 抓"年费""取现"等专有名词，补向量短板 |
| 融合 | RRF (k=60) | RRF (k=60) | 两路分数量纲不同，倒数排名融合无需调权、鲁棒 |
| Rerank | bge-reranker-base | gte-rerank-v2 | 粗排只求"可能相关"，精排逐对判断 query-doc 相关性 |
| 生成 | qwen-plus | qwen-plus | 两栈统一，保证对比公平 |

## 五、数据处理

- 原始资料：领用合约+章程（约 2 万字）、激活/取现/账单、循环信用指南、价格表。
- **切片策略**：chunk ≈ 500 字、overlap ≈ 80 字，按句子边界切，不切碎业务规则；
  价格表按"每个费用项独立成块"，避免多费用项混在一块导致召回歧义。
- 结果：89 个知识块，平均 281 字/块。
- 踩过的坑：python-docx 把段落和表格分开遍历，导致价格表错位、合并单元格重复 4 次稀释关键词；
  已改为按文档真实顺序交错提取段落与表格行。

## 六、消融实验结果

自构 20 道真实业务问题（取现/账单/年费/激活/还款等），对比四种检索配置在 Top-4 的命中率：

| 检索配置 | 命中题数 | Recall@4 |
|---|---|---|
| 纯向量 | 17 / 20 | 85% |
| 纯 BM25 | 18 / 20 | 90% |
| 混合 RRF | 18 / 20 | 90% |
| **混合 + Rerank** | **19 / 20** | **95%** |

结论：BM25 在术语匹配上略强于向量；RRF 融合稳定不跌点；**Rerank 是效果跃升的关键**。

## 七、防幻觉设计

- Prompt 约束：数字必须原样引用资料，资料没有就明说并引导至客服 4008895558；
- `temperature=0.2` 降低发散；
- 回答附"召回来源"折叠区，可展开看依据的是哪份文档；
- 实测裸模型 qwen 会把日利率说成 0.035%–0.05%，RAG 后回到文档原文 0%–0.05%。

## 八、部署

- 本地：`python app.py` → http://127.0.0.1:5000
- 公网：代码推到 GitHub，接入 Streamlit Community Cloud，`DASHSCOPE_API_KEY` 以 Secret 方式托管，
  不硬编码密钥。
- 前端：中信红品牌风，右侧"召回来源"默认折叠，快捷问题按类分组，底部免责声明。
