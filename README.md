# Financial Text Sentiment Analysis with Explainable AI
# 面向金融市场的可解释文本情感分析

> **Explainable financial sentiment analysis pipeline** — combining FinBERT fine-tuning, attention/LIME/SHAP explainability, LLM chain-of-thought reasoning, and sentiment-vs-financials correlation analysis.
>
> **可解释的金融情感分析流水线** —— 融合 FinBERT 微调、注意力/LIME/SHAP 可解释性方法、大语言模型思维链推理，以及「市场情感 × 财务指标」相关性分析。

![Project](https://img.shields.io/badge/Project%20ID-%2360-blue)
![Domain](https://img.shields.io/badge/Domain-Financial%20NLP-green)
![Models](https://img.shields.io/badge/Models-FinBERT%20%7C%20GPT--4o%20%7C%20Phi--2-orange)
![Explainability](https://img.shields.io/badge/XAI-Attention%20%7C%20LIME%20%7C%20SHAP-purple)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

---

## 📑 Table of Contents / 目录

- [Overview / 项目概述](#overview--项目概述)
- [Motivation / 研究动机](#motivation--研究动机)
- [Pipeline Architecture / 流水线架构](#pipeline-architecture--流水线架构)
- [1. Data Curation / 数据整理](#1-data-curation--数据整理)
- [2. FinBERT & Explainability / FinBERT 与可解释性](#2-finbert--explainability--finbert-与可解释性)
- [3. LLM Reasoning / 大语言模型推理](#3-llm-reasoning--大语言模型推理)
- [4. Correlation Analysis / 相关性分析](#4-correlation-analysis--相关性分析)
- [Results / 实验结果](#results--实验结果)
- [Tech Stack / 技术栈](#tech-stack--技术栈)
- [Repository Structure / 仓库结构](#repository-structure--仓库结构)
- [Getting Started / 快速开始](#getting-started--快速开始)
- [Limitations & Future Work / 局限性与未来工作](#limitations--future-work--局限性与未来工作)
- [Authors & Advisors / 作者与导师](#authors--advisors--作者与导师)
- [References / 参考文献](#references--参考文献)

---

## Overview / 项目概述

**EN** — The financial sector generates massive textual data, but most sentiment models behave as black boxes, offering predictions without human-readable justifications. This project builds a data pipeline that transforms labeled financial text into **explainable** datasets: it predicts sentiment **and** generates reasoning for *why* a given label was assigned, letting stakeholders trace model outputs back to the underlying text. Beyond classification, the project quantifies how market sentiment correlates with company financial metrics across industries.

**中文** —— 金融领域每天产生海量文本数据，但绝大多数情感分析模型都是「黑箱」，只给结论、不给依据。本项目构建了一条数据流水线，将带标签的金融文本转化为**可解释**数据集：模型不仅预测情感，还能生成「为什么是这个标签」的推理，让使用者可以把模型输出回溯到原始文本。除分类任务外，项目还量化分析了不同行业中市场情感与公司财务指标之间的相关性。

**Target audiences / 目标用户：**

| Audience / 用户 | Use case / 应用场景 |
| --- | --- |
| Financial institutions & banks / 金融机构与银行 | Gather market sentiment to optimize investment strategies / 采集市场情感以优化投资策略 |
| FinTech startups / 金融科技初创企业 | Integrate the sentiment pipeline into their own analytics products / 将情感流水线集成到自有分析产品中 |
| Market researchers & academics / 市场研究者与学者 | Enrich datasets for model training and evaluation / 丰富训练与评估数据集 |

---

## Motivation / 研究动机

**EN** — Two gaps drive this work:

1. **Explainability gap.** Existing approaches (lexicon-based, classical ML, deep learning, pre-trained transformers) struggle with context, jargon, and transparency. Attention weights alone do not produce human-understandable reasoning.
2. **Sentiment-vs-market gap.** Prior literature mostly studies short-term sentiment impact and aggregates sentiment without distinguishing types or drivers — and it largely ignores that the sentiment–performance relationship **varies by industry**.

**中文** —— 本项目针对两大研究空白：

1. **可解释性空白。** 现有方法（基于词典、传统机器学习、深度学习、预训练 Transformer）在上下文理解、专业术语、透明度上均有不足；仅靠注意力权重无法给出人类可理解的推理。
2. **情感与市场关系的空白。** 既有文献多关注短期情感影响，且在聚合情感时不区分类型或驱动因素，更普遍忽略了「情感—业绩」关系**因行业而异**这一事实。

---

## Pipeline Architecture / 流水线架构

```
┌─────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐   ┌──────────────────────────┐
│ 1. Data Curation│ → │ 2. FinBERT Fine-tune │ → │ 3. LLM Reasoning    │ → │ 4. Correlation Analysis  │
│  数据整理         │   │    + Explainability  │   │    (CoT, GPT-4o)    │   │    情感 × 财务指标         │
│                 │   │    FinBERT 微调       │   │    大语言模型推理      │   │                          │
│ collect/clean/  │   │    + 可解释性          │   │                     │   │                          │
│ augment/filter  │   │ Attention/LIME/SHAP  │   │ top-k attention →   │   │ Pearson / regression /   │
│                 │   │                      │   │ chain-of-thought    │   │ time-series              │
└─────────────────┘   └──────────────────────┘   └─────────────────────┘   └──────────────────────────┘
```

---

## 1. Data Curation / 数据整理

### Collection / 数据采集
**EN** — Reviewed 50+ open-source datasets and selected the best by annotation quality and language diversity. Live financial text was also pulled via NewsAPI and lightweight crawlers (BeautifulSoup, Selenium). Representative sources include news reports, earnings summaries, financial headlines, and market commentary.

**中文** —— 共审阅了 50+ 个开源数据集，按标注质量与语言多样性筛选优质数据；同时使用 NewsAPI 及轻量爬虫（BeautifulSoup、Selenium）抓取实时金融文本。来源覆盖新闻报道、财报摘要、财经标题与市场评论。

| Dataset / 数据集 | Size / 规模 | Distribution / 分布 (pos / neu / neg) |
| --- | --- | --- |
| Short Financial Sentences / 短金融句 | 211 | 62 / 65 / 84 |
| Full Financial Paragraphs / 段落级新闻 | 400 | 215 / 1 / 184 |
| Retail Investor Headlines / 散户投资者标题 | 4,846 | 1,363 / 2,879 / 604 |
| Entity-Level Annotated Headlines / 实体级标注标题 | 10,753 | 4,170 / 3,444 / 3,139 |

### Preprocessing / 预处理
**EN** — Unified heterogeneous datasets (Financial PhraseBank, FiQA, multiple Kaggle sets): dropped null rows, fixed encoding/whitespace, merged all text fields into a single `news` column, and standardized labels to `-1` (negative), `0` (neutral), `1` (positive). Split 70% / 30% train/test with preserved class proportions.

**中文** —— 统一多个异构数据集（Financial PhraseBank、FiQA、多个 Kaggle 数据集）：删除空值行，修复编码与空格，将所有文本字段合并到单一 `news` 列，并将标签标准化为 `-1`（负面）、`0`（中性）、`1`（正面）。按 **70% / 30%** 划分训练/测试集，并保持类别比例一致。

### Augmentation / 数据增强
**EN** — Raw distribution was severely imbalanced (~67% positive, ~25% neutral, ~8% negative). A **hybrid strategy** rebalanced minority classes to ~33.3% each:
- **Back Translation** — EN → FR → EN via Google Translate API (preserves sentiment, diversifies phrasing).
- **Synonym Substitution** — NLTK WordNet synsets replace 1–2 non-stopword keywords, with rules to filter low-readability outputs.
- **Hybrid** — minority samples split 50/50 between the two methods to avoid overfitting and semantic drift.

**中文** —— 原始分布严重失衡（约 67% 正面、25% 中性、8% 负面）。采用**混合增强策略**将少数类各自补齐至约 33.3%：
- **回译（Back Translation）** —— 经 Google Translate API 做 英→法→英 转换，保持情感不变、丰富表达。
- **同义词替换（Synonym Substitution）** —— 用 NLTK WordNet 同义词集替换 1–2 个非停用词关键词，并设规则过滤可读性下降的样本。
- **混合（Hybrid）** —— 少数类样本按 50/50 随机分配给上述两种方法，避免单一方法导致的过拟合或语义漂移。

### Filtering / 数据过滤
**EN** — Inspired by *Textbooks Are All You Need*, a classifier-based quality filter was applied. **Phi-2** runs at high temperature to produce 5 quality classifications per sample, then **majority voting** keeps only high-quality data. Decontamination via n-gram overlap and embedding-similarity checks maintains a strict train/test boundary.

**中文** —— 受《Textbooks Are All You Need》启发，引入基于分类器的质量过滤。使用 **Phi-2** 在高温度下对每个样本输出 5 次质量判定，再通过**多数投票**仅保留高质量数据；同时用 n-gram 重叠与嵌入相似度做去污染处理，严格隔离训练集与测试集。

---

## 2. FinBERT & Explainability / FinBERT 与可解释性

**EN** — A baseline **BERT** was fine-tuned on financial data, then **FinBERT** (domain-adapted via continued pretraining on financial news and corporate reports) was further tuned with hyperparameter search over batch size, learning rate, and weight decay. Three complementary explainability methods were layered on top:

- **Attention scores** — accumulated across transformer layers to surface the tokens the model attends to.
- **LIME** — perturbs input text and fits local interpretable models to highlight influential words per prediction.
- **SHAP** — quantifies each feature's contribution to the output, giving a global, game-theoretic view.

**中文** —— 先在金融数据上微调 **BERT** 作为基线，再对 **FinBERT**（在金融新闻与企业报告上继续预训练的领域专用版本）做超参数搜索（批大小、学习率、权重衰减）。在其之上叠加三种互补的可解释性方法：

- **注意力分数（Attention）** —— 跨 Transformer 层累积，揭示模型关注的 token。
- **LIME** —— 扰动输入文本并拟合局部可解释模型，标出对单条预测最有影响的词。
- **SHAP** —— 基于博弈论量化每个特征对输出的贡献，提供全局视角。

---

## 3. LLM Reasoning / 大语言模型推理

**EN** — The top-*k* highest-attention words from FinBERT are injected into a **chain-of-thought (CoT)** prompt for an LLM (**GPT-4o**), aiming to produce human-readable justifications alongside sentiment labels. Experiments compare:

1. **Baseline reasoning** — generated directly from the input sentence.
2. **Augmented reasoning** — guided by attention / explainability signals.

Generated reasoning is scored by an LLM-based evaluator across five dimensions: **Helpfulness, Truthfulness, Insightfulness, Relevance, Clarity**.

**中文** —— 将 FinBERT 注意力最高的 top-*k* 词注入大语言模型（**GPT-4o**）的**思维链（CoT）**提示，使其在给出情感标签的同时输出人类可读的推理。实验对比两种设置：

1. **基线推理** —— 直接基于输入句子生成。
2. **增强推理** —— 由注意力/可解释性信号引导。

生成的推理由一个基于 LLM 的评估器从五个维度打分：**有用性、真实性、洞察力、相关性、清晰度**。

---

## 4. Correlation Analysis / 相关性分析

> Authored by **Ke Zhang** (Section 8) · 本章节由 **Ke Zhang** 撰写

**EN** — Financial text is encoded by a BERT model (CLS vector + classifier) into sentiment scores in `[-1, 1]`. Company financials (P/E, ROE, EBITDA, Net Income, Total Debt, Total Assets, Revenue, ROI) are collected from Macrotrends, normalized to a comparable range, and aligned on a quarterly timeline (daily/monthly sentiment is averaged per quarter). **Pearson correlation, multiple linear regression, and time-series analysis** then quantify the sentiment–financials relationship per company and per industry.

**中文** —— 用 BERT 模型（CLS 向量 + 分类器）将金融文本编码为 `[-1, 1]` 区间的情感分数；公司财务指标（市盈率、ROE、EBITDA、净利润、总负债、总资产、营收、ROI）取自 Macrotrends，归一化到可比区间，并按季度对齐（日/月情感分数按季度取平均，如 1–3 月平均情感对应 Q1 财务指标）。再用 **Pearson 相关系数、多元线性回归、时间序列分析**，在公司与行业两个层面量化「情感—财务」关系。

### Key industry findings / 行业关键发现

**FinTech / 金融科技** — PayPal, Fiserv, Intuit, Block, JPMorgan
- **EBITDA** and **ROI** → positive correlation with sentiment (operational strength boosts tone). / **EBITDA** 与 **ROI** 与情感正相关（经营实力提振情绪）。
- **Total Debt** → negative correlation (higher leverage = perceived risk). / **总负债** 与情感负相关（高杠杆被视为风险）。
- Net Income, Revenue, Total Assets → larger error margins / noisier. / 净利润、营收、总资产误差较大、噪声更高。

**Consumer Staples / 消费必需品** — Coca-Cola, Hershey, Mondelez, Monster, Philip Morris, P&G, AB InBev
- **Net Income** → the most consistently correlated metric; public tone tracks real-time profit momentum. / **净利润** 是最稳定相关的指标，舆论情绪反映实时利润动能。
- **Total Assets** → mixed (some firms invest during good sentiment; others practice capital discipline). / **总资产** 表现不一（部分公司在情绪利好期扩张投资，部分则保持资本纪律）。
- EBITDA, Revenue, Long-Term Debt → weak / moderate, no clear significance. / EBITDA、营收、长期负债 相关性弱至中等，无明显显著性。

**Takeaway / 核心结论：** the most explanatory metric is **industry-dependent** — there is no single universal driver. / 最具解释力的指标**因行业而异**，不存在放之四海皆准的单一驱动因素。

---

## Results / 实验结果

### Multiclass classification / 多分类（positive / neutral / negative）

| Setup / 方案 | Accuracy / 准确率 |
| --- | --- |
| Fine-tuned BERT baseline / 微调 BERT 基线 | **0.8950** |
| 5 attention words + few-shot + CoT / 5 注意力词 + 少样本 + 思维链 | 0.8431 |
| 3 attention words + few-shot + CoT / 3 注意力词 + 少样本 + 思维链 | 0.8504 |

> **Finding / 发现:** In the multiclass setting, attention-guided prompting **hurt** accuracy — many positive/negative cases collapsed into *neutral*. 在多分类设置下，注意力引导提示**反而降低**了准确率，许多正面/负面样本被误判为中性。

### Binary classification / 二分类（positive / negative，移除中性类）

| Setup / 方案 | Accuracy / 准确率 |
| --- | --- |
| Fine-tuned BERT (binary) / 微调 BERT（二分类） | 0.8860 |
| 3 attention words + few-shot + CoT / 3 注意力词 + 少样本 + 思维链 | **0.9863** |
| Zero-shot prompting / 零样本提示 | **0.9908** |

> **Finding / 发现:** Removing the neutral class unlocked large gains — attention-guided reasoning improved ~11% over the BERT baseline, though it sat marginally below pure zero-shot prompting. 移除中性类后效果显著提升：注意力引导推理相比 BERT 基线提升约 **11%**，但仍略低于纯零样本提示。

### Reasoning quality / 推理质量（1–5）

| Dimension / 维度 | Zero-shot / 零样本 | Attention-augmented / 注意力增强 | Δ |
| --- | --- | --- | --- |
| Helpfulness / 有用性 | 4.67 | **4.86** | +0.19 |
| Truthfulness / 真实性 | 4.82 | 4.81 | −0.01 |
| Insightfulness / 洞察力 | 3.84 | **4.01** | +0.17 |
| Relevance / 相关性 | 4.84 | 4.87 | +0.03 |
| Clarity / 清晰度 | 4.95 | 4.94 | −0.01 |

> **Finding / 发现:** Even when attention signals don't improve label accuracy, they make explanations **more helpful and insightful** — a valuable complementary signal for richer rationales. 即便注意力信号未提升标签准确率，它仍能让解释**更有用、更具洞察**，是生成丰富推理的有价值补充信号。

---

## Tech Stack / 技术栈

| Layer / 层级 | Tools / 工具 |
| --- | --- |
| Models / 模型 | FinBERT, BERT, GPT-4o, Phi-2 |
| Explainability / 可解释性 | Attention scores, LIME, SHAP |
| Augmentation / 数据增强 | Google Translate API (back-translation), NLTK WordNet |
| Collection / 数据采集 | NewsAPI, BeautifulSoup, Selenium |
| Analysis / 分析 | Pearson correlation, multiple linear regression, time-series |
| Data sources / 数据源 | Kaggle, UCI, Financial PhraseBank, FiQA, Macrotrends |

---

## Repository Structure / 仓库结构

> Suggested layout — adapt to your actual repo. / 建议结构，请按实际仓库调整。

```
.
├── data/
│   ├── raw/                 # Collected datasets / 采集的原始数据集
│   ├── processed/           # Cleaned & unified (news + label) / 清洗合并后数据
│   └── augmented/           # Back-translation + synonym sub / 增强后数据
├── src/
│   ├── data_curation/       # Collection, cleaning, augmentation, Phi-2 filter / 数据整理
│   ├── finbert/             # Fine-tuning + hyperparameter search / FinBERT 微调
│   ├── explainability/      # Attention, LIME, SHAP / 可解释性
│   ├── llm_reasoning/       # CoT prompting + GPT-4o eval / 思维链推理与评估
│   └── correlation/         # Pearson, regression, time-series / 相关性分析
├── notebooks/               # Exploratory analysis & figures / 探索分析与图表
├── results/                 # Metrics tables & plots / 指标表与图
├── requirements.txt
└── README.md
```

---

## Getting Started / 快速开始

```bash
# 1. Clone / 克隆仓库
git clone https://github.com/<your-org>/financial-sentiment-xai.git
cd financial-sentiment-xai

# 2. Environment / 创建环境
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure API keys / 配置 API 密钥
#   OpenAI (GPT-4o), Google Translate, NewsAPI ...
export OPENAI_API_KEY="..."

# 4. Run the pipeline / 运行流水线
python src/data_curation/run.py
python src/finbert/train.py
python src/llm_reasoning/generate.py
python src/correlation/analyze.py
```

> ⚠️ API keys are required for GPT-4o, Google Translate, and NewsAPI. Store them as environment variables — never commit them. / GPT-4o、Google Translate、NewsAPI 均需 API 密钥，请用环境变量存储，切勿提交到仓库。

---

## Limitations & Future Work / 局限性与未来工作

**EN**
1. **High-quality CoT dataset.** Highest-attention words are *not* reliable predictors of sentiment class. Future work: given ground-truth labels + top attention words, test whether the model can generate accurate, high-quality reasoning, and iteratively refine the CoT dataset.
2. **Reasoning-specific models.** Current CoT explanations aren't yet good enough to train downstream models. Goal: build a human-aligned reasoning dataset via **RLHF**, then train reasoning/transformer models specialized for explainable sentiment analysis.
3. **Broader correlation scope.** Extend beyond industry segmentation (e.g., high- vs low-capital firms, domestic vs international) and explore Pharmacy and Customer Service sectors.

**中文**
1. **高质量思维链数据集。** 注意力最高的词**并非**情感类别的可靠预测因子。未来方向：在已知真实标签 + 高注意力词的前提下，验证模型能否生成准确、高质量的推理，并迭代优化思维链数据集。
2. **专用推理模型。** 当前思维链解释质量尚不足以训练下游模型。目标：通过 **RLHF** 构建与人类对齐的推理数据集，进而训练专门用于可解释情感分析的推理/Transformer 模型。
3. **更广的相关性研究。** 在行业划分之外引入更多维度（如高资本 vs 低资本、国内 vs 国际企业），并探索医药与客户服务等行业。

---

## Authors & Advisors / 作者与导师

**Authors / 作者:** Yitian Sun · Sirui Li · Jiehao Xing · Ke Zhang · Jalen Xing

**Advisors / 导师:** Adheesh Shenoy · Chan Yang · **SheltonAI**

| Section / 章节 | Lead / 负责人 |
| --- | --- |
| Executive Summary, Results, Limitations / 摘要、结果、局限 | Jalen Xing (JX) |
| Problem Space, High-Level Approach, Data Curation / 问题空间、总体方法、数据整理 | Yitian Sun (YS) |
| Existing Solutions, Model Architecture / 已有方案、模型架构 | Sirui Li (SL) |
| Data Curation / 数据整理 | Jiehao Xing (JhX) |
| Correlation Analysis / 相关性分析 | **Ke Zhang (KZ)** |

---

## References / 参考文献

Selected key references (full list in the report). / 精选核心参考文献（完整列表见报告）：

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805
- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903
- Wang, X. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* arXiv:2203.11171
- Gunasekar, S. et al. (2023). *Textbooks Are All You Need.* arXiv:2306.11644
- Doshi-Velez, F. & Kim, B. (2017). *Towards a Rigorous Science of Interpretable Machine Learning.* arXiv:1702.08608
- Tetlock, P. C. (2007). *Giving Content to Investor Sentiment: The Role of Media in the Stock Market.* The Journal of Finance, 62(3).
- Li, X., Xie, Q. & Jiang, L. (2021). *Explainable Deep Learning Models in Sentiment Analysis: A Systematic Review.* Information Fusion, 68.

---

## Acknowledgments / 致谢

This project was conducted with **SheltonAI**, an AI solutions company focused on transparency and control for investors. / 本项目与 **SheltonAI**（专注于为投资者提供透明度与可控性的 AI 解决方案公司）合作完成。
