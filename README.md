# RAG Project in Go 🦾

An Advanced **Retrieval Augmented Generation (RAG)** system built from scratch in Go, translating the Python LlamaIndex & TruLens advanced RAG curriculum into idiomatic, concurrent Go.

---

## 📚 Curriculum & Implemented Pipelines

| Pipeline / Technique | Description | Status |
|---|---|---|
| **Baseline RAG** | Fixed-size overlapping chunks + Top-K similarity search (`Direct Query Engine`) | ✅ Complete |
| **Sentence-Window Retrieval** | Indexes single sentences for fine-grained search, expands to surrounding sentence window + reranking (`Sentence Window Query Engine`) | ✅ Complete |
| **Auto-Merging Retrieval** | Hierarchical document trees (2048 → 512 → 128); searches leaf nodes and merges upwards to parent chunks (`Automerging Query Engine`) | ✅ Complete |
| **RAG Triad Evaluation** | Evaluates **Context Relevance**, **Groundedness**, and **Answer Relevance** using LLM-as-judge | ✅ Complete |
| **TruLens-style Leaderboard** | Side-by-side comparative table benchmarking all pipelines against evaluation questions | ✅ Complete |

---

## 🏗️ Project Architecture

```
rag_project/
├── go.mod
├── README.md
├── RAG_Course_Guide.md       # Comprehensive lesson guide & conceptual walkthrough
├── eval_questions.txt        # Benchmark evaluation queries
├── internal/
│   ├── loader/               # PDF loading, sentence segmenter, hierarchical chunker
│   ├── embedder/             # OpenAI text-embedding-3-small provider
│   ├── vectorstore/          # In-memory cosine vector store + DocStore for trees
│   ├── retriever/            # Similarity, Sentence-Window, and Auto-Merging retrievers + Reranker
│   ├── llm/                  # OpenAI chat completions client (gpt-3.5-turbo / gpt-4o)
│   └── eval/                 # RAG Triad evaluation metrics & Leaderboard printer
├── pipeline/
│   ├── baseline.go           # Baseline RAG assembly
│   ├── sentence_window.go    # Sentence-window indexing & query workflow
│   └── automerging.go        # Auto-merging indexing & query workflow
└── cmd/
    └── rag/                  # Multi-mode CLI entry point
```

> 📖 **Deep Dive**: See [RAG_Course_Guide.md](file:///Users/vikash/projects/rag_project/RAG_Course_Guide.md) for in-depth algorithmic explanations, Mermaid architecture diagrams, and Python vs Go comparison tables.

---

## 🚀 Getting Started

### 1. Prerequisites
- Go 1.25+
- OpenAI API key

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env:
OPENAI_API_KEY=sk-...
```

### 3. Run Lessons & Pipelines

#### Run Lesson 3: Sentence Window Retrieval In-Depth (Window 1 vs 3, Reranker Demo & Persistence)
```bash
go run ./cmd/rag -mode=lesson3
```
Add `-dashboard` to launch the interactive web dashboard on `http://localhost:8501/`:
```bash
go run ./cmd/rag -mode=lesson3 -dashboard
```

#### Run Lesson 2: RAG Triad with Chain-of-Thought
```bash
go run ./cmd/rag -mode=lesson2 -dashboard
```

#### Run All Pipelines (Comparative Leaderboard: Baseline vs Sentence-Window vs Auto-Merging)
```bash
go run ./cmd/rag -mode=all
```

#### Run a Specific Architecture:
```bash
# Sentence-Window Retrieval
go run ./cmd/rag -mode=sentence

# Auto-Merging Retrieval
go run ./cmd/rag -mode=automerging

# Baseline RAG
go run ./cmd/rag -mode=baseline
```

*(Optional)* Specify a custom PDF or evaluation file:
```bash
go run ./cmd/rag -pdf=eBook-How-to-Build-a-Career-in-AI.pdf -eval=eval_questions.txt
```

### 4. Run Tests
```bash
go test -v ./...
```
