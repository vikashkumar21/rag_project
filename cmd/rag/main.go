// cmd/rag is the CLI entry point for the RAG project.
// Run it with:
//
//	OPENAI_API_KEY=sk-... go run ./cmd/rag
//
// Or with a .env file:
//
//	go run ./cmd/rag
//
// Flags:
//
//	-pdf   path to the source PDF (default: eBook-How-to-Build-a-Career-in-AI.pdf)
//	-eval  path to eval questions file (default: eval_questions.txt)
//	-mode  pipeline to run: "baseline", "sentence", "automerging", or "all" (default: "all")
//	-topk  number of chunks to retrieve per query (default: 6)
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/vikash/rag_project/internal/embedder"
	"github.com/vikash/rag_project/internal/eval"
	"github.com/vikash/rag_project/internal/llm"
	"github.com/vikash/rag_project/internal/loader"
	"github.com/vikash/rag_project/internal/retriever"
	"github.com/vikash/rag_project/internal/vectorstore"
	"github.com/vikash/rag_project/pipeline"
)

// fallbackSampleText is used if the course PDF is not yet placed in the workspace.
const fallbackSampleText = `How to Build a Career in AI
Chapter 1: Finding Projects to Build Your Experience
When starting out in AI, building projects is essential for gaining hands-on experience and proving your capabilities.
Here are the key steps to take when finding projects:
1. Start with simple projects that allow you to master fundamental concepts before moving to complex models.
2. Choose projects related to domains you care about or have prior knowledge in. This domain interest helps sustain motivation.
3. Aim to build projects that show a progression of complexity: starting from reproducing existing models or tutorials, moving to applying established models to novel datasets, and finally creating end-to-end applications that solve real-world problems.
4. Focus on scoping projects carefully. Avoid overly ambitious goals initially. It is far better to complete a well-scoped project end-to-end with evaluation and deployment than to leave an ambitious project half-finished.
5. Work with public datasets (such as Kaggle, Hugging Face, or government data) or collect your own specialized dataset to demonstrate data engineering acumen.

Chapter 2: Building an AI Portfolio
A strong AI portfolio is more convincing to employers than a list of courses.
To build an effective portfolio:
1. Document your work clearly with public GitHub repositories, readable README files, and clean reproducible code.
2. Write blog posts or technical summaries explaining the problem, your architecture decisions, challenges encountered, and evaluation metrics.
3. Share your progress on platforms like LinkedIn, X, or technical forums. Communicating your work helps build your professional network and establishes credibility.
4. Highlight real-world utility: demonstrate how your project delivers business value, solves a practical bottleneck, or handles messy edge cases.

Chapter 3: Choosing the Right AI Job Role
There are multiple career pathways in artificial intelligence:
- Machine Learning Engineer: Focuses on software engineering, model deployment, pipelines, and scaling inference in production.
- Data Scientist: Focuses on analyzing data, formulating business hypotheses, experimentation, and statistical modeling.
- AI Research Scientist: Pushes the frontiers of machine learning, developing novel neural architectures and publishing papers.
- Full-Stack AI Application Developer: Integrates foundation models via APIs, building user-facing RAG systems, agents, and intuitive workflows.
Select the role that matches your strengths in coding, mathematics, or product thinking.`

func main() {
	// ── Flags ──────────────────────────────────────────────────────────────────
	pdfPath := flag.String("pdf", "eBook-How-to-Build-a-Career-in-AI.pdf", "path to source PDF")
	evalFile := flag.String("eval", "eval_questions.txt", "path to evaluation questions file")
	mode := flag.String("mode", "all", "pipeline mode: baseline, sentence, automerging, or all")
	topK := flag.Int("topk", 6, "number of chunks to retrieve per query")
	flag.Parse()

	ctx := context.Background()

	// ── Load .env (optional) ───────────────────────────────────────────────────
	_ = godotenv.Load()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is not set. Export it or create a .env file with OPENAI_API_KEY=sk-...")
	}

	// ── 1. Load Document ───────────────────────────────────────────────────────
	var document loader.Document
	if _, err := os.Stat(*pdfPath); err == nil {
		fmt.Printf("📄 Loading PDF: %s...\n", *pdfPath)
		pages, err := loader.LoadPDF(*pdfPath)
		if err != nil {
			log.Fatalf("failed to load PDF: %v", err)
		}
		document = loader.MergeDocuments(pages, "\n\n")
		fmt.Printf("   Loaded %d pages (%d runes)\n", len(pages), len([]rune(document.Text)))
	} else {
		fmt.Printf("ℹ️  '%s' not found in workspace.\n   Using built-in sample text ('How to Build a Career in AI').\n", *pdfPath)
		document = loader.Document{
			ID:       "sample-career-in-ai",
			Text:     fallbackSampleText,
			Metadata: map[string]string{"source": "sample_ebook"},
		}
	}

	// Shared clients
	emb := embedder.NewOpenAI(apiKey, "text-embedding-3-small")
	lm := llm.NewOpenAI(apiKey, "gpt-3.5-turbo", 0.1)
	judge := llm.NewOpenAI(apiKey, "gpt-3.5-turbo", 0.1)
	ranker := retriever.NewLLMReranker(judge)

	// Load evaluation questions
	evalQs := loadLines(*evalFile)
	if len(evalQs) == 0 {
		evalQs = []string{
			"What are steps to take when finding projects to build your experience?",
			"How do I build a portfolio of AI projects?",
			"What is the right AI job for me?",
		}
	} else {
		evalQs = append(evalQs, "What is the right AI job for me?")
	}

	var evaluators []*eval.Evaluator

	runBaseline := *mode == "baseline" || *mode == "all"
	runSentence := *mode == "sentence" || *mode == "all"
	runAutoMerge := *mode == "automerging" || *mode == "all"

	// ══════════════════════════════════════════════════════════════════════════
	// 1. BASELINE RAG PIPELINE (Direct Query Engine)
	// ══════════════════════════════════════════════════════════════════════════
	if runBaseline {
		fmt.Println("\n=======================================================")
		fmt.Println("🚀 1. BASELINE RAG PIPELINE (Direct Query Engine)")
		fmt.Println("=======================================================")
		store := vectorstore.NewInMemory()
		rag, err := pipeline.Build(ctx, []loader.Document{document}, pipeline.Config{
			Embedder: emb,
			Store:    store,
			LLM:      lm,
			TopK:     *topK,
		})
		if err != nil {
			log.Fatalf("build baseline: %v", err)
		}

		evaluator := eval.NewEvaluator(judge, "Direct Query Engine (Baseline)")
		runEvaluation(ctx, "Baseline", rag, evalQs, evaluator)
		evaluators = append(evaluators, evaluator)
	}

	// ══════════════════════════════════════════════════════════════════════════
	// 2. SENTENCE-WINDOW RAG PIPELINE (Sentence Window Query Engine)
	// ══════════════════════════════════════════════════════════════════════════
	if runSentence {
		fmt.Println("\n=======================================================")
		fmt.Println("🚀 2. ADVANCED RAG: SENTENCE-WINDOW RETRIEVAL")
		fmt.Println("=======================================================")
		store := vectorstore.NewInMemory()
		rag, err := pipeline.BuildSentenceWindow(ctx, []loader.Document{document}, pipeline.SentenceWindowConfig{
			Embedder:        emb,
			Store:           store,
			LLM:             lm,
			Reranker:        ranker,
			SimilarityTopK:  6,
			RerankTopN:      2,
			SentenceOptions: loader.DefaultSentenceWindowOptions(),
		})
		if err != nil {
			log.Fatalf("build sentence-window: %v", err)
		}

		evaluator := eval.NewEvaluator(judge, "Sentence Window Query Engine")
		runEvaluation(ctx, "Sentence-Window", rag, evalQs, evaluator)
		evaluators = append(evaluators, evaluator)
	}

	// ══════════════════════════════════════════════════════════════════════════
	// 3. AUTO-MERGING RAG PIPELINE (Automerging Query Engine)
	// ══════════════════════════════════════════════════════════════════════════
	if runAutoMerge {
		fmt.Println("\n=======================================================")
		fmt.Println("🚀 3. ADVANCED RAG: AUTO-MERGING RETRIEVAL")
		fmt.Println("=======================================================")
		store := vectorstore.NewInMemory()
		docStore := vectorstore.NewInMemoryDocStore()
		rag, err := pipeline.BuildAutoMerging(ctx, []loader.Document{document}, pipeline.AutoMergingConfig{
			Embedder:       emb,
			Store:          store,
			DocStore:       docStore,
			LLM:            lm,
			Reranker:       ranker,
			SimilarityTopK: 12,
			RerankTopN:     2,
			Threshold:      0.5,
			Verbose:        false,
			HierarchyOpts:  loader.DefaultHierarchicalOptions(),
		})
		if err != nil {
			log.Fatalf("build automerging: %v", err)
		}

		evaluator := eval.NewEvaluator(judge, "Automerging Query Engine")
		runEvaluation(ctx, "Auto-Merging", rag, evalQs, evaluator)
		evaluators = append(evaluators, evaluator)
	}

	// ══════════════════════════════════════════════════════════════════════════
	// TRULENS LEADERBOARD SUMMARY
	// ══════════════════════════════════════════════════════════════════════════
	if len(evaluators) > 0 {
		eval.PrintLeaderboard(evaluators...)
	}
}

type queryable interface {
	Query(ctx context.Context, question string) (pipeline.QueryResult, error)
}

func runEvaluation(ctx context.Context, name string, rag queryable, questions []string, evaluator *eval.Evaluator) {
	fmt.Printf("\nEvaluating %d question(s) with RAG Triad...\n", len(questions))
	fmt.Printf("   %-50s  %-8s %-8s %-8s\n", "Question", "Ctx Rel", "Ground", "Ans Rel")
	fmt.Println("   " + strings.Repeat("─", 82))

	for i, q := range questions {
		res, err := rag.Query(ctx, q)
		if err != nil {
			log.Printf("[%s] query err: %v", name, err)
			continue
		}

		// First question: print snippet of sample output
		if i == 0 {
			fmt.Printf("\n💬 Sample Answer [%s]:\n%s\n\n", name, strings.TrimSpace(res.Answer))
		}

		rec, eErr := evaluator.Evaluate(ctx, q, res.Context, res.Answer)
		if eErr != nil {
			log.Printf("[%s] eval err: %v", name, eErr)
			continue
		}
		fmt.Printf("   %-50s  %-8.3f %-8.3f %-8.3f\n",
			truncate(q, 48), rec.ContextRelevance, rec.Groundedness, rec.AnswerRelevance)
	}
}

func loadLines(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			lines = append(lines, line)
		}
	}
	return lines
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}
