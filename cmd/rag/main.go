// cmd/rag is the CLI entry point for the RAG project.
//
// Usage:
//
//	# Run Lesson 3: Sentence Window Retrieval (window 1 vs 3, reranker demo, index persistence)
//	go run ./cmd/rag -mode=lesson3
//
//	# Run Lesson 2: RAG Triad of Metrics with Sentence-Window & CoT
//	go run ./cmd/rag -mode=lesson2
//
//	# Run with interactive dashboard on :8501
//	go run ./cmd/rag -mode=lesson3 -dashboard
//
//	# Run all pipelines (Baseline, Sentence-Window, Auto-Merging)
//	go run ./cmd/rag -mode=all
//
// Flags:
//
//	-mode       pipeline/lesson mode: "lesson3", "lesson2", "baseline", "sentence", "automerging", or "all" (default: "lesson3")
//	-dashboard  launch TruLens-style interactive web dashboard at http://localhost:8501 (default: false)
//	-pdf        path to source PDF (default: eBook-How-to-Build-a-Career-in-AI.pdf)
//	-eval       path to eval questions file (default: eval_questions.txt)
//	-topk       number of chunks to retrieve per query (default: 6)
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
Select the role that matches your strengths in coding, mathematics, or product thinking.

Chapter 4: Keys to Success in AI
To be successful in AI, continuous learning and perseverance are paramount. The field evolves rapidly, so developing strong foundations in core algorithms, staying curious, and reading research papers weekly will keep you at the frontier. Cultivate good communication skills so you can explain technical machine learning trade-offs clearly to non-technical stakeholders.`

func main() {
	// ── Flags ──────────────────────────────────────────────────────────────────
	pdfPath := flag.String("pdf", "eBook-How-to-Build-a-Career-in-AI.pdf", "path to source PDF")
	evalFile := flag.String("eval", "eval_questions.txt", "path to evaluation questions file")
	mode := flag.String("mode", "lesson3", "mode: lesson3, lesson2, baseline, sentence, automerging, or all")
	dashboard := flag.Bool("dashboard", false, "start interactive dashboard on http://localhost:8501")
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

	var evaluators []*eval.Evaluator

	// ══════════════════════════════════════════════════════════════════════════
	// LESSON 3: SENTENCE WINDOW RETRIEVAL IN-DEPTH (Window Sizing, Reranking & Persistence)
	// ══════════════════════════════════════════════════════════════════════════
	if *mode == "lesson3" {
		fmt.Println("\n=======================================================")
		fmt.Println("🎓 LESSON 3: SENTENCE WINDOW RETRIEVAL IN-DEPTH")
		fmt.Println("=======================================================")

		// 1. Concept Exploration: SentenceWindowNodeParser
		fmt.Println("\n--- 1. Sentence Window Node Parser Demo ---")
		text1 := "hello. how are you? I am fine!  "
		nodes1 := loader.BuildSentenceWindows(loader.Document{ID: "demo1", Text: text1}, loader.SentenceWindowOptions{WindowSize: 3})
		var sents1 []string
		for _, n := range nodes1 {
			sents1 = append(sents1, n.Text)
		}
		fmt.Printf("Input text 1: %q\n", text1)
		fmt.Printf("Extracted sentences: %v\n", sents1)
		if len(nodes1) > 1 {
			fmt.Printf("nodes[1].metadata[\"window\"] -> %q\n\n", nodes1[1].Metadata["window"])
		}

		text2 := "hello. foo bar. cat dog. mouse"
		nodes2 := loader.BuildSentenceWindows(loader.Document{ID: "demo2", Text: text2}, loader.SentenceWindowOptions{WindowSize: 3})
		var sents2 []string
		for _, n := range nodes2 {
			sents2 = append(sents2, n.Text)
		}
		fmt.Printf("Input text 2: %q\n", text2)
		fmt.Printf("Extracted sentences: %v\n", sents2)
		if len(nodes2) > 0 {
			fmt.Printf("nodes[0].metadata[\"window\"] -> %q\n", nodes2[0].Metadata["window"])
		}

		// 2. Metadata Replacement PostProcessor Demo
		fmt.Println("\n--- 2. Metadata Replacement PostProcessor Demo ---")
		if len(nodes1) > 1 {
			fmt.Printf("Node text before postprocessor: %q\n", nodes1[1].Text)
			fmt.Printf("Node text after MetadataReplacementPostProcessor: %q\n", nodes1[1].Metadata["window"])
		}

		// 3. Reranker Demonstration (Bi-Encoder vs Cross-Encoder)
		fmt.Println("\n--- 3. Reranker Demonstration (Why Reranking Matters) ---")
		toyQuery := "I want a dog."
		toyCandidates := []retriever.Chunk{
			{ID: "c1", Text: "This is a cat", Score: 0.6},
			{ID: "c2", Text: "This is a dog", Score: 0.4},
		}
		fmt.Printf("Query: %q\n", toyQuery)
		fmt.Printf("Bi-encoder similarity scores (misled by topic similarity):\n")
		fmt.Printf("  • %q -> score: %.2f\n", toyCandidates[0].Text, toyCandidates[0].Score)
		fmt.Printf("  • %q -> score: %.2f\n", toyCandidates[1].Text, toyCandidates[1].Score)

		fmt.Println("Applying LLM / Cross-Encoder Reranker...")
		rerankedToy, err := ranker.Rerank(ctx, toyQuery, toyCandidates, 2)
		if err == nil && len(rerankedToy) > 0 {
			fmt.Printf("Reranked Output (Cross-attention fixes ranking):\n")
			for i, c := range rerankedToy {
				fmt.Printf("  #%d: %q (score: %.2f)\n", i+1, c.Text, c.Score)
			}
		}

		// 4. Persistence & Caching: Window Size = 1 vs Window Size = 3
		fmt.Println("\n--- 4. Building / Loading Sentence Window Indexes (Persistent) ---")
		engine1, err := pipeline.BuildSentenceWindow(ctx, []loader.Document{document}, pipeline.SentenceWindowConfig{
			Embedder:        emb,
			LLM:             lm,
			Reranker:        ranker,
			SimilarityTopK:  6,
			RerankTopN:      2,
			SaveDir:         "./sentence_index_1",
			SentenceOptions: loader.SentenceWindowOptions{WindowSize: 1},
		})
		if err != nil {
			log.Fatalf("build engine 1: %v", err)
		}

		engine3, err := pipeline.BuildSentenceWindow(ctx, []loader.Document{document}, pipeline.SentenceWindowConfig{
			Embedder:        emb,
			LLM:             lm,
			Reranker:        ranker,
			SimilarityTopK:  6,
			RerankTopN:      2,
			SaveDir:         "./sentence_index_3",
			SentenceOptions: loader.SentenceWindowOptions{WindowSize: 3},
		})
		if err != nil {
			log.Fatalf("build engine 3: %v", err)
		}

		// 5. Query execution
		sampleQ := "What are the keys to building a career in AI?"
		fmt.Printf("\n❓ Query: %s\n", sampleQ)
		out3, err := engine3.Query(ctx, sampleQ)
		if err == nil {
			fmt.Printf("💬 Answer (Window 3 Engine):\n%s\n", strings.TrimSpace(out3.Answer))
		}

		// 6. Systematic Experimentation & TruLens Leaderboard
		fmt.Println("\n--- 5. TruLens Evaluation: Window Size 1 vs Window Size 3 ---")
		evalQuestions := loadLines("generated_questions.text")
		if len(evalQuestions) == 0 {
			evalQuestions = []string{"What are the keys to building a career in AI?"}
		}

		eval1 := eval.NewEvaluator(judge, "sentence window engine 1")
		eval.RegisterEvaluator(eval1)
		eval3 := eval.NewEvaluator(judge, "sentence window engine 3")
		eval.RegisterEvaluator(eval3)

		fmt.Printf("Evaluating %d question(s) with RAG Triad...\n", len(evalQuestions))
		for _, q := range evalQuestions {
			r1, _ := engine1.Query(ctx, q)
			rec1, _ := eval1.Evaluate(ctx, q, r1.Context, r1.Answer)

			r3, _ := engine3.Query(ctx, q)
			rec3, _ := eval3.Evaluate(ctx, q, r3.Context, r3.Answer)

			fmt.Printf("   [Q: %s]\n", truncate(q, 45))
			fmt.Printf("     • Engine 1 (Window 1): Ctx: %.2f | Grd: %.2f | Ans: %.2f\n",
				rec1.ContextRelevance.Score, rec1.Groundedness.Score, rec1.AnswerRelevance.Score)
			fmt.Printf("     • Engine 3 (Window 3): Ctx: %.2f | Grd: %.2f | Ans: %.2f\n",
				rec3.ContextRelevance.Score, rec3.Groundedness.Score, rec3.AnswerRelevance.Score)
		}

		// Leaderboard
		eval.PrintLeaderboard(eval1, eval3)

		_ = eval1.SaveJSON("eval_results_sentence_win1.json")
		_ = eval3.SaveJSON("eval_results_sentence_win3.json")
		fmt.Println("\n💾 Results saved to eval_results_sentence_win1.json & eval_results_sentence_win3.json")

		if *dashboard {
			if err := eval.StartDashboard("127.0.0.1:8501"); err != nil {
				log.Fatalf("dashboard: %v", err)
			}
		} else {
			fmt.Println("\n💡 Tip: Run with -dashboard to launch the interactive TruLens UI on http://localhost:8501/")
		}
		return
	}

	// ══════════════════════════════════════════════════════════════════════════
	// LESSON 2 SPECIFIC MODE (App_1 with Sentence-Window & Full CoT RAG Triad)
	// ══════════════════════════════════════════════════════════════════════════
	if *mode == "lesson2" || *mode == "app1" {
		fmt.Println("\n=======================================================")
		fmt.Println("🎓 LESSON 2: RAG TRIAD OF METRICS (App_1)")
		fmt.Println("=======================================================")

		// Build sentence window engine
		store := vectorstore.NewInMemory()
		sentenceEngine, err := pipeline.BuildSentenceWindow(ctx, []loader.Document{document}, pipeline.SentenceWindowConfig{
			Embedder:        emb,
			Store:           store,
			LLM:             lm,
			Reranker:        ranker,
			SimilarityTopK:  6,
			RerankTopN:      2,
			SentenceOptions: loader.DefaultSentenceWindowOptions(),
		})
		if err != nil {
			log.Fatalf("build sentence engine: %v", err)
		}

		// Initial test query from Lesson 2
		testQuery := "How do you create your AI portfolio?"
		fmt.Printf("\n❓ Initial Query: %s\n", testQuery)
		out, err := sentenceEngine.Query(ctx, testQuery)
		if err != nil {
			log.Fatalf("query: %v", err)
		}
		fmt.Printf("💬 Output:\n%s\n", strings.TrimSpace(out.Answer))

		// TruLlama recorder for App_1
		evaluator := eval.NewEvaluator(judge, "App_1")
		eval.RegisterEvaluator(evaluator)
		evaluators = append(evaluators, evaluator)

		// Load evaluation questions and append Lesson 2's new question
		evalQs := loadLines(*evalFile)
		if len(evalQs) == 0 {
			evalQs = []string{
				"How do you build a portfolio of AI projects?",
				"What skills are most important for an AI engineer?",
				"How do I transition into an AI career from a software engineering background?",
			}
		}
		evalQs = append(evalQs, "How can I be successful in AI?")

		fmt.Printf("\nEvaluating %d question(s) with Feedback Functions:\n", len(evalQs))
		fmt.Printf(" 1. Answer Relevance (with CoT reasons)\n")
		fmt.Printf(" 2. Context Relevance (aggregated across nodes with CoT reasons)\n")
		fmt.Printf(" 3. Groundedness (statement-level verification with CoT reasons)\n\n")

		for _, q := range evalQs {
			res, qErr := sentenceEngine.Query(ctx, q)
			if qErr != nil {
				log.Printf("query error on %q: %v", q, qErr)
				continue
			}
			rec, eErr := evaluator.Evaluate(ctx, q, res.Context, res.Answer)
			if eErr != nil {
				log.Printf("eval error on %q: %v", q, eErr)
				continue
			}
			fmt.Printf("   ✔ [%-40s] Ctx: %.2f | Grd: %.2f | Ans: %.2f\n",
				truncate(q, 38), rec.ContextRelevance.Score, rec.Groundedness.Score, rec.AnswerRelevance.Score)
		}

		// Print records table matching: records[["input", "output"] + feedback]
		eval.PrintRecordsDetail(evaluator)

		// Print TruLens Leaderboard
		eval.PrintLeaderboard(evaluator)

		// Save records to JSON file
		_ = evaluator.SaveJSON("eval_results_lesson2.json")
		fmt.Println("\n💾 Records saved to eval_results_lesson2.json")

		if *dashboard {
			if err := eval.StartDashboard("127.0.0.1:8501"); err != nil {
				log.Fatalf("dashboard: %v", err)
			}
		} else {
			fmt.Println("\n💡 Tip: Run with -dashboard to launch the interactive TruLens UI on http://localhost:8501/")
		}
		return
	}

	// ══════════════════════════════════════════════════════════════════════════
	// MULTI-PIPELINE COMPARISON MODES (Baseline, Sentence-Window, Auto-Merging)
	// ══════════════════════════════════════════════════════════════════════════
	evalQs := loadLines(*evalFile)
	if len(evalQs) == 0 {
		evalQs = []string{
			"What are steps to take when finding projects to build your experience?",
			"How do I build a portfolio of AI projects?",
			"How can I be successful in AI?",
		}
	} else {
		evalQs = append(evalQs, "How can I be successful in AI?")
	}

	runBaseline := *mode == "baseline" || *mode == "all"
	runSentence := *mode == "sentence" || *mode == "all"
	runAutoMerge := *mode == "automerging" || *mode == "all"

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
		eval.RegisterEvaluator(evaluator)
		runEvaluation(ctx, "Baseline", rag, evalQs, evaluator)
		evaluators = append(evaluators, evaluator)
	}

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
		eval.RegisterEvaluator(evaluator)
		runEvaluation(ctx, "Sentence-Window", rag, evalQs, evaluator)
		evaluators = append(evaluators, evaluator)
	}

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
		eval.RegisterEvaluator(evaluator)
		runEvaluation(ctx, "Auto-Merging", rag, evalQs, evaluator)
		evaluators = append(evaluators, evaluator)
	}

	if len(evaluators) > 0 {
		eval.PrintLeaderboard(evaluators...)
	}

	if *dashboard {
		if err := eval.StartDashboard("127.0.0.1:8501"); err != nil {
			log.Fatalf("dashboard: %v", err)
		}
	}
}

type queryable interface {
	Query(ctx context.Context, question string) (pipeline.QueryResult, error)
}

func runEvaluation(ctx context.Context, name string, rag queryable, questions []string, evaluator *eval.Evaluator) {
	fmt.Printf("\nEvaluating %d question(s) with RAG Triad...\n", len(questions))
	for i, q := range questions {
		res, err := rag.Query(ctx, q)
		if err != nil {
			log.Printf("[%s] query err: %v", name, err)
			continue
		}

		rec, eErr := evaluator.Evaluate(ctx, q, res.Context, res.Answer)
		if eErr != nil {
			log.Printf("[%s] eval err: %v", name, eErr)
			continue
		}
		if i == 0 {
			fmt.Printf("   Sample Answer [%s]:\n   %s\n\n", name, truncate(strings.TrimSpace(res.Answer), 120))
		}
		fmt.Printf("   ✔ [%-40s] Ctx: %.2f | Grd: %.2f | Ans: %.2f\n",
			truncate(q, 38), rec.ContextRelevance.Score, rec.Groundedness.Score, rec.AnswerRelevance.Score)
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
