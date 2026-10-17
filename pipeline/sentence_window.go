package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/vikash/rag_project/internal/embedder"
	"github.com/vikash/rag_project/internal/llm"
	"github.com/vikash/rag_project/internal/loader"
	"github.com/vikash/rag_project/internal/retriever"
	"github.com/vikash/rag_project/internal/vectorstore"
)

// SentenceWindowRAG coordinates sentence-level indexing, window expansion,
// reranking, and generation.
//
// Python equivalent:
//
//	sentence_index = build_sentence_window_index(...)
//	sentence_window_engine = get_sentence_window_query_engine(sentence_index)
type SentenceWindowRAG struct {
	ret retriever.Retriever
	lm  llm.LLM
}

// SentenceWindowConfig configures the SentenceWindow pipeline.
type SentenceWindowConfig struct {
	Embedder        embedder.Embedder
	Store           vectorstore.Store
	LLM             llm.LLM
	Reranker        retriever.Reranker
	SimilarityTopK  int
	RerankTopN      int
	SentenceOptions loader.SentenceWindowOptions
}

// BuildSentenceWindow creates and populates a sentence-window pipeline.
func BuildSentenceWindow(ctx context.Context, docs []loader.Document, cfg SentenceWindowConfig) (*SentenceWindowRAG, error) {
	if cfg.SimilarityTopK <= 0 {
		cfg.SimilarityTopK = 6
	}
	if cfg.RerankTopN <= 0 {
		cfg.RerankTopN = 2
	}
	if cfg.SentenceOptions.WindowSize <= 0 {
		cfg.SentenceOptions = loader.DefaultSentenceWindowOptions()
	}

	// 1. Build sentence windows
	var allNodes []loader.SentenceNode
	for _, d := range docs {
		nodes := loader.BuildSentenceWindows(d, cfg.SentenceOptions)
		allNodes = append(allNodes, nodes...)
	}

	fmt.Printf("[sentence-window] extracted %d sentences (window_size=%d)\n",
		len(allNodes), cfg.SentenceOptions.WindowSize)

	// 2. Embed the focus sentences in batches
	const batchSize = 256
	for batchStart := 0; batchStart < len(allNodes); batchStart += batchSize {
		end := batchStart + batchSize
		if end > len(allNodes) {
			end = len(allNodes)
		}
		batch := allNodes[batchStart:end]

		texts := make([]string, len(batch))
		for i, n := range batch {
			texts[i] = n.Text
		}

		embeddings, err := cfg.Embedder.Embed(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("sentence_window/build: embed batch at %d: %w", batchStart, err)
		}

		vecs := make([]vectorstore.Vector, len(batch))
		for i, n := range batch {
			vecs[i] = vectorstore.Vector{
				ID:        n.ID,
				Text:      n.Text,
				Embedding: embeddings[i],
				Metadata:  n.Metadata,
			}
		}

		if err := cfg.Store.Add(ctx, vecs); err != nil {
			return nil, fmt.Errorf("sentence_window/build: store batch: %w", err)
		}
		fmt.Printf("[sentence-window] indexed %d / %d sentences\n", end, len(allNodes))
	}

	// 3. Assemble retriever
	ret := retriever.NewSentenceWindow(retriever.SentenceWindowConfig{
		Embedder:        cfg.Embedder,
		Store:           cfg.Store,
		Reranker:        cfg.Reranker,
		SimilarityTopK:  cfg.SimilarityTopK,
		RerankTopN:      cfg.RerankTopN,
		MetaReplacement: "window",
	})

	return &SentenceWindowRAG{
		ret: ret,
		lm:  cfg.LLM,
	}, nil
}

// Query retrieves expanded window context and generates an answer.
func (p *SentenceWindowRAG) Query(ctx context.Context, question string) (QueryResult, error) {
	chunks, err := p.ret.Retrieve(ctx, question, 0)
	if err != nil {
		return QueryResult{}, fmt.Errorf("sentence_window/query: retrieve: %w", err)
	}

	var sb strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&sb, "--- Window Context %d (score %.3f) ---\n%s\n\n",
			i+1, c.Score, c.Text)
	}
	ctxText := sb.String()

	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: systemPrompt},
		{Role: llm.RoleUser, Content: fmt.Sprintf(
			"Context:\n%s\nQuestion: %s", ctxText, question,
		)},
	}
	answer, err := p.lm.Complete(ctx, messages)
	if err != nil {
		return QueryResult{}, fmt.Errorf("sentence_window/query: generate: %w", err)
	}

	return QueryResult{Answer: answer, Context: ctxText}, nil
}
