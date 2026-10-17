package retriever

import (
	"context"
	"fmt"

	"github.com/vikash/rag_project/internal/embedder"
	"github.com/vikash/rag_project/internal/vectorstore"
)

// SentenceWindowRetriever implements the sentence-window retrieval strategy:
// 1. Searches vectors using similarity on the focus sentence.
// 2. Replaces the sentence text with its expanded window context (Metadata Replacement).
// 3. Applies a reranker to select the top-N most coherent windows.
//
// Python equivalent:
//
//	sentence_window_engine = get_sentence_window_query_engine(
//	    sentence_index,
//	    similarity_top_k=6,
//	    rerank_top_n=2,
//	)
type SentenceWindowRetriever struct {
	embedder        embedder.Embedder
	store           vectorstore.Store
	reranker        Reranker
	similarityTopK  int
	rerankTopN      int
	metaReplacement string
}

// SentenceWindowConfig configures the SentenceWindowRetriever.
type SentenceWindowConfig struct {
	Embedder        embedder.Embedder
	Store           vectorstore.Store
	Reranker        Reranker
	SimilarityTopK  int
	RerankTopN      int
	MetaReplacement string // Default "window"
}

// NewSentenceWindow creates a new SentenceWindowRetriever.
func NewSentenceWindow(cfg SentenceWindowConfig) *SentenceWindowRetriever {
	if cfg.SimilarityTopK <= 0 {
		cfg.SimilarityTopK = 6
	}
	if cfg.RerankTopN <= 0 {
		cfg.RerankTopN = 2
	}
	if cfg.MetaReplacement == "" {
		cfg.MetaReplacement = "window"
	}
	if cfg.Reranker == nil {
		cfg.Reranker = PassThroughReranker{}
	}

	return &SentenceWindowRetriever{
		embedder:        cfg.Embedder,
		store:           cfg.Store,
		reranker:        cfg.Reranker,
		similarityTopK:  cfg.SimilarityTopK,
		rerankTopN:      cfg.RerankTopN,
		metaReplacement: cfg.MetaReplacement,
	}
}

// Retrieve implements Retriever.
func (r *SentenceWindowRetriever) Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error) {
	// If caller specified topK > 0, we can use it to override similarityTopK
	k := r.similarityTopK
	if topK > 0 {
		k = topK
	}

	// 1. Embed query
	qVecs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("sentence_window: embed query: %w", err)
	}

	// 2. Nearest-neighbour search on focus sentences
	results, err := r.store.Search(ctx, qVecs[0], k)
	if err != nil {
		return nil, fmt.Errorf("sentence_window: search: %w", err)
	}

	// 3. Metadata replacement: swap focus sentence for surrounding window
	candidates := make([]Chunk, len(results))
	for i, res := range results {
		chunkText := res.Text
		if windowText, ok := res.Metadata[r.metaReplacement]; ok && windowText != "" {
			chunkText = windowText
		}
		candidates[i] = Chunk{
			ID:    res.ID,
			Text:  chunkText,
			Score: res.Score,
		}
	}

	// 4. Deduplicate overlapping windows if any
	candidates = deduplicateChunks(candidates)

	// 5. Rerank down to rerankTopN
	finalChunks, err := r.reranker.Rerank(ctx, query, candidates, r.rerankTopN)
	if err != nil {
		return nil, fmt.Errorf("sentence_window: rerank: %w", err)
	}

	return finalChunks, nil
}

func deduplicateChunks(chunks []Chunk) []Chunk {
	seen := make(map[string]bool)
	unique := make([]Chunk, 0, len(chunks))
	for _, c := range chunks {
		if !seen[c.Text] {
			seen[c.Text] = true
			unique = append(unique, c)
		}
	}
	return unique
}
