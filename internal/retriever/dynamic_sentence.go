package retriever

import (
	"context"
	"fmt"
	"strconv"

	"github.com/vikash/rag_project/internal/embedder"
	"github.com/vikash/rag_project/internal/loader"
	"github.com/vikash/rag_project/internal/vectorstore"
)

// DynamicSentenceRetriever searches focused sentence embeddings in the vector store,
// and dynamically slices and merges surrounding context windows from the SentenceCorpus
// on-the-fly at query time.
type DynamicSentenceRetriever struct {
	embedder          embedder.Embedder
	store             vectorstore.Store
	corpus            *loader.SentenceCorpus
	reranker          Reranker
	similarityTopK    int
	rerankTopN        int
	defaultWindowSize int
}

// DynamicSentenceConfig configures the DynamicSentenceRetriever.
type DynamicSentenceConfig struct {
	Embedder          embedder.Embedder
	Store             vectorstore.Store
	Corpus            *loader.SentenceCorpus
	Reranker          Reranker
	SimilarityTopK    int
	RerankTopN        int
	DefaultWindowSize int
}

// NewDynamicSentence creates a new DynamicSentenceRetriever.
func NewDynamicSentence(cfg DynamicSentenceConfig) *DynamicSentenceRetriever {
	if cfg.SimilarityTopK <= 0 {
		cfg.SimilarityTopK = 6
	}
	if cfg.RerankTopN <= 0 {
		cfg.RerankTopN = 2
	}
	if cfg.DefaultWindowSize <= 0 {
		cfg.DefaultWindowSize = 3
	}
	if cfg.Reranker == nil {
		cfg.Reranker = PassThroughReranker{}
	}

	return &DynamicSentenceRetriever{
		embedder:          cfg.Embedder,
		store:             cfg.Store,
		corpus:            cfg.Corpus,
		reranker:          cfg.Reranker,
		similarityTopK:    cfg.SimilarityTopK,
		rerankTopN:        cfg.RerankTopN,
		defaultWindowSize: cfg.DefaultWindowSize,
	}
}

// Retrieve implements the standard Retriever interface using the default window size.
func (r *DynamicSentenceRetriever) Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error) {
	return r.RetrieveWithWindow(ctx, query, r.defaultWindowSize)
}

// RetrieveWithWindow dynamically extracts windows of size windowSize around retrieved focus sentences,
// merges overlapping sentence intervals to avoid duplicate prompt context, and reranks the results.
func (r *DynamicSentenceRetriever) RetrieveWithWindow(ctx context.Context, query string, windowSize int) ([]Chunk, error) {
	// 1. Embed query
	qVecs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("dynamic_retriever: embed query: %w", err)
	}

	// 2. Nearest-neighbour search on focus sentence embeddings
	results, err := r.store.Search(ctx, qVecs[0], r.similarityTopK)
	if err != nil {
		return nil, fmt.Errorf("dynamic_retriever: search: %w", err)
	}

	if len(results) == 0 {
		return nil, nil
	}

	// 3. Group hit sentence indices by document ID
	type docHit struct {
		indices []int
		scores  []float32
	}
	docHits := make(map[string]*docHit)

	for _, res := range results {
		docID := res.Metadata["doc_id"]
		idxStr := res.Metadata["sentence_idx"]
		idx, pErr := strconv.Atoi(idxStr)
		if pErr != nil || docID == "" {
			continue
		}

		if _, exists := docHits[docID]; !exists {
			docHits[docID] = &docHit{}
		}
		docHits[docID].indices = append(docHits[docID].indices, idx)
		docHits[docID].scores = append(docHits[docID].scores, res.Score)
	}

	// 4. Dynamically extract and merge sentence window intervals from the corpus
	var candidates []Chunk
	for docID, hit := range docHits {
		mergedPassages, mErr := r.corpus.GetMergedWindows(docID, hit.indices, windowSize)
		if mErr != nil {
			return nil, fmt.Errorf("dynamic_retriever: get merged windows for %s: %w", docID, mErr)
		}

		for i, passage := range mergedPassages {
			score := float32(0.0)
			if i < len(hit.scores) {
				score = hit.scores[i]
			}
			candidates = append(candidates, Chunk{
				ID:    fmt.Sprintf("%s#merged-win-%d", docID, i),
				Text:  passage,
				Score: score,
			})
		}
	}

	// 5. Rerank down to rerankTopN
	finalChunks, err := r.reranker.Rerank(ctx, query, candidates, r.rerankTopN)
	if err != nil {
		return nil, fmt.Errorf("dynamic_retriever: rerank: %w", err)
	}

	return finalChunks, nil
}
