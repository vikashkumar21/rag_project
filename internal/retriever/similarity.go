package retriever

import (
	"context"
	"fmt"

	"github.com/vikash/rag_project/internal/embedder"
	"github.com/vikash/rag_project/internal/vectorstore"
)

// SimilarityRetriever embeds the query with an Embedder, then looks up the
// topK most similar vectors in the Store.
//
// This is the "baseline" retrieval strategy — the starting point before we
// introduce sentence-window and auto-merging in later lessons.
//
// Python equivalent:
//
//	index = VectorStoreIndex.from_documents(...)
//	query_engine = index.as_query_engine()
//	# internally: embed query → cosine search → top-k chunks
type SimilarityRetriever struct {
	embedder embedder.Embedder
	store    vectorstore.Store
}

// NewSimilarity returns a SimilarityRetriever backed by the given components.
func NewSimilarity(e embedder.Embedder, s vectorstore.Store) *SimilarityRetriever {
	return &SimilarityRetriever{embedder: e, store: s}
}

// Retrieve embeds the query and returns the topK most similar stored chunks.
func (r *SimilarityRetriever) Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error) {
	// 1. Embed the query using the same model used for documents.
	//    It's crucial to use the SAME model for both — mismatching models
	//    produces meaningless similarity scores.
	vecs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("retriever/similarity: embed query: %w", err)
	}

	// 2. Nearest-neighbour search in the vector store.
	results, err := r.store.Search(ctx, vecs[0], topK)
	if err != nil {
		return nil, fmt.Errorf("retriever/similarity: search: %w", err)
	}

	// 3. Map store results to the generic Chunk type.
	chunks := make([]Chunk, len(results))
	for i, res := range results {
		chunks[i] = Chunk{
			ID:    res.ID,
			Text:  res.Text,
			Score: res.Score,
		}
	}
	return chunks, nil
}
