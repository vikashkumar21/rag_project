// Package vectorstore persists dense vectors and supports
// nearest-neighbour retrieval.
package vectorstore

import "context"

// Vector is a stored embedding alongside its source text and metadata.
type Vector struct {
	ID        string
	Text      string
	Embedding []float32
	Metadata  map[string]string
}

// SearchResult pairs a stored Vector with its similarity score.
// Score is in [0, 1] for cosine similarity.
type SearchResult struct {
	Vector
	Score float32
}

// Store persists vectors and supports nearest-neighbour lookup.
//
// Python equivalent:
//
//	VectorStoreIndex  (LlamaIndex)
//
// Implementations must be safe for concurrent use.
type Store interface {
	// Add appends vectors to the store.
	Add(ctx context.Context, vectors []Vector) error

	// Search returns the topK vectors most similar to query,
	// sorted by descending score.
	Search(ctx context.Context, query []float32, topK int) ([]SearchResult, error)

	// Len returns the current number of stored vectors.
	Len() int
}
