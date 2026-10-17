package vectorstore

import (
	"context"
	"math"
	"sort"
	"sync"
)

// InMemoryStore holds all vectors in RAM and retrieves by cosine similarity.
//
// Why cosine similarity?
// Embedding models are trained to encode semantic meaning as *direction*, not
// magnitude. Two texts with the same meaning produce vectors that point in the
// same direction even if their magnitudes differ. Cosine similarity captures
// this by measuring the angle between vectors.
//
// InMemoryStore is fine for learning and small datasets. For production you'd
// swap it with a persistent store like Qdrant, Weaviate, or pgvector — the
// Store interface makes this a one-line change.
type InMemoryStore struct {
	mu      sync.RWMutex
	vectors []Vector
}

// NewInMemory returns an empty InMemoryStore.
func NewInMemory() *InMemoryStore {
	return &InMemoryStore{}
}

// Add appends the given vectors to the store.
func (s *InMemoryStore) Add(_ context.Context, vecs []Vector) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.vectors = append(s.vectors, vecs...)
	return nil
}

// Search returns the topK vectors most similar to the query embedding.
// It performs a linear scan — O(n·d) where d = embedding dimension.
// This is fast enough for thousands of chunks; use an ANN index for millions.
func (s *InMemoryStore) Search(_ context.Context, query []float32, topK int) ([]SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	type entry struct {
		idx   int
		score float32
	}
	scored := make([]entry, len(s.vectors))
	for i, v := range s.vectors {
		scored[i] = entry{i, cosine(query, v.Embedding)}
	}
	// Sort descending by score.
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	if topK > len(scored) {
		topK = len(scored)
	}
	results := make([]SearchResult, topK)
	for i := range results {
		results[i] = SearchResult{
			Vector: s.vectors[scored[i].idx],
			Score:  scored[i].score,
		}
	}
	return results, nil
}

// Len returns the number of vectors currently in the store.
func (s *InMemoryStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.vectors)
}

// cosine computes the cosine similarity between two equal-length vectors.
// Returns 0 for zero-magnitude or mismatched vectors.
func cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}
