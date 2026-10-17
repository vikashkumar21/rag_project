// Package retriever fetches relevant text chunks for a natural-language query.
// It defines the Retriever interface and several implementations that we'll
// build up lesson by lesson (baseline similarity → sentence-window → auto-merging).
package retriever

import "context"

// Chunk is a retrieved piece of text paired with its relevance score.
// Score is provider-defined; for cosine similarity it is in [0, 1].
type Chunk struct {
	ID    string
	Text  string
	Score float32
}

// Retriever fetches the most relevant Chunks for a natural-language query.
//
// Python equivalent:
//
//	index.as_query_engine()   (the retrieval half of the query engine)
//
// Implementations must be safe for concurrent use.
type Retriever interface {
	Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error)
}
