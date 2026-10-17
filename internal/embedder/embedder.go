// Package embedder converts text into dense vector representations.
// It defines the Embedder interface so the rest of the codebase never
// depends on a specific provider (OpenAI, Cohere, local models, etc.).
package embedder

import "context"

// Embedder converts a batch of texts into their dense vector representations.
//
// Python equivalent:
//
//	embed_model="local:BAAI/bge-small-en-v1.5"   (inside ServiceContext)
//
// Implementations must be safe for concurrent use.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}
