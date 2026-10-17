package embedder

import (
	"context"
	"fmt"

	oai "github.com/sashabaranov/go-openai"
)

// OpenAIEmbedder calls the OpenAI embeddings endpoint.
//
// Python equivalent:
//
//	embed_model="local:BAAI/bge-small-en-v1.5"   (inside ServiceContext)
//
// We use OpenAI's "text-embedding-3-small" instead of the local BAAI model
// because Go doesn't have a mature ONNX runtime ecosystem yet.
// text-embedding-3-small is cheap (~$0.02 / 1M tokens) and high-quality.
//
// Model options:
//   - "text-embedding-3-small"  – cheapest, 1536 dims
//   - "text-embedding-3-large"  – highest quality, 3072 dims
//   - "text-embedding-ada-002"  – legacy, still widely used
type OpenAIEmbedder struct {
	client *oai.Client
	model  string
}

// NewOpenAI returns an OpenAIEmbedder for the named model.
func NewOpenAI(apiKey, model string) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		client: oai.NewClient(apiKey),
		model:  model,
	}
}

// Embed returns one float32 vector per input text.
// The order of the output matches the order of the input.
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	resp, err := e.client.CreateEmbeddings(ctx, oai.EmbeddingRequestStrings{
		Input: texts,
		Model: oai.EmbeddingModel(e.model),
	})
	if err != nil {
		return nil, fmt.Errorf("embedder/openai: %w", err)
	}

	// OpenAI returns embeddings in the same order as the input.
	out := make([][]float32, len(resp.Data))
	for _, d := range resp.Data {
		out[d.Index] = d.Embedding
	}
	return out, nil
}
