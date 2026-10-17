package retriever

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/vikash/rag_project/internal/llm"
)

// Reranker takes an initial list of candidate chunks and re-scores them against
// the user query, returning the topN most relevant chunks.
//
// Python equivalent:
//
//	from llama_index.indices.postprocessor import SentenceTransformerRerank
//	rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")
type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []Chunk, topN int) ([]Chunk, error)
}

// LLMReranker uses an LLM to evaluate the relevance of each candidate chunk
// with respect to the query, providing a cross-attention score similar to
// cross-encoder models like BAAI/bge-reranker-base.
type LLMReranker struct {
	client llm.LLM
}

// NewLLMReranker creates a new LLMReranker.
func NewLLMReranker(client llm.LLM) *LLMReranker {
	return &LLMReranker{client: client}
}

type scoredChunk struct {
	chunk Chunk
	score float64
}

// Rerank evaluates candidate chunks and returns the topN candidates sorted by LLM score.
func (r *LLMReranker) Rerank(ctx context.Context, query string, candidates []Chunk, topN int) ([]Chunk, error) {
	if len(candidates) == 0 || topN <= 0 {
		return nil, nil
	}
	if topN > len(candidates) {
		topN = len(candidates)
	}

	scored := make([]scoredChunk, 0, len(candidates))
	for _, c := range candidates {
		prompt := fmt.Sprintf(`You are a search relevance scoring engine.
Rate how relevant the following passage is to answering the user query.

Query: %s

Passage:
%s

Respond with ONLY a number from 0.0 (irrelevant) to 10.0 (perfectly answers the query).`, query, c.Text)

		ans, err := r.client.Complete(ctx, []llm.Message{
			{Role: llm.RoleSystem, Content: "You are an objective relevance ranker. Return ONLY a decimal score between 0.0 and 10.0."},
			{Role: llm.RoleUser, Content: prompt},
		})
		score := 0.0
		if err == nil {
			var s float64
			if _, pErr := fmt.Sscanf(strings.TrimSpace(ans), "%f", &s); pErr == nil {
				score = s
			}
		}

		cCopy := c
		cCopy.Score = float32(score / 10.0)
		scored = append(scored, scoredChunk{chunk: cCopy, score: score})
	}

	// Sort descending by score
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	out := make([]Chunk, topN)
	for i := 0; i < topN; i++ {
		out[i] = scored[i].chunk
	}
	return out, nil
}

// PassThroughReranker returns the first topN candidates without altering them.
type PassThroughReranker struct{}

// Rerank truncates candidates to topN.
func (p PassThroughReranker) Rerank(_ context.Context, _ string, candidates []Chunk, topN int) ([]Chunk, error) {
	if topN > len(candidates) {
		topN = len(candidates)
	}
	return candidates[:topN], nil
}
