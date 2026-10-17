// Package pipeline assembles retrieval + generation components into runnable
// RAG pipelines. Lesson 1 provides the Baseline pipeline; later lessons will
// add SentenceWindow and AutoMerging pipelines in separate files.
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

// systemPrompt instructs the LLM to stay grounded in the provided context.
// This wording is important: it prevents hallucination by telling the model
// to answer ONLY from what it's given (improving Groundedness scores).
const systemPrompt = `You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer confidently, say so honestly.`

// BaselineRAG is the simplest possible RAG pipeline:
//
//	documents → chunk → embed → index
//	query     → embed → top-k search → LLM(query + chunks) → answer
//
// Python equivalent (all done implicitly by LlamaIndex):
//
//	index = VectorStoreIndex.from_documents([document], service_context=...)
//	query_engine = index.as_query_engine()
//	response = query_engine.query("...")
type BaselineRAG struct {
	ret  retriever.Retriever
	lm   llm.LLM
	topK int
}

// Config wires the external components into a BaselineRAG.
// This mirrors LlamaIndex's ServiceContext but is explicit about each part.
type Config struct {
	Embedder  embedder.Embedder
	Store     vectorstore.Store
	LLM       llm.LLM
	TopK      int               // chunks to retrieve per query (default 6)
	ChunkOpts loader.ChunkOptions // how to split documents
}

// QueryResult holds both the answer and the context that produced it.
// Having the context lets the caller run RAG Triad evaluation.
type QueryResult struct {
	Answer  string
	Context string // concatenated retrieved chunks (raw text)
}

// Build indexes all documents and returns a ready-to-query BaselineRAG.
//
// Steps:
//  1. Chunk every document into overlapping windows.
//  2. Embed each chunk (batched to respect API rate limits).
//  3. Store vectors in the vector store.
//  4. Wire up a SimilarityRetriever over the populated store.
func Build(ctx context.Context, docs []loader.Document, cfg Config) (*BaselineRAG, error) {
	// Apply defaults.
	if cfg.TopK == 0 {
		cfg.TopK = 6
	}
	if cfg.ChunkOpts.ChunkSize == 0 {
		cfg.ChunkOpts = loader.DefaultChunkOptions()
	}

	// ── Step 1: Chunk ─────────────────────────────────────────────────────────
	var chunks []loader.Document
	for _, d := range docs {
		chunks = append(chunks, loader.Chunk(d, cfg.ChunkOpts)...)
	}
	fmt.Printf("[baseline] %d document(s) → %d chunks (size=%d, overlap=%d)\n",
		len(docs), len(chunks), cfg.ChunkOpts.ChunkSize, cfg.ChunkOpts.ChunkOverlap)

	// ── Step 2 & 3: Embed + Store (batched) ──────────────────────────────────
	// OpenAI allows up to 2048 texts per embedding request.
	// We use 256 as a conservative batch size.
	const batchSize = 256
	for batchStart := 0; batchStart < len(chunks); batchStart += batchSize {
		end := batchStart + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[batchStart:end]

		texts := make([]string, len(batch))
		for i, c := range batch {
			texts[i] = c.Text
		}

		embeddings, err := cfg.Embedder.Embed(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("baseline/build: embed batch at %d: %w", batchStart, err)
		}

		vecs := make([]vectorstore.Vector, len(batch))
		for i, c := range batch {
			vecs[i] = vectorstore.Vector{
				ID:        c.ID,
				Text:      c.Text,
				Embedding: embeddings[i],
				Metadata:  c.Metadata,
			}
		}
		if err := cfg.Store.Add(ctx, vecs); err != nil {
			return nil, fmt.Errorf("baseline/build: store batch: %w", err)
		}
		fmt.Printf("[baseline] indexed %d / %d chunks\n", end, len(chunks))
	}

	ret := retriever.NewSimilarity(cfg.Embedder, cfg.Store)
	return &BaselineRAG{ret: ret, lm: cfg.LLM, topK: cfg.TopK}, nil
}

// Query retrieves the topK most relevant chunks for the question and
// generates an answer by passing them as context to the LLM.
func (p *BaselineRAG) Query(ctx context.Context, question string) (QueryResult, error) {
	// ── Retrieve ──────────────────────────────────────────────────────────────
	chunks, err := p.ret.Retrieve(ctx, question, p.topK)
	if err != nil {
		return QueryResult{}, fmt.Errorf("baseline/query: retrieve: %w", err)
	}

	// Build a single context string from all retrieved chunks.
	// We label each chunk so the LLM can reference them.
	var sb strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&sb, "--- Context %d (relevance %.3f) ---\n%s\n\n",
			i+1, c.Score, c.Text)
	}
	ctxText := sb.String()

	// ── Generate ──────────────────────────────────────────────────────────────
	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: systemPrompt},
		{Role: llm.RoleUser, Content: fmt.Sprintf(
			"Context:\n%s\nQuestion: %s", ctxText, question,
		)},
	}
	answer, err := p.lm.Complete(ctx, messages)
	if err != nil {
		return QueryResult{}, fmt.Errorf("baseline/query: generate: %w", err)
	}
	return QueryResult{Answer: answer, Context: ctxText}, nil
}
