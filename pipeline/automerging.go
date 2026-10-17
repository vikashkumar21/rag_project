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

// AutoMergingRAG coordinates hierarchical indexing, leaf vector search,
// dynamic child-to-parent merging, and response generation.
//
// Python equivalent:
//
//	automerging_index = build_automerging_index(...)
//	automerging_query_engine = get_automerging_query_engine(automerging_index)
type AutoMergingRAG struct {
	ret retriever.Retriever
	lm  llm.LLM
}

// AutoMergingConfig configures the AutoMerging pipeline.
type AutoMergingConfig struct {
	Embedder       embedder.Embedder
	Store          vectorstore.Store
	DocStore       vectorstore.DocStore
	LLM            llm.LLM
	Reranker       retriever.Reranker
	SimilarityTopK int
	RerankTopN     int
	Threshold      float64
	Verbose        bool
	HierarchyOpts  loader.HierarchicalOptions
}

// BuildAutoMerging creates and indexes an auto-merging RAG pipeline.
func BuildAutoMerging(ctx context.Context, docs []loader.Document, cfg AutoMergingConfig) (*AutoMergingRAG, error) {
	if cfg.SimilarityTopK <= 0 {
		cfg.SimilarityTopK = 12
	}
	if cfg.RerankTopN <= 0 {
		cfg.RerankTopN = 2
	}
	if cfg.Threshold <= 0 {
		cfg.Threshold = 0.5
	}
	if len(cfg.HierarchyOpts.ChunkSizes) == 0 {
		cfg.HierarchyOpts = loader.DefaultHierarchicalOptions()
	}
	if cfg.DocStore == nil {
		cfg.DocStore = vectorstore.NewInMemoryDocStore()
	}

	// 1. Build hierarchy for all documents
	var allNodes []vectorstore.HierarchyNode
	var leafNodes []vectorstore.HierarchyNode

	for _, d := range docs {
		nodes, leaves := loader.BuildHierarchy(d, cfg.HierarchyOpts)
		allNodes = append(allNodes, nodes...)
		leafNodes = append(leafNodes, leaves...)
	}

	fmt.Printf("[automerging] created %d total hierarchy nodes (%d leaf nodes to embed)\n",
		len(allNodes), len(leafNodes))

	// 2. Add all nodes to DocStore
	cfg.DocStore.Add(allNodes)

	// 3. Embed and store ONLY the leaf nodes
	const batchSize = 256
	for batchStart := 0; batchStart < len(leafNodes); batchStart += batchSize {
		end := batchStart + batchSize
		if end > len(leafNodes) {
			end = len(leafNodes)
		}
		batch := leafNodes[batchStart:end]

		texts := make([]string, len(batch))
		for i, n := range batch {
			texts[i] = n.Text
		}

		embeddings, err := cfg.Embedder.Embed(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("automerging/build: embed batch at %d: %w", batchStart, err)
		}

		vecs := make([]vectorstore.Vector, len(batch))
		for i, n := range batch {
			vecs[i] = vectorstore.Vector{
				ID:        n.ID,
				Text:      n.Text,
				Embedding: embeddings[i],
				Metadata:  n.Metadata,
			}
		}

		if err := cfg.Store.Add(ctx, vecs); err != nil {
			return nil, fmt.Errorf("automerging/build: store batch: %w", err)
		}
		fmt.Printf("[automerging] indexed %d / %d leaf nodes\n", end, len(leafNodes))
	}

	// 4. Assemble AutoMerging retriever
	ret := retriever.NewAutoMerging(retriever.AutoMergingConfig{
		Embedder:       cfg.Embedder,
		Store:          cfg.Store,
		DocStore:       cfg.DocStore,
		Reranker:       cfg.Reranker,
		SimilarityTopK: cfg.SimilarityTopK,
		RerankTopN:     cfg.RerankTopN,
		Threshold:      cfg.Threshold,
		Verbose:        cfg.Verbose,
	})

	return &AutoMergingRAG{
		ret: ret,
		lm:  cfg.LLM,
	}, nil
}

// Query retrieves auto-merged chunks and generates an answer.
func (p *AutoMergingRAG) Query(ctx context.Context, question string) (QueryResult, error) {
	chunks, err := p.ret.Retrieve(ctx, question, 0)
	if err != nil {
		return QueryResult{}, fmt.Errorf("automerging/query: retrieve: %w", err)
	}

	var sb strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&sb, "--- Merged Context %d (score %.3f, id %s) ---\n%s\n\n",
			i+1, c.Score, c.ID, c.Text)
	}
	ctxText := sb.String()

	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: systemPrompt},
		{Role: llm.RoleUser, Content: fmt.Sprintf(
			"Context:\n%s\nQuestion: %s", ctxText, question,
		)},
	}
	answer, err := p.lm.Complete(ctx, messages)
	if err != nil {
		return QueryResult{}, fmt.Errorf("automerging/query: generate: %w", err)
	}

	return QueryResult{Answer: answer, Context: ctxText}, nil
}
