package retriever

import (
	"context"
	"fmt"

	"github.com/vikash/rag_project/internal/embedder"
	"github.com/vikash/rag_project/internal/vectorstore"
)

// AutoMergingRetriever retrieves fine-grained leaf nodes and merges them
// into larger parent chunks if enough siblings from the same parent are retrieved.
//
// Python equivalent:
//
//	retriever = AutoMergingRetriever(
//	    base_retriever, automerging_index.storage_context, verbose=True
//	)
type AutoMergingRetriever struct {
	embedder       embedder.Embedder
	store          vectorstore.Store
	docStore       vectorstore.DocStore
	reranker       Reranker
	similarityTopK int
	rerankTopN     int
	threshold      float64 // Minimum ratio of retrieved children needed to merge into parent (e.g. 0.5)
	verbose        bool
}

// AutoMergingConfig configures the AutoMergingRetriever.
type AutoMergingConfig struct {
	Embedder       embedder.Embedder
	Store          vectorstore.Store
	DocStore       vectorstore.DocStore
	Reranker       Reranker
	SimilarityTopK int     // Default: 12
	RerankTopN     int     // Default: 2
	Threshold      float64 // Default: 0.5 (50% of children)
	Verbose        bool
}

// NewAutoMerging returns a new AutoMergingRetriever.
func NewAutoMerging(cfg AutoMergingConfig) *AutoMergingRetriever {
	if cfg.SimilarityTopK <= 0 {
		cfg.SimilarityTopK = 12
	}
	if cfg.RerankTopN <= 0 {
		cfg.RerankTopN = 2
	}
	if cfg.Threshold <= 0 {
		cfg.Threshold = 0.5
	}
	if cfg.Reranker == nil {
		cfg.Reranker = PassThroughReranker{}
	}

	return &AutoMergingRetriever{
		embedder:       cfg.Embedder,
		store:          cfg.Store,
		docStore:       cfg.DocStore,
		reranker:       cfg.Reranker,
		similarityTopK: cfg.SimilarityTopK,
		rerankTopN:     cfg.RerankTopN,
		threshold:      cfg.Threshold,
		verbose:        cfg.Verbose,
	}
}

type activeNode struct {
	id    string
	score float32
}

// Retrieve performs leaf-node retrieval, recursive auto-merging, and reranking.
func (r *AutoMergingRetriever) Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error) {
	k := r.similarityTopK
	if topK > 0 {
		k = topK
	}

	// 1. Embed query
	qVecs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("automerging: embed query: %w", err)
	}

	// 2. Retrieve top-k leaf nodes from vector store
	leafResults, err := r.store.Search(ctx, qVecs[0], k)
	if err != nil {
		return nil, fmt.Errorf("automerging: search leaves: %w", err)
	}

	// 3. Populate initial active set
	currentNodes := make(map[string]activeNode)
	for _, res := range leafResults {
		currentNodes[res.ID] = activeNode{id: res.ID, score: res.Score}
	}

	// 4. Recursive merging loop: check parents and merge upwards
	for {
		mergedAny := false

		// Collect unique parent IDs for all currently active nodes
		parents := make(map[string]bool)
		for nodeID := range currentNodes {
			node, ok := r.docStore.Get(nodeID)
			if ok && node.ParentID != "" {
				parents[node.ParentID] = true
			}
		}

		for parentID := range parents {
			parent, ok := r.docStore.Get(parentID)
			if !ok || len(parent.ChildIDs) == 0 {
				continue
			}

			// Count how many children of this parent are in currentNodes
			var matchedChildren []string
			var scoreSum float32
			for _, childID := range parent.ChildIDs {
				if act, exists := currentNodes[childID]; exists {
					matchedChildren = append(matchedChildren, childID)
					scoreSum += act.score
				}
			}

			ratio := float64(len(matchedChildren)) / float64(len(parent.ChildIDs))
			if ratio >= r.threshold {
				if r.verbose {
					fmt.Printf("[automerging] merging %d/%d children into parent %s\n",
						len(matchedChildren), len(parent.ChildIDs), parentID)
				}

				// Evict children
				for _, cid := range matchedChildren {
					delete(currentNodes, cid)
				}

				// Insert parent with averaged score
				avgScore := scoreSum / float32(len(matchedChildren))
				currentNodes[parentID] = activeNode{id: parentID, score: avgScore}
				mergedAny = true
			}
		}

		// If no merges happened in this pass, we are done
		if !mergedAny {
			break
		}
	}

	// 5. Build candidate chunks from docStore
	candidates := make([]Chunk, 0, len(currentNodes))
	for id, act := range currentNodes {
		node, ok := r.docStore.Get(id)
		text := ""
		if ok {
			text = node.Text
		}
		candidates = append(candidates, Chunk{
			ID:    id,
			Text:  text,
			Score: act.score,
		})
	}

	// 6. Rerank down to rerankTopN
	finalChunks, err := r.reranker.Rerank(ctx, query, candidates, r.rerankTopN)
	if err != nil {
		return nil, fmt.Errorf("automerging: rerank: %w", err)
	}

	return finalChunks, nil
}
