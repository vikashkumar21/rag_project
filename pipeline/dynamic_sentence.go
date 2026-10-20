package pipeline

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/vikash/rag_project/internal/embedder"
	"github.com/vikash/rag_project/internal/llm"
	"github.com/vikash/rag_project/internal/loader"
	"github.com/vikash/rag_project/internal/retriever"
	"github.com/vikash/rag_project/internal/vectorstore"
)

// DynamicSentenceRAG coordinates indexing sentences once into a SentenceCorpus,
// embedding focus sentences into the vector store without redundant metadata,
// and dynamically slicing windows at query time.
type DynamicSentenceRAG struct {
	ret               *retriever.DynamicSentenceRetriever
	lm                llm.LLM
	defaultWindowSize int
}

// DynamicSentenceConfig configures the DynamicSentenceRAG pipeline.
type DynamicSentenceConfig struct {
	Embedder          embedder.Embedder
	Store             vectorstore.Store
	Corpus            *loader.SentenceCorpus
	LLM               llm.LLM
	Reranker          retriever.Reranker
	SimilarityTopK    int
	RerankTopN        int
	DefaultWindowSize int
	SaveDir           string // Directory to persist/load both corpus.json and vectors.json
}

// BuildDynamicSentence constructs or loads a DynamicSentenceRAG pipeline.
func BuildDynamicSentence(ctx context.Context, docs []loader.Document, cfg DynamicSentenceConfig) (*DynamicSentenceRAG, error) {
	if cfg.SimilarityTopK <= 0 {
		cfg.SimilarityTopK = 6
	}
	if cfg.RerankTopN <= 0 {
		cfg.RerankTopN = 2
	}
	if cfg.DefaultWindowSize <= 0 {
		cfg.DefaultWindowSize = 3
	}

	// 1. Check if persistent index and corpus exist on disk
	if cfg.SaveDir != "" && vectorstore.Exists(cfg.SaveDir) && loader.CorpusExists(cfg.SaveDir) {
		diskStore, sErr := vectorstore.LoadFromDisk(cfg.SaveDir)
		diskCorpus, cErr := loader.LoadCorpusFromDisk(cfg.SaveDir)
		if sErr == nil && cErr == nil {
			fmt.Printf("[dynamic-rag] loaded %d cached vectors and corpus from %s\n", diskStore.Len(), cfg.SaveDir)
			cfg.Store = diskStore
			cfg.Corpus = diskCorpus
			ret := retriever.NewDynamicSentence(retriever.DynamicSentenceConfig{
				Embedder:          cfg.Embedder,
				Store:             cfg.Store,
				Corpus:            cfg.Corpus,
				Reranker:          cfg.Reranker,
				SimilarityTopK:    cfg.SimilarityTopK,
				RerankTopN:        cfg.RerankTopN,
				DefaultWindowSize: cfg.DefaultWindowSize,
			})
			return &DynamicSentenceRAG{
				ret:               ret,
				lm:                cfg.LLM,
				defaultWindowSize: cfg.DefaultWindowSize,
			}, nil
		}
		fmt.Printf("[dynamic-rag] warning: failed to load %s, rebuilding...\n", cfg.SaveDir)
	}

	if cfg.Store == nil {
		cfg.Store = vectorstore.NewInMemory()
	}
	if cfg.Corpus == nil {
		cfg.Corpus = loader.NewSentenceCorpus()
	}

	// 2. Populate Corpus and prepare focus sentences to embed
	type sentenceEntry struct {
		docID string
		idx   int
		text  string
	}
	var allEntries []sentenceEntry

	for _, doc := range docs {
		sentences := cfg.Corpus.AddDocument(doc)
		for idx, s := range sentences {
			allEntries = append(allEntries, sentenceEntry{
				docID: doc.ID,
				idx:   idx,
				text:  s,
			})
		}
	}

	fmt.Printf("[dynamic-rag] segmented %d document(s) into %d sentences (stored once in corpus)\n",
		len(docs), len(allEntries))

	// 3. Embed focus sentences in batches
	const batchSize = 256
	for batchStart := 0; batchStart < len(allEntries); batchStart += batchSize {
		end := batchStart + batchSize
		if end > len(allEntries) {
			end = len(allEntries)
		}
		batch := allEntries[batchStart:end]

		texts := make([]string, len(batch))
		for i, e := range batch {
			texts[i] = e.text
		}

		embeddings, err := cfg.Embedder.Embed(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("dynamic-rag/build: embed batch at %d: %w", batchStart, err)
		}

		vecs := make([]vectorstore.Vector, len(batch))
		for i, e := range batch {
			vecs[i] = vectorstore.Vector{
				ID:        fmt.Sprintf("%s#s-%d", e.docID, e.idx),
				Text:      e.text,
				Embedding: embeddings[i],
				Metadata: map[string]string{
					"doc_id":       e.docID,
					"sentence_idx": strconv.Itoa(e.idx),
				},
			}
		}

		if err := cfg.Store.Add(ctx, vecs); err != nil {
			return nil, fmt.Errorf("dynamic-rag/build: store batch: %w", err)
		}
		fmt.Printf("[dynamic-rag] indexed %d / %d sentences\n", end, len(allEntries))
	}

	// 4. Persist to disk if SaveDir is configured
	if cfg.SaveDir != "" {
		if memStore, ok := cfg.Store.(*vectorstore.InMemoryStore); ok {
			_ = memStore.Save(cfg.SaveDir)
		}
		_ = cfg.Corpus.Save(cfg.SaveDir)
		fmt.Printf("[dynamic-rag] persisted index and corpus to %s\n", cfg.SaveDir)
	}

	// 5. Assemble retriever
	ret := retriever.NewDynamicSentence(retriever.DynamicSentenceConfig{
		Embedder:          cfg.Embedder,
		Store:             cfg.Store,
		Corpus:            cfg.Corpus,
		Reranker:          cfg.Reranker,
		SimilarityTopK:    cfg.SimilarityTopK,
		RerankTopN:        cfg.RerankTopN,
		DefaultWindowSize: cfg.DefaultWindowSize,
	})

	return &DynamicSentenceRAG{
		ret:               ret,
		lm:                cfg.LLM,
		defaultWindowSize: cfg.DefaultWindowSize,
	}, nil
}

// Query performs retrieval with the specified windowSize and generates an answer.
// If windowSize <= 0, the defaultWindowSize configured on the pipeline is used.
func (p *DynamicSentenceRAG) Query(ctx context.Context, question string, windowSize int) (QueryResult, error) {
	if windowSize <= 0 {
		windowSize = p.defaultWindowSize
	}

	chunks, err := p.ret.RetrieveWithWindow(ctx, question, windowSize)
	if err != nil {
		return QueryResult{}, fmt.Errorf("dynamic_rag/query: retrieve: %w", err)
	}

	var sb strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&sb, "--- Dynamic Window Context %d (score %.3f) ---\n%s\n\n",
			i+1, c.Score, c.Text)
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
		return QueryResult{}, fmt.Errorf("dynamic_rag/query: generate: %w", err)
	}

	return QueryResult{Answer: answer, Context: ctxText}, nil
}
