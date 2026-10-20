package retriever_test

import (
	"context"
	"strconv"
	"testing"

	"github.com/vikash/rag_project/internal/loader"
	"github.com/vikash/rag_project/internal/retriever"
	"github.com/vikash/rag_project/internal/vectorstore"
)

// mockEmbedder returns synthetic 1D vector based on text length or index.
type mockEmbedder struct{}

func (m *mockEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = []float32{float32(len(t))}
	}
	return out, nil
}

func TestDynamicSentenceRetriever_VariableWindowAndMerging(t *testing.T) {
	ctx := context.Background()
	corpus := loader.NewSentenceCorpus()

	docID := "doc-ai"
	sentences := []string{
		"Sentence 0.",
		"Sentence 1.",
		"Sentence 2.",
		"Sentence 3.",
		"Sentence 4.",
		"Sentence 5.",
		"Sentence 6.",
	}
	corpus.SetSentences(docID, sentences)

	store := vectorstore.NewInMemory()
	// Store sentence 3 and sentence 4 as hits
	vecs := []vectorstore.Vector{
		{
			ID:        "doc-ai#s-3",
			Text:      "Sentence 3.",
			Embedding: []float32{10.0},
			Metadata:  map[string]string{"doc_id": docID, "sentence_idx": "3"},
		},
		{
			ID:        "doc-ai#s-4",
			Text:      "Sentence 4.",
			Embedding: []float32{9.9},
			Metadata:  map[string]string{"doc_id": docID, "sentence_idx": "4"},
		},
	}
	for i := 0; i < len(sentences); i++ {
		if i != 3 && i != 4 {
			vecs = append(vecs, vectorstore.Vector{
				ID:        "doc-ai#s-" + strconv.Itoa(i),
				Text:      sentences[i],
				Embedding: []float32{1.0},
				Metadata:  map[string]string{"doc_id": docID, "sentence_idx": strconv.Itoa(i)},
			})
		}
	}
	_ = store.Add(ctx, vecs)

	ret := retriever.NewDynamicSentence(retriever.DynamicSentenceConfig{
		Embedder:       &mockEmbedder{},
		Store:          store,
		Corpus:         corpus,
		SimilarityTopK: 2, // will hit sentence 3 and sentence 4
		RerankTopN:     2,
	})

	// Query with WindowSize = 1:
	// Sentence 3 window: [2, 4]
	// Sentence 4 window: [3, 5]
	// Merged interval: [2, 5] -> "Sentence 2. Sentence 3. Sentence 4. Sentence 5."
	chunksWin1, err := ret.RetrieveWithWindow(ctx, "query", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunksWin1) != 1 {
		t.Fatalf("expected 1 merged chunk, got %d", len(chunksWin1))
	}
	expectedWin1 := "Sentence 2. Sentence 3. Sentence 4. Sentence 5."
	if chunksWin1[0].Text != expectedWin1 {
		t.Errorf("expected %q, got %q", expectedWin1, chunksWin1[0].Text)
	}

	// Query with WindowSize = 2 on the SAME index:
	// Sentence 3 window: [1, 5]
	// Sentence 4 window: [2, 6]
	// Merged interval: [1, 6] -> "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6."
	chunksWin2, err := ret.RetrieveWithWindow(ctx, "query", 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunksWin2) != 1 {
		t.Fatalf("expected 1 merged chunk, got %d", len(chunksWin2))
	}
	expectedWin2 := "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6."
	if chunksWin2[0].Text != expectedWin2 {
		t.Errorf("expected %q, got %q", expectedWin2, chunksWin2[0].Text)
	}
}
