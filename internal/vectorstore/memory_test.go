package vectorstore_test

import (
	"context"
	"testing"

	"github.com/vikash/rag_project/internal/vectorstore"
)

func TestInMemoryStoreCosineSearch(t *testing.T) {
	ctx := context.Background()
	store := vectorstore.NewInMemory()

	// Vector 1 points along the X axis
	// Vector 2 points at 45 degrees between X and Y
	// Vector 3 points along the Y axis
	vecs := []vectorstore.Vector{
		{ID: "x-axis", Text: "X axis doc", Embedding: []float32{1.0, 0.0}},
		{ID: "diag", Text: "Diagonal doc", Embedding: []float32{0.7071, 0.7071}},
		{ID: "y-axis", Text: "Y axis doc", Embedding: []float32{0.0, 1.0}},
	}

	if err := store.Add(ctx, vecs); err != nil {
		t.Fatalf("failed to add vectors: %v", err)
	}

	if store.Len() != 3 {
		t.Fatalf("expected store len 3, got %d", store.Len())
	}

	// Query along X axis: [1.0, 0.0]
	results, err := store.Search(ctx, []float32{1.0, 0.0}, 2)
	if err != nil {
		t.Fatalf("failed to search: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// 1st should be x-axis with score ~1.0
	if results[0].ID != "x-axis" {
		t.Errorf("expected top result 'x-axis', got %s", results[0].ID)
	}
	if results[0].Score < 0.99 {
		t.Errorf("expected score ~1.0, got %f", results[0].Score)
	}

	// 2nd should be diagonal with score ~0.707
	if results[1].ID != "diag" {
		t.Errorf("expected 2nd result 'diag', got %s", results[1].ID)
	}
}
