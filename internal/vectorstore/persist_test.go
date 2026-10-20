package vectorstore_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/vikash/rag_project/internal/vectorstore"
)

func TestPersistAndLoad(t *testing.T) {
	ctx := context.Background()
	tmpDir, err := os.MkdirTemp("", "rag-vectorstore-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	storePath := filepath.Join(tmpDir, "test_index")

	if vectorstore.Exists(storePath) {
		t.Errorf("expected Exists to be false before saving")
	}

	store := vectorstore.NewInMemory()
	vecs := []vectorstore.Vector{
		{ID: "v1", Text: "Hello world", Embedding: []float32{0.1, 0.2, 0.3}, Metadata: map[string]string{"k": "v"}},
	}
	if err := store.Add(ctx, vecs); err != nil {
		t.Fatalf("failed to add vectors: %v", err)
	}

	if err := store.Save(storePath); err != nil {
		t.Fatalf("failed to save store: %v", err)
	}

	if !vectorstore.Exists(storePath) {
		t.Errorf("expected Exists to be true after saving")
	}

	loadedStore, err := vectorstore.LoadFromDisk(storePath)
	if err != nil {
		t.Fatalf("failed to load store: %v", err)
	}

	if loadedStore.Len() != 1 {
		t.Fatalf("expected 1 vector, got %d", loadedStore.Len())
	}

	results, err := loadedStore.Search(ctx, []float32{0.1, 0.2, 0.3}, 1)
	if err != nil {
		t.Fatalf("failed to search: %v", err)
	}
	if len(results) != 1 || results[0].ID != "v1" {
		t.Errorf("unexpected search result: %+v", results)
	}
}
