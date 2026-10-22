package vectorstore_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vikash/rag_project/internal/vectorstore"
)

func TestPointerBasedDocStore(t *testing.T) {
	docStore := vectorstore.NewInMemoryDocStore()
	docID := "doc-pointer"
	rawText := "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	docStore.AddDocument(docID, rawText)

	nodes := []vectorstore.HierarchyNode{
		{
			ID:          "p1",
			DocID:       docID,
			StartOffset: 0,
			EndOffset:   20,
			Level:       0,
			ChildIDs:    []string{"c1", "c2"},
		},
		{
			ID:          "c1",
			DocID:       docID,
			StartOffset: 0,
			EndOffset:   10,
			Level:       1,
			ParentID:    "p1",
		},
		{
			ID:          "c2",
			DocID:       docID,
			StartOffset: 10,
			EndOffset:   20,
			Level:       1,
			ParentID:    "p1",
		},
	}
	docStore.AddNodes(nodes)

	// Verify pointer slicing without storing text in node
	p1Text, ok := docStore.GetText("p1")
	if !ok || p1Text != "0123456789ABCDEFGHIJ" {
		t.Errorf("expected %q, got %q", "0123456789ABCDEFGHIJ", p1Text)
	}

	c1Text, ok := docStore.GetText("c1")
	if !ok || c1Text != "0123456789" {
		t.Errorf("expected %q, got %q", "0123456789", c1Text)
	}

	c2Text, ok := docStore.GetText("c2")
	if !ok || c2Text != "ABCDEFGHIJ" {
		t.Errorf("expected %q, got %q", "ABCDEFGHIJ", c2Text)
	}
}

func TestDocStore_SaveAndLoad(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "docstore-test-*")
	if err != nil {
		t.Fatalf("temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	saveDir := filepath.Join(tmpDir, "merging_store")

	if vectorstore.DocStoreExists(saveDir) {
		t.Errorf("expected DocStoreExists to be false before save")
	}

	ds1 := vectorstore.NewInMemoryDocStore()
	ds1.AddDocument("doc1", "The quick brown fox jumps over the lazy dog.")
	ds1.AddNodes([]vectorstore.HierarchyNode{
		{
			ID:          "node-1",
			DocID:       "doc1",
			StartOffset: 4,
			EndOffset:   19, // "quick brown fox"
			Level:       1,
		},
	})

	if err := ds1.Save(saveDir); err != nil {
		t.Fatalf("save: %v", err)
	}

	if !vectorstore.DocStoreExists(saveDir) {
		t.Errorf("expected DocStoreExists to be true after save")
	}

	ds2, err := vectorstore.LoadDocStoreFromDisk(saveDir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	txt, ok := ds2.GetText("node-1")
	if !ok || txt != "quick brown fox" {
		t.Errorf("expected %q, got %q", "quick brown fox", txt)
	}
}
