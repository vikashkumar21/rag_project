package loader_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vikash/rag_project/internal/loader"
)

func TestSentenceCorpus_GetWindow(t *testing.T) {
	corpus := loader.NewSentenceCorpus()
	sents := []string{
		"Sentence zero.",
		"Sentence one.",
		"Sentence two.",
		"Sentence three.",
		"Sentence four.",
	}
	corpus.SetSentences("doc1", sents)

	// Center at index 2, windowSize 1 -> should return sentences 1, 2, 3
	win, err := corpus.GetWindow("doc1", 2, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := "Sentence one. Sentence two. Sentence three."
	if win != expected {
		t.Errorf("expected %q, got %q", expected, win)
	}

	// Boundary: Center at index 0, windowSize 2 -> should return sentences 0, 1, 2
	win0, err := corpus.GetWindow("doc1", 0, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected0 := "Sentence zero. Sentence one. Sentence two."
	if win0 != expected0 {
		t.Errorf("expected %q, got %q", expected0, win0)
	}
}

func TestSentenceCorpus_GetMergedWindows(t *testing.T) {
	corpus := loader.NewSentenceCorpus()
	sents := []string{
		"S0.", "S1.", "S2.", "S3.", "S4.", "S5.", "S6.", "S7.", "S8.", "S9.",
	}
	corpus.SetSentences("doc1", sents)

	// Hits at Sentence 3 and Sentence 4 with windowSize 1:
	// Hit 3: [2, 4]
	// Hit 4: [3, 5]
	// Should merge into single passage: [2, 5] -> "S2. S3. S4. S5."
	passages, err := corpus.GetMergedWindows("doc1", []int{3, 4}, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(passages) != 1 {
		t.Fatalf("expected 1 merged passage, got %d", len(passages))
	}
	expected := "S2. S3. S4. S5."
	if passages[0] != expected {
		t.Errorf("expected %q, got %q", expected, passages[0])
	}

	// Hits at Sentence 1 and Sentence 8 with windowSize 1 (disjoint):
	// Hit 1: [0, 2]
	// Hit 8: [7, 9]
	// Should return 2 separate passages
	disjoint, err := corpus.GetMergedWindows("doc1", []int{1, 8}, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(disjoint) != 2 {
		t.Fatalf("expected 2 disjoint passages, got %d", len(disjoint))
	}
	if disjoint[0] != "S0. S1. S2." || disjoint[1] != "S7. S8. S9." {
		t.Errorf("unexpected disjoint passages: %+v", disjoint)
	}
}

func TestSentenceCorpus_SaveAndLoad(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "corpus-test-*")
	if err != nil {
		t.Fatalf("temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	corpusPath := filepath.Join(tmpDir, "corpus_store")

	if loader.CorpusExists(corpusPath) {
		t.Errorf("expected CorpusExists to be false before save")
	}

	c1 := loader.NewSentenceCorpus()
	c1.SetSentences("doc-abc", []string{"First sentence.", "Second sentence."})
	if err := c1.Save(corpusPath); err != nil {
		t.Fatalf("save: %v", err)
	}

	if !loader.CorpusExists(corpusPath) {
		t.Errorf("expected CorpusExists to be true after save")
	}

	c2, err := loader.LoadCorpusFromDisk(corpusPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	sents, ok := c2.GetSentences("doc-abc")
	if !ok || len(sents) != 2 || sents[0] != "First sentence." {
		t.Errorf("unexpected loaded sentences: %+v", sents)
	}
}
