package loader_test

import (
	"testing"

	"github.com/vikash/rag_project/internal/loader"
)

func TestChunk(t *testing.T) {
	doc := loader.Document{
		ID:   "doc-1",
		Text: "The quick brown fox jumps over the lazy dog. Artificial Intelligence is transforming every industry.",
		Metadata: map[string]string{
			"source": "test.txt",
		},
	}

	opts := loader.ChunkOptions{
		ChunkSize:    30,
		ChunkOverlap: 10,
	}

	chunks := loader.Chunk(doc, opts)
	if len(chunks) == 0 {
		t.Fatalf("expected chunks, got 0")
	}

	for i, c := range chunks {
		if c.ID == "" {
			t.Errorf("chunk %d has empty ID", i)
		}
		if c.Metadata["source"] != "test.txt" {
			t.Errorf("chunk %d metadata lost parent source", i)
		}
		if len([]rune(c.Text)) > opts.ChunkSize {
			t.Errorf("chunk %d text length %d exceeds max %d", i, len([]rune(c.Text)), opts.ChunkSize)
		}
	}
}

func TestSentenceWindows(t *testing.T) {
	doc := loader.Document{
		ID:   "doc-sentences",
		Text: "First sentence here. Second sentence follows. Third sentence appears. Fourth sentence is here. Fifth sentence ends.",
	}

	opts := loader.SentenceWindowOptions{WindowSize: 1}
	nodes := loader.BuildSentenceWindows(doc, opts)

	if len(nodes) < 4 {
		t.Fatalf("expected at least 4 sentences, got %d", len(nodes))
	}

	// For node 1 ("Second sentence follows."), its window should contain First, Second, and Third
	n1 := nodes[1]
	if n1.Text != "Second sentence follows." {
		t.Errorf("unexpected sentence text: %q", n1.Text)
	}
	if n1.Metadata["window"] == "" {
		t.Errorf("expected window metadata, got empty string")
	}
}

func TestHierarchicalChunking(t *testing.T) {
	text := "Building an AI project requires careful scoping. Start with a simple baseline. Evaluate with rigorous metrics. Iterate on retrieval."
	doc := loader.Document{
		ID:   "doc-hier",
		Text: text,
	}

	allNodes, leafNodes := loader.BuildHierarchy(doc, loader.HierarchicalOptions{
		ChunkSizes: []int{80, 40, 20},
	})

	if len(allNodes) == 0 {
		t.Fatalf("expected nodes in hierarchy, got 0")
	}
	if len(leafNodes) == 0 {
		t.Fatalf("expected leaf nodes, got 0")
	}
	if len(leafNodes) >= len(allNodes) {
		t.Errorf("expected leaf nodes count (%d) to be strictly less than all nodes count (%d)",
			len(leafNodes), len(allNodes))
	}
}
