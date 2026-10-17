package loader

import (
	"fmt"
	"strings"

	"github.com/vikash/rag_project/internal/vectorstore"
)

// HierarchicalOptions configures the recursive chunking sizes.
type HierarchicalOptions struct {
	// ChunkSizes in descending order (e.g., [2048, 512, 128])
	ChunkSizes []int
}

// DefaultHierarchicalOptions matches LlamaIndex defaults used in the course.
func DefaultHierarchicalOptions() HierarchicalOptions {
	return HierarchicalOptions{
		ChunkSizes: []int{2048, 512, 128},
	}
}

// BuildHierarchy splits a Document into a multi-level hierarchy of nodes:
// Top level (parents) -> Middle level -> Leaf level (smallest chunks).
//
// Returns:
//   - allNodes: every node in the hierarchy (to be stored in the DocStore)
//   - leafNodes: only the deepest nodes (to be indexed in the VectorStore)
func BuildHierarchy(doc Document, opts HierarchicalOptions) (allNodes []vectorstore.HierarchyNode, leafNodes []vectorstore.HierarchyNode) {
	sizes := opts.ChunkSizes
	if len(sizes) == 0 {
		sizes = []int{2048, 512, 128}
	}

	// We recursively build level by level.
	// Initial root chunk: the whole document or level 0 chunks.
	level0Chunks := splitFixed(doc.Text, sizes[0], 0)
	var currentLevelNodes []vectorstore.HierarchyNode

	for i, text := range level0Chunks {
		id := fmt.Sprintf("%s#l0-%d", doc.ID, i)
		currentLevelNodes = append(currentLevelNodes, vectorstore.HierarchyNode{
			ID:       id,
			Text:     text,
			ParentID: "",
			Level:    0,
			Metadata: copyMeta(doc.Metadata),
		})
	}

	// For each subsequent level:
	for lvl := 1; lvl < len(sizes); lvl++ {
		targetSize := sizes[lvl]
		var nextLevelNodes []vectorstore.HierarchyNode

		for pIdx := range currentLevelNodes {
			parent := &currentLevelNodes[pIdx]
			childrenText := splitFixed(parent.Text, targetSize, 0)

			for cIdx, cText := range childrenText {
				childID := fmt.Sprintf("%s-c%d", parent.ID, cIdx)
				parent.ChildIDs = append(parent.ChildIDs, childID)

				childNode := vectorstore.HierarchyNode{
					ID:       childID,
					Text:     cText,
					ParentID: parent.ID,
					Level:    lvl,
					Metadata: copyMeta(parent.Metadata),
				}
				nextLevelNodes = append(nextLevelNodes, childNode)
			}
		}

		allNodes = append(allNodes, currentLevelNodes...)
		currentLevelNodes = nextLevelNodes
	}

	// The remaining currentLevelNodes are the leaf nodes
	allNodes = append(allNodes, currentLevelNodes...)
	leafNodes = currentLevelNodes
	return allNodes, leafNodes
}

// splitFixed splits text into chunks of targetSize runes.
func splitFixed(text string, targetSize, overlap int) []string {
	runes := []rune(text)
	total := len(runes)
	if total == 0 {
		return nil
	}
	if targetSize <= 0 {
		targetSize = 512
	}

	step := targetSize - overlap
	if step <= 0 {
		step = targetSize
	}

	var chunks []string
	for start := 0; start < total; start += step {
		end := start + targetSize
		if end > total {
			end = total
		}
		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
	}
	return chunks
}
