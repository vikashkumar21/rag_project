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

// BuildHierarchy splits a Document into a multi-level hierarchy of pointer-based nodes.
// Each node stores interval offsets [StartOffset, EndOffset] into the original text,
// ensuring zero text duplication across tree tiers.
//
// Returns:
//   - allNodes: every pointer node in the hierarchy (to be stored in the DocStore)
//   - leafNodes: only the deepest pointer nodes (whose resolved text is embedded in VectorStore)
func BuildHierarchy(doc Document, opts HierarchicalOptions) (allNodes []vectorstore.HierarchyNode, leafNodes []vectorstore.HierarchyNode) {
	sizes := opts.ChunkSizes
	if len(sizes) == 0 {
		sizes = []int{2048, 512, 128}
	}

	runes := []rune(doc.Text)
	total := len(runes)
	if total == 0 {
		return nil, nil
	}

	// 1. Level 0: Top-level chunks
	step0 := sizes[0]
	if step0 <= 0 {
		step0 = 2048
	}

	var currentLevelNodes []vectorstore.HierarchyNode
	idx0 := 0
	for start := 0; start < total; start += step0 {
		end := start + step0
		if end > total {
			end = total
		}
		id := fmt.Sprintf("%s#l0-%d", doc.ID, idx0)
		idx0++
		currentLevelNodes = append(currentLevelNodes, vectorstore.HierarchyNode{
			ID:          id,
			DocID:       doc.ID,
			StartOffset: start,
			EndOffset:   end,
			ParentID:    "",
			Level:       0,
			Metadata:    copyMeta(doc.Metadata),
		})
	}

	allNodes = append(allNodes, currentLevelNodes...)

	// 2. Build sub-levels recursively
	for lvl := 1; lvl < len(sizes); lvl++ {
		targetSize := sizes[lvl]
		if targetSize <= 0 {
			targetSize = 512
		}
		var nextLevelNodes []vectorstore.HierarchyNode

		for pIdx := range currentLevelNodes {
			parent := &currentLevelNodes[pIdx]
			pStart := parent.StartOffset
			pEnd := parent.EndOffset
			pLen := pEnd - pStart

			cIdx := 0
			for subStart := 0; subStart < pLen; subStart += targetSize {
				subEnd := subStart + targetSize
				if subEnd > pLen {
					subEnd = pLen
				}
				childID := fmt.Sprintf("%s-c%d", parent.ID, cIdx)
				cIdx++
				parent.ChildIDs = append(parent.ChildIDs, childID)

				childNode := vectorstore.HierarchyNode{
					ID:          childID,
					DocID:       doc.ID,
					StartOffset: pStart + subStart,
					EndOffset:   pStart + subEnd,
					ParentID:    parent.ID,
					Level:       lvl,
					Metadata:    copyMeta(parent.Metadata),
				}
				nextLevelNodes = append(nextLevelNodes, childNode)
			}
		}

		allNodes = append(allNodes, nextLevelNodes...)
		currentLevelNodes = nextLevelNodes
	}

	leafNodes = currentLevelNodes
	return allNodes, leafNodes
}

// ResolveNodeText extracts the exact substring represented by a HierarchyNode
// from the document text using rune offsets.
func ResolveNodeText(docText string, n vectorstore.HierarchyNode) string {
	runes := []rune(docText)
	start := n.StartOffset
	end := n.EndOffset
	if start < 0 {
		start = 0
	}
	if end > len(runes) {
		end = len(runes)
	}
	if start >= end {
		return ""
	}
	return strings.TrimSpace(string(runes[start:end]))
}
