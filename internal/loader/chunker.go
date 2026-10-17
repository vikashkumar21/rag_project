package loader

import (
	"fmt"
	"strings"
)

// ChunkOptions controls how a Document is split into overlapping sub-documents.
//
// ChunkSize    – maximum size of each chunk, measured in Unicode code points (runes).
//
//	Measuring in runes (not bytes) is safer for multilingual text.
//
// ChunkOverlap – number of runes shared between consecutive chunks.
//
//	Overlap lets the LLM see bridging context that would otherwise be
//	cut at a boundary — the same idea used by LlamaIndex's node parsers.
type ChunkOptions struct {
	ChunkSize    int
	ChunkOverlap int
}

// DefaultChunkOptions returns sensible defaults that match common RAG practice.
func DefaultChunkOptions() ChunkOptions {
	return ChunkOptions{ChunkSize: 1024, ChunkOverlap: 128}
}

// Chunk splits doc.Text into a slice of overlapping sub-Documents.
// Each sub-document inherits the parent's metadata and gets a unique ID.
//
// Python equivalent (LlamaIndex handles this internally when building
// VectorStoreIndex.from_documents):
//
//	SentenceSplitter(chunk_size=1024, chunk_overlap=128)
func Chunk(doc Document, opts ChunkOptions) []Document {
	runes := []rune(doc.Text)
	total := len(runes)
	if total == 0 {
		return nil
	}

	step := opts.ChunkSize - opts.ChunkOverlap
	if step <= 0 {
		step = opts.ChunkSize // safety: avoid infinite loop
	}

	var chunks []Document
	for start, idx := 0, 0; start < total; start, idx = start+step, idx+1 {
		end := start + opts.ChunkSize
		if end > total {
			end = total
		}
		chunks = append(chunks, Document{
			ID:       fmt.Sprintf("%s#chunk-%d", doc.ID, idx),
			Text:     strings.TrimSpace(string(runes[start:end])),
			Metadata: copyMeta(doc.Metadata),
		})
	}
	return chunks
}

// MergeDocuments concatenates multiple Documents into a single one,
// joining their texts with sep.
//
// Python equivalent:
//
//	Document(text="\n\n".join([doc.text for doc in documents]))
func MergeDocuments(docs []Document, sep string) Document {
	parts := make([]string, 0, len(docs))
	for _, d := range docs {
		if d.Text != "" {
			parts = append(parts, d.Text)
		}
	}
	return Document{
		ID:       "merged",
		Text:     strings.Join(parts, sep),
		Metadata: map[string]string{"source": "merged"},
	}
}

// copyMeta returns a shallow copy of a metadata map.
func copyMeta(m map[string]string) map[string]string {
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
