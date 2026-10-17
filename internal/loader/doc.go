// Package loader provides types and functions for reading source documents
// and splitting them into smaller chunks suitable for embedding.
package loader

// Document represents a text document loaded from a source file.
// Every Document has a unique ID, the raw extracted text, and optional
// key-value metadata (e.g. "source", "page").
//
// This is the Go equivalent of LlamaIndex's Document class.
type Document struct {
	ID       string
	Text     string
	Metadata map[string]string
}
