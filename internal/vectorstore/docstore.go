package vectorstore

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// HierarchyNode represents a node in a hierarchical document tree using
// a zero-copy pointer/span approach.
//
// Rather than duplicating text strings at every tier of the tree (e.g. duplicating
// the text in a 128-char leaf, a 512-char parent, and a 2048-char grandparent),
// HierarchyNode stores only the interval coordinates [StartOffset, EndOffset]
// into the single master document stored in the DocStore.
type HierarchyNode struct {
	ID          string            `json:"id"`
	DocID       string            `json:"doc_id"`
	StartOffset int               `json:"start_offset"` // Rune offset in source doc
	EndOffset   int               `json:"end_offset"`   // Rune offset in source doc
	ParentID    string            `json:"parent_id,omitempty"`
	ChildIDs    []string          `json:"child_ids,omitempty"`
	Level       int               `json:"level"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// DocStore provides storage for master document texts and hierarchical interval nodes.
//
// Python equivalent:
//
//	storage_context.docstore (LlamaIndex)
type DocStore interface {
	AddDocument(docID, text string)
	AddNodes(nodes []HierarchyNode)
	GetNode(id string) (HierarchyNode, bool)
	Get(id string) (HierarchyNode, bool) // Alias for GetNode
	GetText(id string) (string, bool)
	GetAllNodes() map[string]HierarchyNode
	Save(dirPath string) error
}

// docStorePayload is the serialized representation stored on disk.
type docStorePayload struct {
	Docs  map[string]string        `json:"docs"`  // Master document texts stored once
	Nodes map[string]HierarchyNode `json:"nodes"` // Interval pointers
}

// InMemoryDocStore implements DocStore with thread-safe in-memory maps.
type InMemoryDocStore struct {
	mu    sync.RWMutex
	docs  map[string]string        // docID -> raw text string
	runes map[string][]rune        // docID -> parsed runes for fast sub-rune slicing
	nodes map[string]HierarchyNode // nodeID -> interval node
}

// NewInMemoryDocStore returns a new empty InMemoryDocStore.
func NewInMemoryDocStore() *InMemoryDocStore {
	return &InMemoryDocStore{
		docs:  make(map[string]string),
		runes: make(map[string][]rune),
		nodes: make(map[string]HierarchyNode),
	}
}

// AddDocument registers a master document text once.
func (s *InMemoryDocStore) AddDocument(docID, text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docs[docID] = text
	s.runes[docID] = []rune(text)
}

// AddNodes appends hierarchical interval nodes to the store.
func (s *InMemoryDocStore) AddNodes(nodes []HierarchyNode) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, n := range nodes {
		s.nodes[n.ID] = n
	}
}

// Add is kept for backward compatibility; adds nodes to the store.
func (s *InMemoryDocStore) Add(nodes []HierarchyNode) {
	s.AddNodes(nodes)
}

// GetNode retrieves a node's metadata and pointers by ID.
func (s *InMemoryDocStore) GetNode(id string) (HierarchyNode, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	n, ok := s.nodes[id]
	return n, ok
}

// Get is kept for backward compatibility; alias for GetNode.
func (s *InMemoryDocStore) Get(id string) (HierarchyNode, bool) {
	return s.GetNode(id)
}

// GetText dynamically resolves and slices the text for a node from the master document
// without requiring text duplication in the node struct.
func (s *InMemoryDocStore) GetText(id string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	n, ok := s.nodes[id]
	if !ok {
		return "", false
	}

	r, hasDoc := s.runes[n.DocID]
	if !hasDoc {
		// Fallback to docs map if runes slice wasn't initialized
		rawText, exists := s.docs[n.DocID]
		if !exists {
			return "", false
		}
		r = []rune(rawText)
		s.runes[n.DocID] = r
	}

	start := n.StartOffset
	end := n.EndOffset
	if start < 0 {
		start = 0
	}
	if end > len(r) {
		end = len(r)
	}
	if start >= end {
		return "", true
	}

	return string(r[start:end]), true
}

// GetAllNodes returns a shallow copy of all nodes in the store.
func (s *InMemoryDocStore) GetAllNodes() map[string]HierarchyNode {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]HierarchyNode, len(s.nodes))
	for k, v := range s.nodes {
		out[k] = v
	}
	return out
}

// GetAll is an alias for GetAllNodes for compatibility.
func (s *InMemoryDocStore) GetAll() map[string]HierarchyNode {
	return s.GetAllNodes()
}

// Save persists both the master documents and the interval nodes to docstore.json.
func (s *InMemoryDocStore) Save(dirPath string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("docstore: mkdir %q: %w", dirPath, err)
	}

	filePath := filepath.Join(dirPath, "docstore.json")
	f, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("docstore: create %q: %w", filePath, err)
	}
	defer f.Close()

	payload := docStorePayload{
		Docs:  s.docs,
		Nodes: s.nodes,
	}

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(payload); err != nil {
		return fmt.Errorf("docstore: encode %q: %w", filePath, err)
	}

	return nil
}

// LoadDocStoreFromDisk loads a docstore.json file from dirPath.
func LoadDocStoreFromDisk(dirPath string) (*InMemoryDocStore, error) {
	filePath := filepath.Join(dirPath, "docstore.json")
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("docstore: open %q: %w", filePath, err)
	}
	defer f.Close()

	var payload docStorePayload
	if err := json.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("docstore: decode %q: %w", filePath, err)
	}

	store := NewInMemoryDocStore()
	store.docs = payload.Docs
	store.nodes = payload.Nodes

	// Pre-index runes for high performance slicing
	for docID, text := range store.docs {
		store.runes[docID] = []rune(text)
	}

	return store, nil
}

// DocStoreExists checks if docstore.json exists in dirPath.
func DocStoreExists(dirPath string) bool {
	filePath := filepath.Join(dirPath, "docstore.json")
	info, err := os.Stat(filePath)
	if err != nil {
		return false
	}
	return !info.IsDir() && info.Size() > 0
}
