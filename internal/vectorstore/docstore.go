package vectorstore

import (
	"sync"
)

// HierarchyNode represents a node in a hierarchical document tree.
type HierarchyNode struct {
	ID       string
	Text     string
	ParentID string
	ChildIDs []string
	Level    int
	Metadata map[string]string
}

// DocStore is a key-value store for retrieving nodes by their unique ID.
//
// Python equivalent:
//
//	storage_context.docstore (LlamaIndex)
type DocStore interface {
	Add(nodes []HierarchyNode)
	Get(id string) (HierarchyNode, bool)
	GetAll() map[string]HierarchyNode
}

// InMemoryDocStore implements DocStore in RAM with a mutex.
type InMemoryDocStore struct {
	mu    sync.RWMutex
	nodes map[string]HierarchyNode
}

// NewInMemoryDocStore returns a new InMemoryDocStore.
func NewInMemoryDocStore() *InMemoryDocStore {
	return &InMemoryDocStore{
		nodes: make(map[string]HierarchyNode),
	}
}

// Add saves nodes into the store.
func (s *InMemoryDocStore) Add(nodes []HierarchyNode) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, n := range nodes {
		s.nodes[n.ID] = n
	}
}

// Get retrieves a node by its ID.
func (s *InMemoryDocStore) Get(id string) (HierarchyNode, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	n, ok := s.nodes[id]
	return n, ok
}

// GetAll returns a copy of all nodes.
func (s *InMemoryDocStore) GetAll() map[string]HierarchyNode {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]HierarchyNode, len(s.nodes))
	for k, v := range s.nodes {
		out[k] = v
	}
	return out
}
