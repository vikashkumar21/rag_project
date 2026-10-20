package vectorstore

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Save writes all vectors and their metadata to a JSON file inside dirPath.
//
// Python equivalent:
//
//	sentence_index.storage_context.persist(persist_dir="./sentence_index")
func (s *InMemoryStore) Save(dirPath string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("vectorstore: mkdir %q: %w", dirPath, err)
	}

	filePath := filepath.Join(dirPath, "vectors.json")
	f, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("vectorstore: create %q: %w", filePath, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(s.vectors); err != nil {
		return fmt.Errorf("vectorstore: encode %q: %w", filePath, err)
	}

	return nil
}

// LoadFromDisk reads a previously persisted vector store from dirPath.
//
// Python equivalent:
//
//	load_index_from_storage(StorageContext.from_defaults(persist_dir=...))
func LoadFromDisk(dirPath string) (*InMemoryStore, error) {
	filePath := filepath.Join(dirPath, "vectors.json")
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("vectorstore: open %q: %w", filePath, err)
	}
	defer f.Close()

	var vecs []Vector
	if err := json.NewDecoder(f).Decode(&vecs); err != nil {
		return nil, fmt.Errorf("vectorstore: decode %q: %w", filePath, err)
	}

	store := NewInMemory()
	store.vectors = vecs
	return store, nil
}

// Exists checks if a persisted index exists at dirPath.
func Exists(dirPath string) bool {
	filePath := filepath.Join(dirPath, "vectors.json")
	info, err := os.Stat(filePath)
	if err != nil {
		return false
	}
	return !info.IsDir() && info.Size() > 0
}
