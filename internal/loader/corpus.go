package loader

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// SentenceCorpus stores sequential sentences per document.
//
// In contrast to static pre-computed sentence windows (where window text is duplicated
// up to 7 times across overlapping metadata fields), SentenceCorpus stores each
// sentence exactly once. Surrounding context windows are sliced dynamically at query time.
type SentenceCorpus struct {
	mu   sync.RWMutex
	docs map[string][]string // docID -> slice of sentences
}

// NewSentenceCorpus creates an empty sentence corpus.
func NewSentenceCorpus() *SentenceCorpus {
	return &SentenceCorpus{
		docs: make(map[string][]string),
	}
}

// AddDocument splits doc.Text into sentences and stores them under doc.ID.
func (c *SentenceCorpus) AddDocument(doc Document) []string {
	sentences := SplitSentences(doc.Text)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.docs[doc.ID] = sentences
	return sentences
}

// SetSentences stores an explicit list of sentences under docID.
func (c *SentenceCorpus) SetSentences(docID string, sentences []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.docs[docID] = sentences
}

// GetSentences returns all sentences for a given document.
func (c *SentenceCorpus) GetSentences(docID string) ([]string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	sents, ok := c.docs[docID]
	return sents, ok
}

// GetWindow extracts a single window of +/- windowSize sentences around centerIdx.
func (c *SentenceCorpus) GetWindow(docID string, centerIdx, windowSize int) (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	sents, ok := c.docs[docID]
	if !ok {
		return "", fmt.Errorf("corpus: document %q not found", docID)
	}
	if len(sents) == 0 {
		return "", nil
	}
	if centerIdx < 0 || centerIdx >= len(sents) {
		return "", fmt.Errorf("corpus: sentence index %d out of range [0, %d)", centerIdx, len(sents))
	}
	if windowSize < 0 {
		windowSize = 0
	}

	start := centerIdx - windowSize
	if start < 0 {
		start = 0
	}
	end := centerIdx + windowSize + 1
	if end > len(sents) {
		end = len(sents)
	}

	return strings.Join(sents[start:end], " "), nil
}

// Interval represents a range of sentence indices [Start, End] inclusive.
type Interval struct {
	Start int
	End   int
}

// GetMergedWindows takes multiple hit sentence indices for a document, expands each
// by +/- windowSize, merges any overlapping or adjacent intervals, and returns clean,
// non-duplicate window passages.
//
// For example: If Sentence 3 and Sentence 4 both match with windowSize=2:
// - Sentence 3 window: [1, 5]
// - Sentence 4 window: [2, 6]
// - Merged interval:   [1, 6]  (Single continuous block, 0 duplicate text!)
func (c *SentenceCorpus) GetMergedWindows(docID string, hitIndices []int, windowSize int) ([]string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	sents, ok := c.docs[docID]
	if !ok {
		return nil, fmt.Errorf("corpus: document %q not found", docID)
	}
	if len(sents) == 0 || len(hitIndices) == 0 {
		return nil, nil
	}
	if windowSize < 0 {
		windowSize = 0
	}

	// 1. Build raw intervals for each hit index
	intervals := make([]Interval, 0, len(hitIndices))
	for _, idx := range hitIndices {
		if idx < 0 || idx >= len(sents) {
			continue
		}
		start := idx - windowSize
		if start < 0 {
			start = 0
		}
		end := idx + windowSize
		if end >= len(sents) {
			end = len(sents) - 1
		}
		intervals = append(intervals, Interval{Start: start, End: end})
	}

	if len(intervals) == 0 {
		return nil, nil
	}

	// 2. Sort intervals by start index
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i].Start == intervals[j].Start {
			return intervals[i].End < intervals[j].End
		}
		return intervals[i].Start < intervals[j].Start
	})

	// 3. Merge overlapping or adjacent intervals
	var merged []Interval
	curr := intervals[0]
	for i := 1; i < len(intervals); i++ {
		next := intervals[i]
		if next.Start <= curr.End+1 {
			// Overlapping or contiguous: extend current interval
			if next.End > curr.End {
				curr.End = next.End
			}
		} else {
			merged = append(merged, curr)
			curr = next
		}
	}
	merged = append(merged, curr)

	// 4. Materialize passage text for each merged interval
	passages := make([]string, len(merged))
	for i, iv := range merged {
		passages[i] = strings.Join(sents[iv.Start:iv.End+1], " ")
	}

	return passages, nil
}

// Save writes the corpus to corpus.json inside dirPath.
func (c *SentenceCorpus) Save(dirPath string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("corpus: mkdir %q: %w", dirPath, err)
	}

	filePath := filepath.Join(dirPath, "corpus.json")
	f, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("corpus: create %q: %w", filePath, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(c.docs)
}

// LoadCorpusFromDisk loads corpus.json from dirPath.
func LoadCorpusFromDisk(dirPath string) (*SentenceCorpus, error) {
	filePath := filepath.Join(dirPath, "corpus.json")
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("corpus: open %q: %w", filePath, err)
	}
	defer f.Close()

	var docs map[string][]string
	if err := json.NewDecoder(f).Decode(&docs); err != nil {
		return nil, fmt.Errorf("corpus: decode %q: %w", filePath, err)
	}

	return &SentenceCorpus{docs: docs}, nil
}

// CorpusExists checks if a corpus.json file exists at dirPath.
func CorpusExists(dirPath string) bool {
	filePath := filepath.Join(dirPath, "corpus.json")
	info, err := os.Stat(filePath)
	if err != nil {
		return false
	}
	return !info.IsDir() && info.Size() > 0
}
