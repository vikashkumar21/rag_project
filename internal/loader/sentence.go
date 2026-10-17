package loader

import (
	"fmt"
	"regexp"
	"strings"
)

// SentenceNode represents an individual sentence along with its expanded surrounding window.
//
// Python equivalent:
//
//	SentenceWindowNodeParser (LlamaIndex)
//
// In sentence-window retrieval:
// - Text is embedded (high precision retrieval of specific facts)
// - Window is passed to the LLM (broad context for coherent generation)
type SentenceNode struct {
	ID           string
	Text         string // The focus sentence
	Window       string // Surrounding sentences [i-windowSize ... i ... i+windowSize]
	Metadata     map[string]string
	OriginalText string
}

// SentenceWindowOptions defines how sentences and windows are extracted.
type SentenceWindowOptions struct {
	WindowSize int // Number of sentences before and after the focus sentence (default 3)
}

// DefaultSentenceWindowOptions returns default configuration matching LlamaIndex.
func DefaultSentenceWindowOptions() SentenceWindowOptions {
	return SentenceWindowOptions{
		WindowSize: 3,
	}
}

// sentenceSplitterRegexp splits text on sentence boundary punctuation followed by space or newline.
var sentenceSplitterRegexp = regexp.MustCompile(`(?:\. |\? |\! |\n\n+)`)

// SplitSentences breaks a text into individual clean sentence strings.
func SplitSentences(text string) []string {
	// Normalize line breaks
	text = strings.ReplaceAll(text, "\r\n", "\n")

	indices := sentenceSplitterRegexp.FindAllStringIndex(text, -1)
	if len(indices) == 0 {
		trimmed := strings.TrimSpace(text)
		if trimmed == "" {
			return nil
		}
		return []string{trimmed}
	}

	var sentences []string
	start := 0
	for _, idx := range indices {
		end := idx[1]
		segment := strings.TrimSpace(text[start:end])
		if segment != "" {
			sentences = append(sentences, segment)
		}
		start = end
	}
	if start < len(text) {
		remainder := strings.TrimSpace(text[start:])
		if remainder != "" {
			sentences = append(sentences, remainder)
		}
	}
	return sentences
}

// BuildSentenceWindows parses a Document into a slice of SentenceNodes.
// Each node has its single sentence as Text, and windowSize sentences before and after
// joined together as Window.
func BuildSentenceWindows(doc Document, opts SentenceWindowOptions) []SentenceNode {
	if opts.WindowSize <= 0 {
		opts.WindowSize = 3
	}

	rawSentences := SplitSentences(doc.Text)
	if len(rawSentences) == 0 {
		return nil
	}

	nodes := make([]SentenceNode, len(rawSentences))
	for i, s := range rawSentences {
		// Calculate window slice bounds
		start := i - opts.WindowSize
		if start < 0 {
			start = 0
		}
		end := i + opts.WindowSize + 1
		if end > len(rawSentences) {
			end = len(rawSentences)
		}

		windowText := strings.Join(rawSentences[start:end], " ")

		meta := copyMeta(doc.Metadata)
		meta["window"] = windowText
		meta["original_text"] = s

		nodes[i] = SentenceNode{
			ID:           fmt.Sprintf("%s#sent-%d", doc.ID, i),
			Text:         s,
			Window:       windowText,
			Metadata:     meta,
			OriginalText: s,
		}
	}

	return nodes
}
