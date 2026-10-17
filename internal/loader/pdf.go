package loader

import (
	"fmt"
	"strings"

	"github.com/ledongthuc/pdf"
)

// LoadPDF reads a PDF file and returns one Document per page.
//
// Python equivalent:
//
//	SimpleDirectoryReader(input_files=["file.pdf"]).load_data()
//
// Each Document's ID encodes the file path and page number so it
// remains traceable after chunking.
func LoadPDF(path string) ([]Document, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("loader: open %q: %w", path, err)
	}
	defer f.Close()

	docs := make([]Document, 0, r.NumPage())
	for i := 1; i <= r.NumPage(); i++ {
		page := r.Page(i)
		if page.V.IsNull() {
			continue
		}
		text, err := page.GetPlainText(nil)
		if err != nil {
			// Non-fatal: log and skip unreadable pages.
			fmt.Printf("[loader] warning: page %d of %q: %v\n", i, path, err)
			continue
		}
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		docs = append(docs, Document{
			ID:   fmt.Sprintf("%s#page-%d", path, i),
			Text: text,
			Metadata: map[string]string{
				"source": path,
				"page":   fmt.Sprintf("%d", i),
			},
		})
	}
	return docs, nil
}
