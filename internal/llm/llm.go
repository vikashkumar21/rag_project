// Package llm defines the interface for large language model clients.
package llm

import "context"

// Standard OpenAI-compatible chat roles.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Message is a single turn in a multi-turn chat conversation.
type Message struct {
	Role    string
	Content string
}

// LLM generates a text completion from a sequence of chat messages.
//
// Python equivalent:
//
//	llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
//
// Implementations must be safe for concurrent use.
type LLM interface {
	Complete(ctx context.Context, messages []Message) (string, error)
}
