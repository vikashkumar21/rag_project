package llm

import (
	"context"
	"fmt"

	oai "github.com/sashabaranov/go-openai"
)

// OpenAILLM wraps the OpenAI chat completions API.
//
// Python equivalent:
//
//	from llama_index.llms import OpenAI
//	llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
type OpenAILLM struct {
	client      *oai.Client
	model       string
	temperature float32
}

// NewOpenAI returns an OpenAILLM.
//
// Good model choices:
//   - "gpt-3.5-turbo"   – fast, cheap; matches the course default
//   - "gpt-4o-mini"     – smarter than gpt-3.5 at similar cost
//   - "gpt-4o"          – highest quality
func NewOpenAI(apiKey, model string, temperature float32) *OpenAILLM {
	return &OpenAILLM{
		client:      oai.NewClient(apiKey),
		model:       model,
		temperature: temperature,
	}
}

// Complete sends messages to OpenAI and returns the assistant's reply text.
func (l *OpenAILLM) Complete(ctx context.Context, messages []Message) (string, error) {
	apiMsgs := make([]oai.ChatCompletionMessage, len(messages))
	for i, m := range messages {
		apiMsgs[i] = oai.ChatCompletionMessage{
			Role:    m.Role,
			Content: m.Content,
		}
	}

	resp, err := l.client.CreateChatCompletion(ctx, oai.ChatCompletionRequest{
		Model:       l.model,
		Messages:    apiMsgs,
		Temperature: l.temperature,
	})
	if err != nil {
		return "", fmt.Errorf("llm/openai: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("llm/openai: no choices returned")
	}
	return resp.Choices[0].Message.Content, nil
}
