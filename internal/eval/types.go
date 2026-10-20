package eval

import (
	"time"
)

// FeedbackResult pairs an evaluation score (0.0 to 1.0) with its
// Chain-of-Thought (CoT) reasoning explanation.
//
// Python equivalent:
//
//	Feedback(provider.relevance_with_cot_reasons, ...)
type FeedbackResult struct {
	Score  float64 `json:"score"`
	Reason string  `json:"reason"`
}

// Record captures one evaluated (question, context, answer) execution trace.
//
// Python equivalent:
//
//	records, feedback = tru.get_records_and_feedback(app_ids=[])
//	records[["input", "output"] + feedback]
type Record struct {
	ID               string         `json:"record_id"`
	Timestamp        time.Time      `json:"timestamp"`
	AppID            string         `json:"app_id"`
	Question         string         `json:"input"`
	Answer           string         `json:"output"`
	RetrievedChunks  []string       `json:"retrieved_chunks,omitempty"`
	ContextRelevance FeedbackResult `json:"context_relevance"`
	Groundedness     FeedbackResult `json:"groundedness"`
	AnswerRelevance  FeedbackResult `json:"answer_relevance"`
}

// LeaderboardRow aggregates RAG Triad scores across multiple questions for an App.
type LeaderboardRow struct {
	AppID            string  `json:"app_id"`
	NumRecords       int     `json:"num_records"`
	ContextRelevance float64 `json:"context_relevance"`
	Groundedness     float64 `json:"groundedness"`
	AnswerRelevance  float64 `json:"answer_relevance"`
}
