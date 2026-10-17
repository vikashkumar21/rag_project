// Package eval implements the RAG Triad evaluation framework.
//
// The RAG Triad (from TruLens) measures three dimensions of RAG quality:
//
//  1. Context Relevance  – Is the retrieved context relevant to the query?
//  2. Groundedness       – Is every claim in the answer supported by the context?
//  3. Answer Relevance   – Does the answer actually address the question?
//
// Python equivalent:
//
//	from trulens_eval import Tru, TruLlama, Feedback, OpenAI
//	from trulens_eval.feedback import Groundedness
//
// We implement the same three metrics using LLM-as-judge (the same technique
// TruLens uses internally), asking an LLM to score each dimension 0.0–1.0.
package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/vikash/rag_project/internal/llm"
)

// Record captures one evaluated (question, context, answer) triple.
// All three RAG Triad scores are in [0.0, 1.0]; higher is better.
type Record struct {
	Timestamp        time.Time `json:"timestamp"`
	AppID            string    `json:"app_id"`
	Question         string    `json:"question"`
	Answer           string    `json:"answer"`
	ContextRelevance float64   `json:"context_relevance"`
	Groundedness     float64   `json:"groundedness"`
	AnswerRelevance  float64   `json:"answer_relevance"`
}

// Evaluator runs RAG Triad scoring using an LLM as judge.
//
// Python equivalent:
//
//	tru_recorder = get_prebuilt_trulens_recorder(query_engine, app_id="...")
//	with tru_recorder as recording:
//	    response = query_engine.query(question)
type Evaluator struct {
	judge llm.LLM
	appID string
	log   []Record
}

// NewEvaluator returns an Evaluator that uses judge as its scoring LLM.
// Use the same LLM you use for generation, or a stronger model for stricter evaluation.
func NewEvaluator(judge llm.LLM, appID string) *Evaluator {
	return &Evaluator{judge: judge, appID: appID}
}

// Evaluate scores a (question, retrieved context, answer) triple using the
// RAG Triad and appends the result to the evaluator's internal log.
func (e *Evaluator) Evaluate(ctx context.Context, question, retrievedContext, answer string) (Record, error) {
	cr, err := e.scoreContextRelevance(ctx, question, retrievedContext)
	if err != nil {
		return Record{}, err
	}
	gr, err := e.scoreGroundedness(ctx, retrievedContext, answer)
	if err != nil {
		return Record{}, err
	}
	ar, err := e.scoreAnswerRelevance(ctx, question, answer)
	if err != nil {
		return Record{}, err
	}

	rec := Record{
		Timestamp:        time.Now().UTC(),
		AppID:            e.appID,
		Question:         question,
		Answer:           answer,
		ContextRelevance: cr,
		Groundedness:     gr,
		AnswerRelevance:  ar,
	}
	e.log = append(e.log, rec)
	return rec, nil
}

// Records returns a copy of all logged evaluations.
func (e *Evaluator) Records() []Record {
	out := make([]Record, len(e.log))
	copy(out, e.log)
	return out
}

// PrintLeaderboard prints a side-by-side comparison table of all evaluated pipelines.
//
// Python equivalent:
//
//	tru.get_leaderboard(app_ids=[])
func PrintLeaderboard(evaluators ...*Evaluator) {
	fmt.Printf("\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║ %-32s │ %-10s │ %-11s │ %-12s │ %-11s ║\n", "Pipeline / App ID", "Questions", "Context Rel", "Groundedness", "Answer Rel")
	fmt.Printf("╠──────────────────────────────────┼────────────┼─────────────┼──────────────┼─────────────╣\n")
	for _, e := range evaluators {
		if len(e.log) == 0 {
			fmt.Printf("║ %-32s │ %-10d │ %-11s │ %-12s │ %-11s ║\n", e.appID, 0, "N/A", "N/A", "N/A")
			continue
		}
		var cr, gr, ar float64
		for _, r := range e.log {
			cr += r.ContextRelevance
			gr += r.Groundedness
			ar += r.AnswerRelevance
		}
		n := float64(len(e.log))
		fmt.Printf("║ %-32s │ %-10d │ %-11.3f │ %-12.3f │ %-11.3f ║\n",
			e.appID, len(e.log), cr/n, gr/n, ar/n)
	}
	fmt.Printf("╚═══════════════════════════════════════════════════════════════════════════════════════════╝\n")
}

// SaveJSON writes all evaluation records to a JSON file.
//
// Python equivalent:
//
//	records, feedback = tru.get_records_and_feedback(app_ids=[])
func (e *Evaluator) SaveJSON(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("eval: create %q: %w", path, err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(e.log)
}

// ── Private scoring helpers ───────────────────────────────────────────────────

// scoreContextRelevance asks the judge: is the context relevant to the question?
func (e *Evaluator) scoreContextRelevance(ctx context.Context, question, context_ string) (float64, error) {
	prompt := fmt.Sprintf(`You are an objective evaluator for RAG systems.

Rate how relevant the retrieved context is to answering the question.

Question: %s

Retrieved Context:
%s

Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Reply with ONLY a decimal number like 0.7`, question, context_)

	return e.askScore(ctx, prompt)
}

// scoreGroundedness asks the judge: is the answer supported by the context?
func (e *Evaluator) scoreGroundedness(ctx context.Context, context_, answer string) (float64, error) {
	prompt := fmt.Sprintf(`You are an objective evaluator for RAG systems.

Rate how well the answer is grounded in (supported by) the provided context.
An answer is grounded if every factual claim it makes can be verified in the context.

Retrieved Context:
%s

Answer:
%s

Score from 0.0 (answer contains unsupported claims) to 1.0 (every claim is in the context).
Reply with ONLY a decimal number like 0.8`, context_, answer)

	return e.askScore(ctx, prompt)
}

// scoreAnswerRelevance asks the judge: does the answer address the question?
func (e *Evaluator) scoreAnswerRelevance(ctx context.Context, question, answer string) (float64, error) {
	prompt := fmt.Sprintf(`You are an objective evaluator for RAG systems.

Rate how well the answer addresses the user's question.

Question: %s

Answer:
%s

Score from 0.0 (answer is off-topic) to 1.0 (answer directly and fully addresses the question).
Reply with ONLY a decimal number like 0.9`, question, answer)

	return e.askScore(ctx, prompt)
}

// askScore sends a scoring prompt to the judge LLM and parses the float response.
func (e *Evaluator) askScore(ctx context.Context, prompt string) (float64, error) {
	resp, err := e.judge.Complete(ctx, []llm.Message{
		{
			Role:    llm.RoleSystem,
			Content: "You are an objective evaluator. Reply with ONLY a single decimal number between 0.0 and 1.0.",
		},
		{Role: llm.RoleUser, Content: prompt},
	})
	if err != nil {
		return 0, fmt.Errorf("eval: judge: %w", err)
	}

	var score float64
	_, parseErr := fmt.Sscanf(strings.TrimSpace(resp), "%f", &score)
	if parseErr != nil {
		return 0, fmt.Errorf("eval: parse score from %q: %w", resp, parseErr)
	}
	// Clamp to [0, 1].
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	return score, nil
}
