// Package eval implements the RAG Triad evaluation framework from TruLens.
//
// The RAG Triad measures three critical dimensions of RAG quality:
//
//  1. Context Relevance  (Input Q  vs  Retrieved Context S)
//     - Is the retrieved context relevant to answering the question?
//  2. Groundedness       (Retrieved Context S  vs  Generated Answer A)
//     - Is every factual claim in the answer supported by the retrieved context?
//  3. Answer Relevance   (Input Q  vs  Generated Answer A)
//     - Does the answer directly and fully address the user's question?
//
// Python equivalent:
//
//	from trulens_eval import Tru, TruLlama, Feedback
//	from trulens_eval.feedback import Groundedness
package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/vikash/rag_project/internal/llm"
)

// Evaluator evaluates RAG executions and tracks evaluation records.
//
// Python equivalent:
//
//	tru = Tru()
//	tru_recorder = TruLlama(..., app_id=..., feedbacks=[...])
type Evaluator struct {
	mu    sync.RWMutex
	judge llm.LLM
	appID string
	log   []Record
}

// NewEvaluator creates a new Evaluator.
func NewEvaluator(judge llm.LLM, appID string) *Evaluator {
	return &Evaluator{
		judge: judge,
		appID: appID,
	}
}

// AppID returns the application identifier for this evaluator.
func (e *Evaluator) AppID() string {
	return e.appID
}

// Evaluate performs the complete RAG Triad evaluation on a single run:
// 1. Context Relevance (with CoT)
// 2. Groundedness (with CoT statement-by-statement check)
// 3. Answer Relevance (with CoT)
func (e *Evaluator) Evaluate(ctx context.Context, question, retrievedContext, answer string) (Record, error) {
	cr, err := e.ScoreContextRelevance(ctx, question, retrievedContext)
	if err != nil {
		return Record{}, fmt.Errorf("eval/context_relevance: %w", err)
	}

	gr, err := e.ScoreGroundedness(ctx, retrievedContext, answer)
	if err != nil {
		return Record{}, fmt.Errorf("eval/groundedness: %w", err)
	}

	ar, err := e.ScoreAnswerRelevance(ctx, question, answer)
	if err != nil {
		return Record{}, fmt.Errorf("eval/answer_relevance: %w", err)
	}

	rec := Record{
		ID:               fmt.Sprintf("rec-%d", time.Now().UnixNano()),
		Timestamp:        time.Now().UTC(),
		AppID:            e.appID,
		Question:         question,
		Answer:           answer,
		ContextRelevance: cr,
		Groundedness:     gr,
		AnswerRelevance:  ar,
	}

	e.mu.Lock()
	e.log = append(e.log, rec)
	e.mu.Unlock()

	return rec, nil
}

// ── 1. Answer Relevance Feedback ──────────────────────────────────────────────
//
// Python equivalent:
//
//	f_qa_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance").on_input_output()
func (e *Evaluator) ScoreAnswerRelevance(ctx context.Context, question, answer string) (FeedbackResult, error) {
	prompt := fmt.Sprintf(`You are an expert evaluator assessing Answer Relevance in a RAG system.

Evaluate whether the Answer directly and comprehensively addresses the Question.
Ignore whether the answer is factually true; focus only on whether it answers what was asked.

Question:
%s

Answer:
%s

Respond with a JSON object containing:
- "score": a number from 0.0 (completely irrelevant or evasive) to 1.0 (perfectly relevant and directly answers the question)
- "reason": a concise chain-of-thought explanation for your score.

JSON response:`, question, answer)

	return e.judgeJSON(ctx, prompt)
}

// ── 2. Context Relevance Feedback ─────────────────────────────────────────────
//
// Python equivalent:
//
//	f_qs_relevance = (Feedback(provider.qs_relevance_with_cot_reasons, name="Context Relevance")
//	    .on_input().on(context_selection).aggregate(np.mean))
func (e *Evaluator) ScoreContextRelevance(ctx context.Context, question, contextText string) (FeedbackResult, error) {
	prompt := fmt.Sprintf(`You are an expert evaluator assessing Context Relevance in a RAG system.

Evaluate how relevant the retrieved context is to answering the user question.
Assess whether the retrieved information is useful, on-topic, or contains excessive irrelevant noise.

Question:
%s

Retrieved Context:
%s

Respond with a JSON object containing:
- "score": a number from 0.0 (completely irrelevant noise) to 1.0 (highly relevant, directly provides the needed info)
- "reason": a concise chain-of-thought explanation for your score.

JSON response:`, question, contextText)

	return e.judgeJSON(ctx, prompt)
}

// ── 3. Groundedness Feedback ──────────────────────────────────────────────────
//
// Python equivalent:
//
//	grounded = Groundedness(groundedness_provider=provider)
//	f_groundedness = (Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
//	    .on(context_selection).on_output().aggregate(grounded.grounded_statements_aggregator))
func (e *Evaluator) ScoreGroundedness(ctx context.Context, contextText, answer string) (FeedbackResult, error) {
	prompt := fmt.Sprintf(`You are an expert evaluator assessing Groundedness (faithfulness / hallucination detection) in a RAG system.

An answer is grounded if its factual claims are directly supported by the retrieved context.
If the answer introduces external facts, assumptions, or hallucinations not present in the context, it is not grounded.

Instructions:
1. Break down the Answer into its individual factual claims.
2. For each claim, check if it can be directly verified from the Retrieved Context.
3. Calculate the groundedness score: (number of supported claims) / (total number of claims). If there are no claims or the model says "I don't have enough context", score is 1.0.

Retrieved Context:
%s

Answer:
%s

Respond with a JSON object containing:
- "score": a number from 0.0 (completely ungrounded/hallucinated) to 1.0 (every claim is supported by context)
- "reason": a chain-of-thought breakdown listing verified statements vs unverified claims.

JSON response:`, contextText, answer)

	return e.judgeJSON(ctx, prompt)
}

// Records returns a copy of all recorded evaluations.
func (e *Evaluator) Records() []Record {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]Record, len(e.log))
	copy(out, e.log)
	return out
}

// LeaderboardRow aggregates the evaluator's current scores.
func (e *Evaluator) LeaderboardRow() LeaderboardRow {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.log) == 0 {
		return LeaderboardRow{AppID: e.appID}
	}

	var cr, gr, ar float64
	for _, r := range e.log {
		cr += r.ContextRelevance.Score
		gr += r.Groundedness.Score
		ar += r.AnswerRelevance.Score
	}
	n := float64(len(e.log))

	return LeaderboardRow{
		AppID:            e.appID,
		NumRecords:       len(e.log),
		ContextRelevance: cr / n,
		Groundedness:     gr / n,
		AnswerRelevance:  ar / n,
	}
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
		row := e.LeaderboardRow()
		if row.NumRecords == 0 {
			fmt.Printf("║ %-32s │ %-10d │ %-11s │ %-12s │ %-11s ║\n", e.appID, 0, "N/A", "N/A", "N/A")
			continue
		}
		fmt.Printf("║ %-32s │ %-10d │ %-11.3f │ %-12.3f │ %-11.3f ║\n",
			row.AppID, row.NumRecords, row.ContextRelevance, row.Groundedness, row.AnswerRelevance)
	}
	fmt.Printf("╚═══════════════════════════════════════════════════════════════════════════════════════════╝\n")
}

// PrintRecordsDetail prints each record with its Chain-of-Thought (CoT) reasoning.
//
// Python equivalent:
//
//	records[["input", "output"] + feedback]
func PrintRecordsDetail(evaluator *Evaluator) {
	records := evaluator.Records()
	fmt.Printf("\n📋 Detailed Evaluation Records with Chain-of-Thought for [%s]:\n", evaluator.appID)
	for i, r := range records {
		fmt.Printf("\n--------------------------------------------------------------------------------\n")
		fmt.Printf("[Record %d] Input: %s\n", i+1, r.Question)
		fmt.Printf("Output: %s\n\n", strings.TrimSpace(r.Answer))
		fmt.Printf("📊 Scores & Reasoning:\n")
		fmt.Printf("  • Context Relevance : %.2f — %s\n", r.ContextRelevance.Score, r.ContextRelevance.Reason)
		fmt.Printf("  • Groundedness      : %.2f — %s\n", r.Groundedness.Score, r.Groundedness.Reason)
		fmt.Printf("  • Answer Relevance  : %.2f — %s\n", r.AnswerRelevance.Score, r.AnswerRelevance.Reason)
	}
	fmt.Printf("--------------------------------------------------------------------------------\n")
}

// SaveJSON writes all evaluation records to a JSON file.
func (e *Evaluator) SaveJSON(path string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("eval: create %q: %w", path, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(e.log)
}

// jsonExtractRegexp extracts content between curly braces if LLM outputs markdown formatting.
var jsonExtractRegexp = regexp.MustCompile(`(?s)\{.*?\}`)

func (e *Evaluator) judgeJSON(ctx context.Context, prompt string) (FeedbackResult, error) {
	resp, err := e.judge.Complete(ctx, []llm.Message{
		{
			Role:    llm.RoleSystem,
			Content: "You are an objective evaluator. You MUST respond with a valid JSON object containing 'score' (number between 0.0 and 1.0) and 'reason' (string explanation).",
		},
		{Role: llm.RoleUser, Content: prompt},
	})
	if err != nil {
		return FeedbackResult{}, err
	}

	clean := strings.TrimSpace(resp)
	// Try parsing directly
	var res struct {
		Score  float64 `json:"score"`
		Reason string  `json:"reason"`
	}

	if err := json.Unmarshal([]byte(clean), &res); err != nil {
		// If wrapped in ```json ... ```, extract with regexp
		match := jsonExtractRegexp.FindString(clean)
		if match != "" {
			if mErr := json.Unmarshal([]byte(match), &res); mErr == nil {
				return clampFeedback(res.Score, res.Reason), nil
			}
		}
		// Fallback: try scanning score as plain number
		var s float64
		if _, pErr := fmt.Sscanf(clean, "%f", &s); pErr == nil {
			return clampFeedback(s, clean), nil
		}
		return FeedbackResult{Score: 0.5, Reason: clean}, nil
	}

	return clampFeedback(res.Score, res.Reason), nil
}

func clampFeedback(score float64, reason string) FeedbackResult {
	if score < 0.0 {
		score = 0.0
	}
	if score > 1.0 {
		score = 1.0
	}
	return FeedbackResult{
		Score:  score,
		Reason: strings.TrimSpace(reason),
	}
}
