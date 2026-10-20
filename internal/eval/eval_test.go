package eval_test

import (
	"context"
	"testing"

	"github.com/vikash/rag_project/internal/eval"
	"github.com/vikash/rag_project/internal/llm"
)

// mockJudge returns synthetic JSON feedback responses for testing.
type mockJudge struct {
	response string
}

func (m *mockJudge) Complete(_ context.Context, _ []llm.Message) (string, error) {
	return m.response, nil
}

func TestEvaluatorWithMockJudge(t *testing.T) {
	ctx := context.Background()

	mock := &mockJudge{
		response: `{"score": 0.85, "reason": "The answer directly addresses the prompt and is grounded."}`,
	}

	evaluator := eval.NewEvaluator(mock, "Mock_App")

	rec, err := evaluator.Evaluate(ctx, "What is AI?", "Artificial Intelligence is...", "AI is...")
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	if rec.AppID != "Mock_App" {
		t.Errorf("expected AppID Mock_App, got %s", rec.AppID)
	}

	if rec.AnswerRelevance.Score != 0.85 {
		t.Errorf("expected score 0.85, got %f", rec.AnswerRelevance.Score)
	}

	if rec.AnswerRelevance.Reason == "" {
		t.Errorf("expected non-empty reason")
	}

	row := evaluator.LeaderboardRow()
	if row.NumRecords != 1 {
		t.Errorf("expected 1 record in leaderboard row, got %d", row.NumRecords)
	}
	if row.AnswerRelevance != 0.85 {
		t.Errorf("expected avg score 0.85, got %f", row.AnswerRelevance)
	}
}
