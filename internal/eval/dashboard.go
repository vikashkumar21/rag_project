package eval

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
)

// DashboardRegistry aggregates evaluators and serves the interactive web dashboard.
//
// Python equivalent:
//
//	tru = Tru()
//	tru.run_dashboard()   # launches on http://localhost:8501/
type DashboardRegistry struct {
	mu         sync.RWMutex
	evaluators []*Evaluator
}

var defaultRegistry = &DashboardRegistry{}

// RegisterEvaluator adds an evaluator to the dashboard registry.
func RegisterEvaluator(e *Evaluator) {
	defaultRegistry.mu.Lock()
	defer defaultRegistry.mu.Unlock()
	defaultRegistry.evaluators = append(defaultRegistry.evaluators, e)
}

// StartDashboard starts an HTTP server serving the interactive RAG Triad dashboard.
// It matches the TruLens Streamlit dashboard interface on http://localhost:8501/.
func StartDashboard(addr string) error {
	mux := http.NewServeMux()

	mux.HandleFunc("/", handleHome)
	mux.HandleFunc("/api/records", handleAPIRecords)
	mux.HandleFunc("/api/leaderboard", handleAPILeaderboard)

	fmt.Printf("\n🚀 TruLens-style RAG Triad Dashboard running at http://%s/\n", addr)
	return http.ListenAndServe(addr, mux)
}

func handleHome(w http.ResponseWriter, r *http.Request) {
	defaultRegistry.mu.RLock()
	evals := defaultRegistry.evaluators
	defaultRegistry.mu.RUnlock()

	var rows []LeaderboardRow
	var allRecords []Record
	for _, e := range evals {
		rows = append(rows, e.LeaderboardRow())
		allRecords = append(allRecords, e.Records()...)
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprintf(w, dashboardHTML, renderLeaderboardRows(rows), renderRecordsRows(allRecords))
}

func handleAPIRecords(w http.ResponseWriter, r *http.Request) {
	defaultRegistry.mu.RLock()
	defer defaultRegistry.mu.RUnlock()

	var all []Record
	for _, e := range defaultRegistry.evaluators {
		all = append(all, e.Records()...)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(all)
}

func handleAPILeaderboard(w http.ResponseWriter, r *http.Request) {
	defaultRegistry.mu.RLock()
	defer defaultRegistry.mu.RUnlock()

	var rows []LeaderboardRow
	for _, e := range defaultRegistry.evaluators {
		rows = append(rows, e.LeaderboardRow())
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(rows)
}

func renderLeaderboardRows(rows []LeaderboardRow) string {
	if len(rows) == 0 {
		return `<tr><td colspan="5" style="text-align:center; padding:20px; color:#888;">No evaluation records recorded yet.</td></tr>`
	}
	var out string
	for _, r := range rows {
		out += fmt.Sprintf(`
		<tr>
			<td style="font-weight:600; color:#1e293b;">%s</td>
			<td style="text-align:center;">%d</td>
			<td><span class="badge %s">%.2f</span></td>
			<td><span class="badge %s">%.2f</span></td>
			<td><span class="badge %s">%.2f</span></td>
		</tr>`,
			r.AppID,
			r.NumRecords,
			scoreClass(r.ContextRelevance), r.ContextRelevance,
			scoreClass(r.Groundedness), r.Groundedness,
			scoreClass(r.AnswerRelevance), r.AnswerRelevance,
		)
	}
	return out
}

func renderRecordsRows(records []Record) string {
	if len(records) == 0 {
		return `<tr><td colspan="6" style="text-align:center; padding:20px; color:#888;">No records to display.</td></tr>`
	}
	var out string
	for i, r := range records {
		out += fmt.Sprintf(`
		<tr class="record-row">
			<td style="font-size:12px; color:#64748b;">#%d</td>
			<td style="font-size:13px; font-weight:600; color:#334155;">%s</td>
			<td style="font-size:14px; max-width:260px;"><strong>%s</strong></td>
			<td style="font-size:13px; max-width:320px; color:#475569;">%s</td>
			<td style="white-space:nowrap;">
				<div style="font-size:12px; margin-bottom:4px;">
					Ctx: <span class="badge %s">%.2f</span>
				</div>
				<div style="font-size:12px; margin-bottom:4px;">
					Grd: <span class="badge %s">%.2f</span>
				</div>
				<div style="font-size:12px;">
					Ans: <span class="badge %s">%.2f</span>
				</div>
			</td>
			<td>
				<details>
					<summary style="cursor:pointer; color:#3b82f6; font-size:13px;">View CoT Reasons</summary>
					<div class="cot-box">
						<p><strong>Context Relevance:</strong> %s</p>
						<p><strong>Groundedness:</strong> %s</p>
						<p><strong>Answer Relevance:</strong> %s</p>
					</div>
				</details>
			</td>
		</tr>`,
			i+1,
			r.AppID,
			r.Question,
			r.Answer,
			scoreClass(r.ContextRelevance.Score), r.ContextRelevance.Score,
			scoreClass(r.Groundedness.Score), r.Groundedness.Score,
			scoreClass(r.AnswerRelevance.Score), r.AnswerRelevance.Score,
			r.ContextRelevance.Reason,
			r.Groundedness.Reason,
			r.AnswerRelevance.Reason,
		)
	}
	return out
}

func scoreClass(s float64) string {
	if s >= 0.8 {
		return "badge-good"
	}
	if s >= 0.5 {
		return "badge-medium"
	}
	return "badge-low"
}

const dashboardHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Triad Leaderboard — Go TruLens</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: #f8fafc; color: #1e293b; padding: 32px 24px; }
  .container { max-width: 1200px; margin: 0 auto; }
  header { margin-bottom: 28px; }
  h1 { font-size: 26px; font-weight: 700; color: #0f172a; margin-bottom: 6px; }
  p.subtitle { color: #64748b; font-size: 14px; }
  .card { background: white; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 28px; overflow: hidden; }
  .card-header { padding: 16px 20px; border-bottom: 1px solid #f1f5f9; background: #fafafa; font-weight: 600; font-size: 16px; color: #334155; }
  table { width: 100%%; border-collapse: collapse; text-align: left; }
  th { background: #f8fafc; padding: 12px 16px; font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #e2e8f0; }
  td { padding: 14px 16px; border-bottom: 1px solid #f1f5f9; font-size: 14px; vertical-align: top; }
  .badge { display: inline-block; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; }
  .badge-good { background: #dcfce7; color: #15803d; }
  .badge-medium { background: #fef9c3; color: #a16207; }
  .badge-low { background: #fee2e2; color: #b91c1c; }
  .cot-box { margin-top: 8px; padding: 12px; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 12px; color: #334155; }
  .cot-box p { margin-bottom: 6px; }
  .cot-box p:last-child { margin-bottom: 0; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🎯 RAG Triad Leaderboard</h1>
    <p class="subtitle">Evaluated on Context Relevance, Groundedness (Statement-level), and Answer Relevance (Chain-of-Thought)</p>
  </header>

  <div class="card">
    <div class="card-header">🏆 Pipeline Leaderboard</div>
    <table>
      <thead>
        <tr>
          <th>Pipeline / App ID</th>
          <th style="text-align:center;">Records</th>
          <th>Context Relevance</th>
          <th>Groundedness</th>
          <th>Answer Relevance</th>
        </tr>
      </thead>
      <tbody>
        %s
      </tbody>
    </table>
  </div>

  <div class="card">
    <div class="card-header">📋 Evaluation Records & Chain-of-Thought Details</div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>App</th>
          <th>User Input</th>
          <th>Generated Output</th>
          <th>Scores</th>
          <th>Chain-of-Thought (CoT)</th>
        </tr>
      </thead>
      <tbody>
        %s
      </tbody>
    </table>
  </div>
</div>
</body>
</html>`
