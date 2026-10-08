package cruxtest

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/apzuk3/crux"
)

// providerDecisions marks the state-and-questions API decision models speak
// (TypeSafe's /v1/systemone, OpenRouter's /alpha/decisions).
const providerDecisions crux.Provider = "cruxtest-decisions"

func isDecisionsPath(path string) bool {
	return strings.HasSuffix(path, "/systemone") || strings.HasSuffix(path, "/decisions")
}

// Answer is one answer of a mocked decision, in the API's wire format.
type Answer map[string]any

// Noul answers a yes or no question: p is the probability of yes.
func Noul(p float64) Answer {
	return Answer{"type": "noul", "noul": p}
}

// Choice answers a choice question with choice and each option's probability.
func Choice(choice string, confidence float64, probabilities map[string]float64) Answer {
	return Answer{"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities}
}

// Score answers a score question with the weighted level index and each
// level's probability, lowest level first.
func Score(score, confidence float64, probabilities ...float64) Answer {
	probs := make(map[string]float64, len(probabilities))
	for i, p := range probabilities {
		probs[fmt.Sprint(i)] = p
	}
	return Answer{"type": "score", "score": score, "confidence": confidence, "probabilities": probs}
}

// ReturnDecision configures the turn to answer a decision model's questions,
// keyed by question name.
func (t *Turn) ReturnDecision(answers map[string]Answer) *Turn {
	t.answers = answers
	return t
}

func buildDecisionsResponse(turn *Turn) ([]byte, error) {
	if len(turn.RawBody) > 0 {
		return turn.RawBody, nil
	}
	if turn.StatusCode != 0 && turn.StatusCode != 200 {
		return json.Marshal(map[string]any{"detail": []map[string]any{{"loc": []string{"body"}, "msg": fmt.Sprintf("HTTP %d error", turn.StatusCode), "type": "error"}}})
	}
	usage := map[string]any{"input_tokens": 0, "output_tokens": 0}
	if turn.Usage != nil {
		usage = map[string]any{"input_tokens": turn.Usage.InputTokens, "output_tokens": turn.Usage.OutputTokens}
	}
	return json.Marshal(map[string]any{"model": "cruxtest-decision", "answers": turn.answers, "usage": usage})
}

func validateDecisionsRequest(body []byte) error {
	var request struct {
		Model     string                     `json:"model"`
		State     json.RawMessage            `json:"state"`
		Questions map[string]json.RawMessage `json:"questions"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return err
	}
	switch {
	case request.Model == "":
		return errors.New("model is required")
	case len(request.State) == 0 || string(request.State) == "null":
		return errors.New("state is required")
	case len(request.Questions) == 0:
		return errors.New("questions must not be empty")
	}
	return nil
}
