//go:build evals

// Scenario: Tool Error Communication & Self-Correction (Error Recovery)
//
// This live evaluation test verifies the Crux library's ability to reliably communicate
// tool execution errors back to LLM providers across different architectures (OpenAI, Anthropic,
// Gemini, xAI). Rather than aborting the session, swallowing errors, or corrupting conversation
// state, Crux must properly format and inject the tool failure into the conversation history so
// the model can observe the feedback and self-correct.
//
// Test Workflow:
// 1. Initial Tool Invocation & Failure:
//    - The agent is instructed to act as a database query assistant and use 'query_user' to look up user profiles.
//    - Prompt: "Find the profile for user 'aramp'".
//    - The model issues a tool call: query_user(identifier="aramp").
//    - The tool returns an intentional domain error:
//      "error: username format invalid, please query with email format: aramp@example.com".
//
// 2. Crux Error Handling & History Communication:
//    - Crux captures the Go error from the tool and records it as a KindToolResult entry with ToolResult.Error set.
//    - Crux faithfully translates and replays this error into provider-native message formats:
//      * OpenAI / xAI: responses/chat tool message with the error body.
//      * Anthropic: tool_result content block with is_error=true.
//      * Gemini: FunctionResponse part with {"error": ...}.
//    - Crux maintains session integrity and continues the turn cycle without terminating early.
//
// 3. Model Observation & Autonomous Recovery:
//    - Having received the error payload from Crux, the model observes the corrective guidance in history.
//    - The model adjusts its query parameter and calls 'query_user' a second time with "aramp@example.com".
//    - The tool succeeds, returning: {"user_id": 42, "role": "admin"}.
//
// 4. Final Answer & Bounds:
//    - The model produces a final response containing user ID 42 and role admin.
//    - The full multi-turn cycle (user input -> tool error -> retry -> final answer) completes within MaxTurns(5).

package evals

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"crux.foo"
	"github.com/stretchr/testify/require"
)

type QueryUserInput struct {
	Identifier string `json:"identifier" description:"The user identifier or email to query"`
	Username   string `json:"username,omitempty"`
	Email      string `json:"email,omitempty"`
}

func (q QueryUserInput) NormalizedIdentifier() string {
	if q.Identifier != "" {
		return strings.Trim(strings.ToLower(q.Identifier), "'\" \t")
	}
	if q.Email != "" {
		return strings.Trim(strings.ToLower(q.Email), "'\" \t")
	}
	return strings.Trim(strings.ToLower(q.Username), "'\" \t")
}

type UserProfile struct {
	UserID int    `json:"user_id"`
	Role   string `json:"role"`
}

func setupErrorRecoveryTools(t *testing.T) crux.ToolsRegistry {
	t.Helper()
	reg := crux.NewToolsRegistry()

	crux.RegisterToolWithRegistry(reg, "query_user", "Query user profile by identifier or email", func(ctx context.Context, in QueryUserInput) (UserProfile, *crux.StateDelta, error) {
		id := in.NormalizedIdentifier()
		switch id {
		case "aramp":
			return UserProfile{}, nil, errors.New("error: username format invalid, please query with email format: aramp@example.com")
		case "aramp@example.com":
			return UserProfile{
				UserID: 42,
				Role:   "admin",
			}, nil, nil
		default:
			return UserProfile{}, nil, fmt.Errorf("unexpected identifier: %q", in.Identifier)
		}
	})

	return reg
}

const errorRecoveryInstructions = `You are a database query assistant. Always use the 'query_user' tool to find user IDs.
If the tool returns an error indicating the format is invalid or recommending an alternative, read the error message carefully, adjust your arguments according to the error message, and retry with the correct parameter.`

const errorRecoveryPrompt = "Find the profile for user 'aramp'."

func Test_ExecuteErrorRecoveryPrompt(t *testing.T) {
	t.Parallel()

	for _, modelname := range []string{
		crux.Gemini3_5FlashLite,
		crux.OpenAIGPT5_6Sol,
		crux.ClaudeHaiku4_5,
		crux.XAIGrok4_20,
		crux.OpenRouterXAIGrok4_20,
		crux.OpenRouterChatModelGPT5_6Luna,
	} {
		t.Run(modelname, func(t *testing.T) {
			executeErrorRecovery(t, modelname)
		})
	}
}

func executeErrorRecovery(t *testing.T, modelname string) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	sess := newErrorRecoverySession(ctx, t, modelname)

	output, err := sess.Run(ctx, errorRecoveryPrompt)
	require.NoError(t, err, "agent should not fail when a tool returns an error")

	activity := collectToolActivity(sess.Logs())
	toolCalls, toolResults := activity.calls, activity.results

	// 1. Assert at least two tool calls and two tool results were made
	require.GreaterOrEqual(t, len(toolCalls), 2, "expected at least two tool calls")
	require.GreaterOrEqual(t, len(toolResults), 2, "expected at least two tool results")

	// 2. Assert that the first tool call fails with ToolResult.Error
	require.Equal(t, "query_user", toolCalls[0].Name)
	require.Contains(t, string(toolCalls[0].Args), "aramp")
	requireNormalizedIdentifier(t, toolCalls[0].Args, "aramp")

	require.Equal(t, toolCalls[0].ID, toolResults[0].CallID)
	require.NotEmpty(t, toolResults[0].Error, "expected first tool result to fail with ToolResult.Error")
	require.Contains(t, toolResults[0].Error, "error: username format invalid, please query with email format: aramp@example.com")
	require.Empty(t, toolResults[0].Output, "first tool result output should be empty on error")

	// 3. Assert the model observed the error in history rather than failing
	require.True(t, activity.firstResultIdx >= 0 && activity.secondCallIdx > activity.firstResultIdx,
		"model must observe the error in history before making the second tool call")

	// 4. Assert the model calls query_user a second time with the corrected argument aramp@example.com
	require.Equal(t, "query_user", toolCalls[1].Name)
	require.Contains(t, string(toolCalls[1].Args), "aramp@example.com")
	requireNormalizedIdentifier(t, toolCalls[1].Args, "aramp@example.com")

	require.Equal(t, toolCalls[1].ID, toolResults[1].CallID)
	require.Empty(t, toolResults[1].Error, "second tool call should succeed without error")
	require.Contains(t, toolResults[1].Output, "42")
	require.Contains(t, strings.ToLower(toolResults[1].Output), "admin")

	// 5. Assert final output contains user ID 42 and role admin within MaxTurns(5)
	outputLower := strings.ToLower(output)
	require.Contains(t, outputLower, "42")
	require.Contains(t, outputLower, "admin")
}

func newErrorRecoverySession(ctx context.Context, t *testing.T, modelname string) *crux.Session {
	t.Helper()
	tools := setupErrorRecoveryTools(t)
	return newSessionOrSkip(ctx, t, "database-assistant", modelname,
		crux.WithInstructions(errorRecoveryInstructions),
		crux.WithToolsRegistry([]string{"query_user"}, tools),
		crux.WithMaxTurns(5),
	)
}

// toolActivity is the tool calls and results of a log, with the positions
// that show the model saw the first result before its second call.
type toolActivity struct {
	calls          []crux.ToolCall
	results        []crux.ToolResult
	firstResultIdx int
	secondCallIdx  int
}

func collectToolActivity(logs []crux.Entry) toolActivity {
	activity := toolActivity{firstResultIdx: -1, secondCallIdx: -1}
	for idx, entry := range logs {
		if entry.Kind == crux.KindToolCall && entry.ToolCall != nil {
			activity.calls = append(activity.calls, *entry.ToolCall)
			if len(activity.calls) == 2 {
				activity.secondCallIdx = idx
			}
		}
		if entry.Kind == crux.KindToolResult && entry.ToolResult != nil {
			activity.results = append(activity.results, *entry.ToolResult)
			if len(activity.results) == 1 {
				activity.firstResultIdx = idx
			}
		}
	}
	return activity
}

// requireNormalizedIdentifier checks the identifier the model sent when the
// arguments decode; providers may use any of the aliases in QueryUserInput.
func requireNormalizedIdentifier(t *testing.T, args json.RawMessage, want string) {
	t.Helper()
	var in QueryUserInput
	if err := json.Unmarshal(args, &in); err == nil {
		require.Equal(t, want, in.NormalizedIdentifier())
	}
}
