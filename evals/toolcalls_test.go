//go:build evals

// Scenario: Multi-Step Tool Chaining & Structured Output Orchestration
//
// This live evaluation test verifies Crux's end-to-end tool orchestration capabilities across
// all supported model providers (OpenAI, Anthropic, Gemini, xAI, OpenRouter). It exercises
// the complete tool lifecycle: schema reflection, multi-step dependency chaining, and response
// extraction in both raw text and structured JSON formats.
//
// Test Workflow:
// 1. Tool Declaration & Schema Generation:
//    - Registers two dependent tools using Go reflection schema generation:
//      * 'lookup_order': maps order ID (e.g., ORD-9921) to customer, carrier, and tracking number (TRK-7712).
//      * 'get_tracking_info': queries shipping status and estimated delivery for a tracking number.
//
// 2. Multi-Step Tool Chaining:
//    - Instructions require the model to first look up the order, extract the tracking number, and then
//      call tracking details before answering.
//    - Prompt: "What is the tracking number, tracking status, and estimated delivery for order ORD-9921?"
//    - Crux receives the first tool call ('lookup_order'), executes the Go handler, and appends the result to history.
//    - Crux feeds the result back to the model, which observes the tracking number and issues the second tool call ('get_tracking_info').
//    - Crux executes the second tool and returns the delivery information.
//
// 3. Unstructured vs. Structured Output Verification:
//    - Text Mode: agent.Run() verifies the final textual response contains carrier delivery details.
//    - Structured Mode: agent.RunInto() validates that Crux constrains or deserializes the model's final
//      synthesis directly into a typed Go struct (DeliveryReport) with validated fields.
//
// 4. History Auditing:
//    - Verifies via agent.Logs() that both tool calls were formally recorded with correct arguments.

package evals

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"crux.foo"
	"github.com/stretchr/testify/require"
)

type OrderLookupInput struct {
	OrderID string `json:"order_id"`
}

type OrderDetails struct {
	OrderID        string `json:"order_id"`
	Customer       string `json:"customer"`
	Carrier        string `json:"carrier"`
	TrackingNumber string `json:"tracking_number"`
}

type TrackingInput struct {
	TrackingNumber string `json:"tracking_number"`
}

type TrackingInfo struct {
	TrackingNumber    string `json:"tracking_number"`
	Status            string `json:"status"`
	EstimatedDelivery string `json:"estimated_delivery"`
}

type DeliveryReport struct {
	OrderID           string `json:"order_id"`
	Carrier           string `json:"carrier"`
	TrackingNumber    string `json:"tracking_number"`
	Status            string `json:"status"`
	EstimatedDelivery string `json:"estimated_delivery"`
}

func setupOrderTools(t *testing.T) crux.ToolsRegistry {
	t.Helper()
	reg := crux.NewToolsRegistry()

	crux.RegisterToolWithRegistry(reg, "lookup_order", "Lookup order details by order ID", func(ctx context.Context, in OrderLookupInput) (OrderDetails, *crux.StateDelta, error) {
		if in.OrderID == "ORD-9921" {
			return OrderDetails{
				OrderID:        "ORD-9921",
				Customer:       "Alice",
				Carrier:        "SpeedyEx",
				TrackingNumber: "TRK-7712",
			}, nil, nil
		}
		return OrderDetails{}, nil, fmt.Errorf("order %q not found", in.OrderID)
	})

	crux.RegisterToolWithRegistry(reg, "get_tracking_info", "Get shipping tracking information by tracking number", func(ctx context.Context, in TrackingInput) (TrackingInfo, *crux.StateDelta, error) {
		if in.TrackingNumber == "TRK-7712" {
			return TrackingInfo{
				TrackingNumber:    "TRK-7712",
				Status:            "Out for Delivery",
				EstimatedDelivery: "Tomorrow by 5 PM",
			}, nil, nil
		}
		return TrackingInfo{}, nil, fmt.Errorf("tracking number %q not found", in.TrackingNumber)
	})

	return reg
}

func Test_ExecuteToolCallsPrompt(t *testing.T) {
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
			executeToolCallsPrompt(t, modelname)
		})

		t.Run(modelname+"_structured", func(t *testing.T) {
			executeToolCallsPromptStructuredOutput(t, modelname)
		})
	}
}

const orderAssistantInstructions = `You are an order support assistant. Always use the available tools to lookup order and tracking details.
First lookup the order to find the carrier and tracking number, then query tracking details to obtain the current status and estimated delivery.
In your final response, always explicitly include the tracking number, tracking status, and estimated delivery.
Never invent tracking numbers or dates.`

func executeToolCallsPrompt(t *testing.T, modelname string) {
	t.Parallel()

	tools := setupOrderTools(t)
	agent, err := crux.New(
		"order-assistant",
		modelname,
		crux.WithInstructions(orderAssistantInstructions),
		crux.WithToolsRegistry([]string{"lookup_order", "get_tracking_info"}, tools),
		crux.WithMaxTurns(10),
		crux.WithOutputSchemaFrom[string](),
	)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	sess, err := crux.NewSession(ctx, agent)
	require.NoError(t, err)

	output, err := sess.Run(ctx, "What is the tracking number, tracking status, and estimated delivery for order ORD-9921?")
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(output), "tomorrow")
	require.Contains(t, strings.ToLower(output), "trk-7712")

	// Verify tool call chain occurred in history
	var sawOrderLookup, sawTrackingInfo bool
	for _, entry := range sess.Logs() {
		if entry.Kind == crux.KindToolCall && entry.ToolCall != nil {
			if entry.ToolCall.Name == "lookup_order" {
				sawOrderLookup = true
			}
			if entry.ToolCall.Name == "get_tracking_info" {
				sawTrackingInfo = true
				require.Contains(t, string(entry.ToolCall.Args), "TRK-7712")
			}
		}
	}
	require.True(t, sawOrderLookup, "expected lookup_order tool to be called")
	require.True(t, sawTrackingInfo, "expected get_tracking_info tool to be called")
}

func executeToolCallsPromptStructuredOutput(t *testing.T, modelname string) {
	t.Parallel()

	tools := setupOrderTools(t)
	agent, err := crux.New(
		"order-assistant",
		modelname,
		crux.WithInstructions(orderAssistantInstructions),
		crux.WithToolsRegistry([]string{"lookup_order", "get_tracking_info"}, tools),
		crux.WithMaxTurns(10),
		crux.WithOutputSchemaFrom[DeliveryReport](),
	)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	sess, err := crux.NewSession(ctx, agent)
	require.NoError(t, err)

	var report DeliveryReport
	err = sess.RunInto(ctx, &report, "What is the tracking number, tracking status, and estimated delivery for order ORD-9921?")
	require.NoError(t, err)

	require.Equal(t, "ORD-9921", report.OrderID)
	require.Equal(t, "TRK-7712", report.TrackingNumber)
	require.Contains(t, strings.ToLower(report.EstimatedDelivery), "tomorrow")

	// Verify tool call chain occurred in history
	var sawOrderLookup, sawTrackingInfo bool
	for _, entry := range sess.Logs() {
		if entry.Kind == crux.KindToolCall && entry.ToolCall != nil {
			if entry.ToolCall.Name == "lookup_order" {
				sawOrderLookup = true
			}
			if entry.ToolCall.Name == "get_tracking_info" {
				sawTrackingInfo = true
				require.Contains(t, string(entry.ToolCall.Args), "TRK-7712")
			}
		}
	}
	require.True(t, sawOrderLookup, "expected lookup_order tool to be called")
	require.True(t, sawTrackingInfo, "expected get_tracking_info tool to be called")
}
