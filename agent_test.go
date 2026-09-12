package crux

import (
	"testing"

	"github.com/invopop/jsonschema"
)

type weatherReport struct {
	Temperature float64 `json:"temperature" jsonschema:"description=Temperature in Celsius"`
	Conditions  string  `json:"conditions" jsonschema:"description=Weather conditions"`
}

func TestNewAgentOutputSchema(t *testing.T) {
	agent := NewAgent("gpt-4o", WithOutputSchemaFrom[weatherReport]())
	if agent.OutputSchema() == nil {
		t.Fatal("expected outputSchema to be non-nil for struct")
	}

	if agent.OutputSchema().Ref == "" && agent.OutputSchema().Properties == nil {
		t.Fatal("expected schema to have ref or properties")
	}

	// For string, outputSchema should be nil
	stringAgent := NewAgent("gpt-4o", WithOutputSchemaFrom[string]())
	if stringAgent.OutputSchema() != nil {
		t.Fatalf("expected nil outputSchema for string, got %+v", stringAgent.OutputSchema())
	}

	// For any, outputSchema should be nil
	anyAgent := NewAgent("gpt-4o", WithOutputSchemaFrom[any]())
	if anyAgent.OutputSchema() != nil {
		t.Fatalf("expected nil outputSchema for any, got %+v", anyAgent.OutputSchema())
	}

	// Custom WithOutputSchema
	customSchema := &jsonschema.Schema{
		Type: "object",
	}
	customAgent := NewAgent("gpt-4o", WithOutputSchema(customSchema))
	if customAgent.OutputSchema() != customSchema {
		t.Fatalf("expected custom outputSchema, got %+v", customAgent.OutputSchema())
	}

	// Default agent without output schema option
	defaultAgent := NewAgent("gpt-4o")
	if defaultAgent.OutputSchema() != nil {
		t.Fatalf("expected nil outputSchema for default agent, got %+v", defaultAgent.OutputSchema())
	}
}

func TestDecodeOutput(t *testing.T) {
	// String decoding
	str, err := decodeOutput[string]("hello world")
	if err != nil || str != "hello world" {
		t.Fatalf("decode string failed: %v, %q", err, str)
	}

	// Any decoding
	val, err := decodeOutput[any]("raw text")
	if err != nil || val != "raw text" {
		t.Fatalf("decode any failed: %v, %v", err, val)
	}

	// Struct decoding
	report, err := decodeOutput[weatherReport](`{"temperature": 22.5, "conditions": "sunny"}`)
	if err != nil {
		t.Fatalf("decode struct failed: %v", err)
	}
	if report.Temperature != 22.5 || report.Conditions != "sunny" {
		t.Fatalf("unexpected struct values: %+v", report)
	}
}
