package provider

import (
	"encoding/json"
	"testing"
)

func TestErrorObjectBody(t *testing.T) {
	fixed, ok := errorObjectBody([]byte(`{"code":"invalid-argument","error":"Bad model."}`))
	if !ok {
		t.Fatal("a string error should be rewritten")
	}
	var body struct {
		Code  string `json:"code"`
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(fixed, &body); err != nil {
		t.Fatal(err)
	}
	if body.Code != "invalid-argument" || body.Error.Code != "invalid-argument" || body.Error.Message != "Bad model." {
		t.Fatalf("unexpected rewrite: %s", fixed)
	}
	for _, raw := range []string{`{"error":{"message":"already an object"}}`, `not json`, `[]`, `{"message":"no error field"}`} {
		if _, ok := errorObjectBody([]byte(raw)); ok {
			t.Fatalf("%s should be left alone", raw)
		}
	}
}
