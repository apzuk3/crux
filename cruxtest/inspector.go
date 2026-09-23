package cruxtest

import (
	"encoding/json"
	"net/http"
	"net/url"
)

// CapturedRequest stores a copy of an HTTP request intercepted by the mock.
type CapturedRequest struct {
	Method string
	URL    *url.URL
	Header http.Header
	Body   []byte
}

// BodyString returns the request body as a string.
func (r *CapturedRequest) BodyString() string {
	return string(r.Body)
}

// UnmarshalBody parses the request body JSON into v.
func (r *CapturedRequest) UnmarshalBody(v any) error {
	return json.Unmarshal(r.Body, v)
}
