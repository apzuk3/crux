package cruxtest_test

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
	"testing/fstest"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

var (
	pngBytes = []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
	pdfBytes = []byte("%PDF-1.4\n%fake\n")
	csvBytes = []byte("region,sales\nnorth,10\n")
	csvText  = "<file name=\"sales.csv\" type=\"text/csv\">\nregion,sales\nnorth,10\n\n</file>"
)

// attachmentInputs is one of each kind of file, after a line of text.
func attachmentInputs() []any {
	return []any{
		"describe these",
		crux.Data(pngBytes),
		crux.Data(pdfBytes).WithName("report.pdf"),
		crux.Data(csvBytes).WithName("sales.csv"),
		crux.URL("https://example.com/cat.jpg"),
	}
}

func TestAttachmentsOnTheWire(t *testing.T) {
	b64 := base64.StdEncoding.EncodeToString

	t.Run("openai", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		s := rulesSession(t, crux.ProviderOpenAI, "test-model", mock)
		_, err := s.Run(t.Context(), attachmentInputs()...)
		require.NoError(t, err)

		var req struct {
			Input []struct {
				Role    string           `json:"role"`
				Content []map[string]any `json:"content"`
			} `json:"input"`
		}
		require.NoError(t, json.Unmarshal(mock.Requests()[0].Body, &req))
		require.Len(t, req.Input, 1)
		require.Equal(t, "user", req.Input[0].Role)
		require.Equal(t, []map[string]any{
			{"type": "input_text", "text": "describe these"},
			{"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64," + b64(pngBytes)},
			{"type": "input_file", "filename": "report.pdf", "file_data": "data:application/pdf;base64," + b64(pdfBytes)},
			{"type": "input_text", "text": csvText},
			{"type": "input_image", "detail": "auto", "image_url": "https://example.com/cat.jpg"},
		}, req.Input[0].Content)
	})

	t.Run("anthropic", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		s := rulesSession(t, crux.ProviderAnthropic, crux.ClaudeHaiku4_5, mock)
		_, err := s.Run(t.Context(), attachmentInputs()...)
		require.NoError(t, err)

		var req struct {
			Messages []struct {
				Role    string           `json:"role"`
				Content []map[string]any `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(mock.Requests()[0].Body, &req))
		require.Len(t, req.Messages, 1)
		require.Equal(t, []map[string]any{
			{"type": "text", "text": "describe these"},
			{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": b64(pngBytes)}},
			{"type": "document", "title": "report.pdf", "source": map[string]any{"type": "base64", "media_type": "application/pdf", "data": b64(pdfBytes)}},
			{"type": "text", "text": csvText},
			{"type": "image", "source": map[string]any{"type": "url", "url": "https://example.com/cat.jpg"}},
		}, req.Messages[0].Content)
	})

	t.Run("gemini", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		s := rulesSession(t, crux.ProviderGoogle, "test-model", mock)
		_, err := s.Run(t.Context(), attachmentInputs()...)
		require.NoError(t, err)

		var req struct {
			Contents []struct {
				Role  string           `json:"role"`
				Parts []map[string]any `json:"parts"`
			} `json:"contents"`
		}
		require.NoError(t, json.Unmarshal(mock.Requests()[0].Body, &req))
		require.Len(t, req.Contents, 1)
		require.Equal(t, []map[string]any{
			{"text": "describe these"},
			{"inlineData": map[string]any{"mimeType": "image/png", "data": b64(pngBytes)}},
			{"inlineData": map[string]any{"mimeType": "application/pdf", "data": b64(pdfBytes)}},
			{"text": csvText},
			{"fileData": map[string]any{"mimeType": "image/jpeg", "fileUri": "https://example.com/cat.jpg"}},
		}, req.Contents[0].Parts)
	})
}

func TestAttachmentsAreStoredAndReplayed(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("a chart")
	mock.Expect().ReturnText("still a chart")
	s := rulesSession(t, crux.ProviderOpenAI, "test-model", mock)

	_, err := s.Run(t.Context(), crux.FileFS(fstest.MapFS{"img/q3.png": {Data: pngBytes}}, "img/q3.png"))
	require.NoError(t, err, "an attachment alone is valid input")
	_, err = s.Run(t.Context(), "and now?")
	require.NoError(t, err)

	var user crux.Entry
	for _, e := range s.Logs() {
		if e.Kind == crux.KindUser {
			user = e
			break
		}
	}
	require.Equal(t, []crux.ContentPart{{Kind: crux.ContentKindFile, Name: "q3.png", MIME: "image/png", Data: pngBytes}}, user.Content)

	require.Contains(t, mock.Requests()[1].BodyString(), base64.StdEncoding.EncodeToString(pngBytes), "history resends the bytes")
}

func TestAttachmentErrors(t *testing.T) {
	run := func(t *testing.T, provider crux.Provider, inputs ...any) error {
		t.Helper()
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		s := rulesSession(t, provider, "test-model", mock)
		_, err := s.Run(t.Context(), inputs...)
		if err != nil {
			require.Empty(t, mock.Requests(), "nothing is sent")
			require.Empty(t, s.Logs(), "nothing is recorded")
		}
		return err
	}

	tests := []struct {
		name   string
		inputs []any
		want   string
	}{
		{"missing file", []any{crux.File("testdata/missing.png")}, `attachment "missing.png": open testdata/missing.png`},
		{"binary bytes", []any{"look", pngBytes}, "wrap it in crux.Data"},
		{"unknown type", []any{crux.Data([]byte{0, 1, 2, 3})}, "can't tell the file type; use WithName or WithMIME"},
		{"unsupported extension", []any{crux.Data([]byte("PK\x03\x04fake")).WithName("sheet.xlsx")}, `unsupported file type ".xlsx"`},
		{"empty", []any{crux.Data(nil)}, "file is empty"},
		{"too large", []any{crux.Reader(strings.NewReader(strings.Repeat("x", 20<<20+1)))}, "larger than 20 MB"},
		{"bad URL", []any{crux.URL("file:///etc/passwd")}, "not an http or https URL"},
		{"URL without type", []any{crux.URL("https://example.com/download")}, "use WithMIME"},
		{"bad MIME", []any{crux.Data(csvBytes).WithMIME("not a type")}, "invalid media type"},
		{"zero value", []any{crux.Attachment{}}, "empty attachment"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.ErrorContains(t, run(t, crux.ProviderOpenAI, tt.inputs...), tt.want)
		})
	}

	t.Run("provider rejects the type", func(t *testing.T) {
		mock := cruxtest.NewMock()
		s := rulesSession(t, crux.ProviderAnthropic, crux.ClaudeHaiku4_5, mock)
		_, err := s.Run(t.Context(), crux.Data([]byte("RIFF\x00\x00\x00\x00WAVEfmt ")))
		require.ErrorContains(t, err, "anthropic does not accept audio/wav files as bytes")
		require.Empty(t, mock.Requests())
	})
}

func TestAttachmentTypeDetection(t *testing.T) {
	jpeg := []byte("\xff\xd8\xff\xe0\x00\x10JFIF")
	tests := []struct {
		name string
		in   crux.Attachment
		want string
	}{
		{"content wins over a wrong extension", crux.Data(jpeg).WithName("photo.png"), "image/jpeg"},
		{"extension for text", crux.Data([]byte("# Title\n")).WithName("notes.md"), "text/markdown"},
		{"plain text without a name", crux.Data([]byte("hello")), "text/plain"},
		{"explicit type", crux.Data(csvBytes).WithMIME("text/csv; charset=utf-8"), "text/csv"},
		{"extension when content is unknown", crux.Data([]byte{0, 1, 2}).WithName("clip.mp3"), "audio/mpeg"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry, err := crux.NewUserEntry(tt.in)
			require.NoError(t, err)
			require.Equal(t, tt.want, entry.Content[0].MIME)
		})
	}
}

func TestRunIntoTargetFirst(t *testing.T) {
	mock := cruxtest.NewMock()
	s := rulesSession(t, crux.ProviderOpenAI, "test-model", mock)
	var out struct{}
	err := s.RunInto(t.Context(), "extract", &out)
	require.ErrorContains(t, err, "RunInto takes the target before the inputs")
}
