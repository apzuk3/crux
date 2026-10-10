package crux

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// maxAttachmentSize is the largest file Run accepts, about what providers
// accept inline.
const maxAttachmentSize = 20 << 20

// Attachment is a file passed to Run, Stream or RunInto next to text:
//
//	session.Run(ctx, "What's wrong with this chart?", crux.File("q3.png"))
//
// The model receives it as an image, a document or inline text, depending on
// its type, which crux detects from the content and the name. Files are read
// when Run starts and stored in the session log, so a resumed session sends
// the same bytes. Create one with File, FileFS, Data, Reader or URL.
type Attachment struct {
	name string
	mime string
	url  string
	load func() ([]byte, error) // nil for URL
}

// File attaches the file at path.
func File(path string) Attachment {
	return Attachment{name: filepath.Base(path), load: func() ([]byte, error) {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return readAttachment(f)
	}}
}

// FileFS attaches the file name from fsys, such as an embed.FS.
func FileFS(fsys fs.FS, name string) Attachment {
	return Attachment{name: path.Base(name), load: func() ([]byte, error) {
		f, err := fsys.Open(name)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return readAttachment(f)
	}}
}

// Data attaches b. The type comes from the content; use WithName or WithMIME
// for formats it doesn't reveal, such as CSV or Markdown.
func Data(b []byte) Attachment {
	return Attachment{load: func() ([]byte, error) {
		if len(b) > maxAttachmentSize {
			return nil, errAttachmentTooLarge
		}
		return bytes.Clone(b), nil
	}}
}

// Reader attaches what r yields, such as an uploaded file. r is read once,
// when Run starts, and is not closed.
func Reader(r io.Reader) Attachment {
	return Attachment{load: func() ([]byte, error) { return readAttachment(r) }}
}

// URL attaches a file the provider downloads itself. Its type comes from the
// URL's extension; use WithMIME when it has none.
func URL(u string) Attachment {
	return Attachment{url: u}
}

// WithName names the file. The model sees the name, and its extension tells
// the type when the content doesn't.
func (a Attachment) WithName(name string) Attachment {
	a.name = name
	return a
}

// WithMIME sets the media type, such as "text/csv", instead of detecting it.
func (a Attachment) WithMIME(mimeType string) Attachment {
	a.mime = mimeType
	return a
}

var errAttachmentTooLarge = fmt.Errorf("file is larger than %d MB", maxAttachmentSize>>20)

func readAttachment(r io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, maxAttachmentSize+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxAttachmentSize {
		return nil, errAttachmentTooLarge
	}
	return data, nil
}

// part reads the attachment and returns it as it is stored in the log.
func (a Attachment) part() (ContentPart, error) {
	part, err := a.resolve()
	if err != nil {
		label := a.name
		if label == "" {
			label = a.url
		}
		if label == "" {
			return ContentPart{}, fmt.Errorf("attachment: %w", err)
		}
		return ContentPart{}, fmt.Errorf("attachment %q: %w", label, err)
	}
	return part, nil
}

func (a Attachment) resolve() (ContentPart, error) {
	mimeType, err := normalizeMIME(a.mime)
	if err != nil {
		return ContentPart{}, err
	}
	if a.url != "" {
		u, err := url.Parse(a.url)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return ContentPart{}, errors.New("not an http or https URL")
		}
		if mimeType == "" {
			mimeType = mimeByExtension[strings.ToLower(path.Ext(u.Path))]
		}
		if mimeType == "" {
			return ContentPart{}, errors.New("can't tell the file type from the URL; use WithMIME")
		}
		return ContentPart{Kind: ContentKindFile, Name: a.name, MIME: mimeType, URL: a.url}, nil
	}
	if a.load == nil {
		return ContentPart{}, errors.New("empty attachment; create it with File, FileFS, Data, Reader or URL")
	}
	data, err := a.load()
	if err != nil {
		return ContentPart{}, err
	}
	if len(data) == 0 {
		return ContentPart{}, errors.New("file is empty")
	}
	if mimeType == "" {
		mimeType = detectMIME(a.name, data)
	}
	if mimeType == "" {
		if ext := path.Ext(a.name); ext != "" {
			return ContentPart{}, fmt.Errorf("unsupported file type %q; crux sends text, JSON, YAML, images, audio, video and PDF", ext)
		}
		return ContentPart{}, errors.New("can't tell the file type; use WithName or WithMIME")
	}
	return ContentPart{Kind: ContentKindFile, Name: a.name, MIME: mimeType, Data: data}, nil
}

func normalizeMIME(mimeType string) (string, error) {
	if mimeType == "" {
		return "", nil
	}
	mediaType, _, err := mime.ParseMediaType(mimeType)
	if err != nil {
		return "", fmt.Errorf("invalid media type %q: %w", mimeType, err)
	}
	return mediaType, nil
}

// detectMIME returns the type of data. The content wins for the binary formats
// it identifies, because files are often misnamed (a JPEG saved as .png), and
// the name decides for text and everything else.
func detectMIME(name string, data []byte) string {
	sniffed, _, _ := mime.ParseMediaType(http.DetectContentType(data))
	if sniffed == "audio/wave" {
		sniffed = "audio/wav"
	}
	for _, prefix := range []string{"image/", "audio/", "video/", "application/pdf"} {
		if strings.HasPrefix(sniffed, prefix) {
			return sniffed
		}
	}
	if t := mimeByExtension[strings.ToLower(path.Ext(name))]; t != "" {
		return t
	}
	if strings.HasPrefix(sniffed, "text/") {
		return sniffed
	}
	return ""
}

// mimeByExtension lists the types providers accept. It is fixed, unlike
// mime.TypeByExtension, so detection is the same on every system.
var mimeByExtension = map[string]string{
	".txt":      "text/plain",
	".md":       "text/markdown",
	".markdown": "text/markdown",
	".csv":      "text/csv",
	".tsv":      "text/tab-separated-values",
	".html":     "text/html",
	".htm":      "text/html",
	".xml":      "text/xml",
	".json":     "application/json",
	".yaml":     "application/yaml",
	".yml":      "application/yaml",
	".pdf":      "application/pdf",
	".png":      "image/png",
	".jpg":      "image/jpeg",
	".jpeg":     "image/jpeg",
	".gif":      "image/gif",
	".webp":     "image/webp",
	".heic":     "image/heic",
	".heif":     "image/heif",
	".mp3":      "audio/mpeg",
	".wav":      "audio/wav",
	".ogg":      "audio/ogg",
	".flac":     "audio/flac",
	".m4a":      "audio/mp4",
	".aac":      "audio/aac",
	".mp4":      "video/mp4",
	".mov":      "video/quicktime",
	".webm":     "video/webm",
}
