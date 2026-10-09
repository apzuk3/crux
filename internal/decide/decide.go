// Package decide turns a Go struct into the typed questions a decision model
// answers, and the answers back into the struct. Each exported field is one
// question:
//
//   - bool is a yes or no question (noul); `true:"…"` and `false:"…"` tags
//     describe the answers;
//   - a string with a `choices:"a=when to pick a|b|c"` tag, or of a type with
//     a Choices() map[string]string method, picks one option;
//   - an integer or float with a `levels:"low|medium|high"` tag is a score:
//     integers get the nearest level's index, floats the weighted position.
//
// The `description` tag is the question's instructions and the json tag its
// name.
package decide

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"

	"crux.foo/internal/provider"
)

const (
	maxChoices = 255
	maxLevels  = 10
)

// Field is a question and the struct field that receives its answer.
type Field struct {
	provider.Question
	index []int
	float bool // a score kept as the weighted position
}

// Chooser is implemented by string types that list their own options.
type Chooser interface {
	Choices() map[string]string
}

var chooserType = reflect.TypeFor[Chooser]()

// Fields describes the questions struct type t asks.
func Fields(t reflect.Type) ([]Field, error) {
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("decision type must be a struct, got %s", t)
	}
	var fields []Field
	seen := make(map[string]bool)
	for _, sf := range reflect.VisibleFields(t) {
		name, asks, err := fieldName(sf)
		if err != nil {
			return nil, fmt.Errorf("decision type %s %w", t, err)
		}
		if !asks {
			continue
		}
		if seen[name] {
			return nil, fmt.Errorf("decision type %s has two fields named %q", t, name)
		}
		seen[name] = true
		f, err := field(sf, name)
		if err != nil {
			return nil, fmt.Errorf("field %s.%s: %w", t, sf.Name, err)
		}
		fields = append(fields, f)
	}
	if len(fields) == 0 {
		return nil, fmt.Errorf("decision type %s has no questions", t)
	}
	return fields, nil
}

// fieldName returns the question name of sf, and false when sf asks no
// question: an embedded struct (whose fields are visible on their own),
// an unexported field or one tagged "-". An embedded pointer is an error.
func fieldName(sf reflect.StructField) (string, bool, error) {
	if sf.Anonymous {
		if sf.Type.Kind() == reflect.Pointer {
			return "", false, fmt.Errorf("embeds pointer %s; embed the struct instead", sf.Type)
		}
		return "", false, nil
	}
	if !sf.IsExported() {
		return "", false, nil
	}
	name, _, _ := strings.Cut(sf.Tag.Get("json"), ",")
	if name == "-" {
		return "", false, nil
	}
	if name == "" {
		name = sf.Name
	}
	return name, true, nil
}

func field(sf reflect.StructField, name string) (Field, error) {
	f := Field{index: sf.Index}
	f.Name = name
	f.Instructions = sf.Tag.Get("description")
	switch t := sf.Type; t.Kind() {
	case reflect.Bool:
		f.Kind = provider.QuestionNoul
		f.True, f.False = sf.Tag.Get("true"), sf.Tag.Get("false")
	case reflect.String:
		options, err := choices(sf)
		if err != nil {
			return f, err
		}
		f.Kind, f.Options = provider.QuestionChoice, options
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		levels, err := levels(sf)
		if err != nil {
			return f, err
		}
		f.Kind, f.Options = provider.QuestionScore, levels
		f.float = t.Kind() == reflect.Float32 || t.Kind() == reflect.Float64
	default:
		return f, fmt.Errorf("type %s can't be decided; use bool, a string with choices or a number with levels", sf.Type)
	}
	return f, nil
}

func choices(sf reflect.StructField) ([]provider.Option, error) {
	var options []provider.Option
	if tag, ok := sf.Tag.Lookup("choices"); ok {
		for part := range strings.SplitSeq(tag, "|") {
			name, description, _ := strings.Cut(part, "=")
			options = append(options, provider.Option{Name: strings.TrimSpace(name), Description: strings.TrimSpace(description)})
		}
	} else if sf.Type.Implements(chooserType) {
		described := reflect.Zero(sf.Type).Interface().(Chooser).Choices()
		for name, description := range described {
			options = append(options, provider.Option{Name: name, Description: description})
		}
		slices.SortFunc(options, func(a, b provider.Option) int { return strings.Compare(a.Name, b.Name) })
	} else {
		return nil, errors.New(`a string needs a choices:"a|b|c" tag or a type with a Choices method; free text can't be decided`)
	}
	if err := checkOptions(options, 2, maxChoices, "choices"); err != nil {
		return nil, err
	}
	return options, nil
}

func levels(sf reflect.StructField) ([]provider.Option, error) {
	tag, ok := sf.Tag.Lookup("levels")
	if !ok {
		return nil, errors.New(`a number needs a levels:"low|medium|high" tag`)
	}
	var options []provider.Option
	for _, name := range strings.Split(tag, "|") {
		options = append(options, provider.Option{Name: strings.TrimSpace(name)})
	}
	if err := checkOptions(options, 2, maxLevels, "levels"); err != nil {
		return nil, err
	}
	return options, nil
}

func checkOptions(options []provider.Option, least, most int, what string) error {
	if len(options) < least || len(options) > most {
		return fmt.Errorf("%d %s, want %d to %d", len(options), what, least, most)
	}
	seen := make(map[string]bool, len(options))
	for _, o := range options {
		if o.Name == "" {
			return fmt.Errorf("empty name in %s", what)
		}
		if seen[o.Name] {
			return fmt.Errorf("%q appears twice in %s", o.Name, what)
		}
		seen[o.Name] = true
	}
	return nil
}

// Questions returns the fields' questions, with goal prepended to each
// question's instructions when it is set.
func Questions(fields []Field, goal string) []provider.Question {
	questions := make([]provider.Question, len(fields))
	for i, f := range fields {
		q := f.Question
		if goal != "" {
			q.Instructions = strings.TrimSpace(goal + "\n\n" + q.Instructions)
		}
		questions[i] = q
	}
	return questions
}

// Assign sets each field of the struct v points to from its answer.
func Assign(v reflect.Value, fields []Field, answers map[string]provider.Answer) error {
	for _, f := range fields {
		a, ok := answers[f.Name]
		if !ok {
			return fmt.Errorf("no answer to %q", f.Name)
		}
		dst := v.FieldByIndex(f.index)
		switch f.Kind {
		case provider.QuestionNoul:
			dst.SetBool(a.Noul >= 0.5)
		case provider.QuestionChoice:
			if !slices.ContainsFunc(f.Options, func(o provider.Option) bool { return o.Name == a.Choice }) {
				return fmt.Errorf("answer %q to %q is not one of its choices", a.Choice, f.Name)
			}
			dst.SetString(a.Choice)
		case provider.QuestionScore:
			top := float64(len(f.Options) - 1)
			if math.IsNaN(a.Score) || a.Score < 0 || a.Score > top {
				return fmt.Errorf("score %v for %q is outside its levels 0 to %v", a.Score, f.Name, top)
			}
			switch {
			case f.float:
				dst.SetFloat(a.Score)
			case dst.CanInt():
				dst.SetInt(int64(math.Round(a.Score)))
			default:
				dst.SetUint(uint64(math.Round(a.Score)))
			}
		}
	}
	return nil
}

// OutputSchema is a JSON schema for the fields' answers, for models that
// answer with structured output instead of a decision API. Scores are level
// indexes.
func OutputSchema(fields []Field, goal string) map[string]any {
	properties := make(map[string]any, len(fields))
	required := make([]string, len(fields))
	for i, f := range fields {
		properties[f.Name] = fieldProperty(f)
		required[i] = f.Name
	}
	schema := map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
	if goal != "" {
		schema["description"] = goal
	}
	return schema
}

// fieldProperty is the property schema of one question. The guide to its
// answers, one line each, is appended to the instructions as its description.
func fieldProperty(f Field) map[string]any {
	var prop map[string]any
	var guide []string
	switch f.Kind {
	case provider.QuestionNoul:
		prop, guide = booleanProperty(f)
	case provider.QuestionChoice:
		prop, guide = choiceProperty(f)
	case provider.QuestionScore:
		prop, guide = scoreProperty(f)
	}
	return withDescription(prop, f.Instructions, guide)
}

func booleanProperty(f Field) (map[string]any, []string) {
	var guide []string
	if f.True != "" {
		guide = append(guide, "true: "+f.True)
	}
	if f.False != "" {
		guide = append(guide, "false: "+f.False)
	}
	return map[string]any{"type": "boolean"}, guide
}

func choiceProperty(f Field) (map[string]any, []string) {
	names := make([]any, len(f.Options))
	var guide []string
	for j, o := range f.Options {
		names[j] = o.Name
		if o.Description != "" {
			guide = append(guide, o.Name+": "+o.Description)
		}
	}
	return map[string]any{"type": "string", "enum": names}, guide
}

func scoreProperty(f Field) (map[string]any, []string) {
	// Not an enum: some providers' schema adapters make enums strings.
	guide := make([]string, len(f.Options))
	for j, o := range f.Options {
		guide[j] = fmt.Sprintf("%d: %s", j, o.Name)
	}
	return map[string]any{"type": "integer", "minimum": 0, "maximum": len(f.Options) - 1}, guide
}

func withDescription(prop map[string]any, instructions string, guide []string) map[string]any {
	description := instructions
	if len(guide) > 0 {
		description = strings.TrimSpace(description + "\n" + strings.Join(guide, "\n"))
	}
	if description != "" {
		prop["description"] = description
	}
	return prop
}
