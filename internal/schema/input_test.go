package schema

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

type address struct {
	Street string `json:"street" description:"Street and number"`
	City   string `json:"city,omitempty"`
}

type audit struct {
	CreatedBy string `json:"created_by"`
	hidden    int
}

type color int

func (c *color) UnmarshalText(text []byte) error { return nil }

type blob struct{ Raw json.RawMessage }

func (b *blob) UnmarshalJSON(data []byte) error { return nil }

type person struct {
	audit
	*address
	Name     string         `json:"name" description:"Full name"`
	Age      int            `json:"age,string"`
	Score    *float64       `json:"score"`
	Tags     []string       `json:"tags,omitempty"`
	Meta     map[string]int `json:"meta"`
	Children []person       `json:"children"`
	Born     time.Time      `json:"born"`
	Favorite color          `json:"favorite"`
	Extra    blob           `json:"extra"`
	Pair     [2]float64     `json:"pair"`
	Data     []byte         `json:"data"`
	Any      any            `json:"any"`
	Skipped  string         `json:"-"`
	Untagged bool
	lower    string
	Nested   struct {
		Deep map[string]address `json:"deep"`
	} `json:"nested"`
}

type ConflictA struct {
	Name string `json:"name"`
}

type ConflictB struct {
	Name string `json:"name"`
}

// go vet rejects repeated json tags in source, so the conflicting types are
// built at run time.
func embedding(embedded []reflect.Type, own ...reflect.StructField) reflect.Type {
	var fields []reflect.StructField
	for _, t := range embedded {
		fields = append(fields, reflect.StructField{Name: t.Name(), Type: t, Anonymous: true})
	}
	return reflect.StructOf(append(fields, own...))
}

var (
	stringType = reflect.TypeFor[string]()
	// Two embedded fields at the same depth cancel each other out.
	conflicts = embedding([]reflect.Type{reflect.TypeFor[ConflictA](), reflect.TypeFor[ConflictB]()},
		reflect.StructField{Name: "Other", Type: stringType, Tag: `json:"other"`},
		reflect.StructField{Name: "Own", Type: stringType, Tag: `json:"own"`})
	// The shallower field wins the name.
	winner = embedding([]reflect.Type{reflect.TypeFor[ConflictA]()},
		reflect.StructField{Name: "Name", Type: stringType, Tag: `json:"name" description:"outer"`})
)

type withDefaults struct {
	Limit int `json:"limit"`
}

func (w *withDefaults) UnmarshalJSON(data []byte) error {
	type plain withDefaults
	return json.Unmarshal(data, (*plain)(w))
}

type promoted struct {
	blob
	Name string `json:"name"`
}

// Agent IDs hash tool input schemas, so the JSON for a type must not change.
// The expected strings were captured from the schema builder before it was
// split into helpers.
func TestJSONSchemaGolden(t *testing.T) {
	const personSchema = `{"properties":{"Untagged":{"type":"boolean"},"age":{"type":"string"},"any":{},"born":{"format":"date-time","type":"string"},"children":{"items":{"type":"object"},"type":"array"},"city":{"type":"string"},"created_by":{"type":"string"},"data":{"description":"Base64-encoded bytes","type":"string"},"extra":{},"favorite":{"type":"string"},"meta":{"additionalProperties":{"type":"integer"},"type":"object"},"name":{"description":"Full name","type":"string"},"nested":{"properties":{"deep":{"additionalProperties":{"properties":{"city":{"type":"string"},"street":{"description":"Street and number","type":"string"}},"required":["street"],"type":"object"},"type":"object"}},"required":["deep"],"type":"object"},"pair":{"items":{"type":"number"},"maxItems":2,"minItems":2,"type":"array"},"score":{"type":"number"},"street":{"description":"Street and number","type":"string"},"tags":{"items":{"type":"string"},"type":"array"}},"required":["created_by","name","age","meta","children","born","favorite","extra","pair","data","any","Untagged","nested"],"type":"object"}`
	tests := []struct {
		name string
		typ  reflect.Type
		want string
	}{
		{"person", reflect.TypeFor[person](), personSchema},
		{"pointer to person", reflect.TypeFor[*person](), personSchema},
		{"conflicts", conflicts, `{"properties":{"other":{"type":"string"},"own":{"type":"string"}},"required":["other","own"],"type":"object"}`},
		{"winner", winner, `{"properties":{"name":{"description":"outer","type":"string"}},"required":["name"],"type":"object"}`},
		{"with defaults", reflect.TypeFor[withDefaults](), `{"properties":{"limit":{"type":"integer"}},"required":["limit"],"type":"object"}`},
		{"promoted", reflect.TypeFor[promoted](), `{}`},
		{"map", reflect.TypeFor[map[string]any](), `{"additionalProperties":{},"type":"object"}`},
		{"text unmarshaler", reflect.TypeFor[color](), `{"type":"string"}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(jsonSchema(tc.typ))
			if err != nil {
				t.Fatal(err)
			}
			if string(raw) != tc.want {
				t.Fatalf("schema changed\n got %s\nwant %s", raw, tc.want)
			}
		})
	}
}

type normalizeInput struct {
	Name     string            `json:"name"`
	Items    []address         `json:"items"`
	Lookup   map[string]person `json:"lookup"`
	Extra    blob              `json:"extra"`
	Grid     [][]address       `json:"grid"`
	Defaults *withDefaults     `json:"defaults"`
}

func TestNormalizeArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		typ     reflect.Type
		want    string
		wantErr string
	}{
		{name: "known keys", args: `{"name":"a","items":[{"street":"s"}]}`, want: `{"items":[{"street":"s"}],"name":"a"}`},
		{name: "unknown key", args: `{"nmae":"a"}`, wantErr: `unknown argument "nmae"`},
		{name: "case mismatch", args: `{"Name":"a"}`, wantErr: `argument "Name" must be spelled "name"`},
		{name: "nested unknown key", args: `{"items":[{"street":"s","zip":1}]}`, wantErr: `unknown argument "items.0.zip"`},
		{name: "nested case mismatch", args: `{"items":[{"Street":"s"}]}`, wantErr: `argument "items.0.Street" must be spelled "items.0.street"`},
		{name: "nulls dropped", args: `{"name":null,"items":[{"street":null,"city":"c"}]}`, want: `{"items":[{"city":"c"}]}`},
		{name: "null array items kept", args: `{"items":[null]}`, want: `{"items":[null]}`},
		{name: "map keys free", args: `{"lookup":{"Any Key":{"name":"n","Nested":null}}}`, wantErr: `argument "lookup.Any Key.Nested" must be spelled "lookup.Any Key.nested"`},
		{name: "map values checked", args: `{"lookup":{"k":{"name":"n","meta":{"x":1},"score":null}}}`, want: `{"lookup":{"k":{"meta":{"x":1},"name":"n"}}}`},
		{name: "own decoder accepts any key", args: `{"extra":{"what":"ever","deep":{"Raw":null}}}`, want: `{"extra":{"deep":{},"what":"ever"}}`},
		{name: "nested arrays", args: `{"grid":[[{"street":"s","city":null}],[]]}`, want: `{"grid":[[{"street":"s"}],[]]}`},
		{name: "pointer struct", args: `{"defaults":{"limit":1,"Limit":2}}`, want: `{"defaults":{"Limit":2,"limit":1}}`},
		{name: "top-level own decoder checked", args: `{"limit":1,"other":2}`, typ: reflect.TypeFor[withDefaults](), wantErr: `unknown argument "other"`},
		{name: "map input", args: `{"anything":{"goes":null}}`, typ: reflect.TypeFor[map[string]any](), want: `{"anything":{}}`},
		{name: "scalar", args: `"text"`, want: `"text"`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var doc any
			if err := json.Unmarshal([]byte(tc.args), &doc); err != nil {
				t.Fatal(err)
			}
			typ := tc.typ
			if typ == nil {
				typ = reflect.TypeFor[normalizeInput]()
			}
			got, err := normalizeArgs(doc, typ, "")
			if tc.wantErr != "" {
				if err == nil || err.Error() != tc.wantErr {
					t.Fatalf("err = %v, want %s", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := json.Marshal(got)
			if string(raw) != tc.want {
				t.Fatalf("normalized\n got %s\nwant %s", raw, tc.want)
			}
		})
	}
}
