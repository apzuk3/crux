package decide

import (
	"reflect"
	"strings"
	"testing"

	"crux.foo/internal/provider"
)

type base struct {
	Spam bool `json:"spam"`
}

type ticket struct {
	base
	Lang    string `json:"lang" choices:"en|fr = French"`
	Skipped string `json:"-"`
	Level   uint8  `json:"level" levels:"low|high"`
}

func TestFields(t *testing.T) {
	fields, err := Fields(reflect.TypeFor[ticket]())
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range fields {
		names = append(names, f.Name)
	}
	if got := strings.Join(names, ","); got != "spam,lang,level" {
		t.Fatalf("fields = %s", got)
	}
	if got := fields[1].Options; !reflect.DeepEqual(got, []provider.Option{{Name: "en"}, {Name: "fr", Description: "French"}}) {
		t.Fatalf("options = %v", got)
	}

	var v ticket
	err = Assign(reflect.ValueOf(&v).Elem(), fields, map[string]provider.Answer{
		"spam":  {Noul: 0.2},
		"lang":  {Choice: "fr"},
		"level": {Score: 0.6},
	})
	if err != nil {
		t.Fatal(err)
	}
	if v.Spam || v.Lang != "fr" || v.Level != 1 {
		t.Fatalf("assigned %+v", v)
	}
}

func TestFieldsLimits(t *testing.T) {
	choices := make([]string, maxChoices+1)
	for i := range choices {
		choices[i] = "c" + strings.Repeat("x", i)
	}
	tests := map[string]reflect.Type{
		"256 choices": reflect.StructOf([]reflect.StructField{{Name: "A", Type: reflect.TypeFor[string](), Tag: reflect.StructTag(`choices:"` + strings.Join(choices, "|") + `"`)}}),
		"11 levels":   reflect.StructOf([]reflect.StructField{{Name: "A", Type: reflect.TypeFor[int](), Tag: `levels:"0|1|2|3|4|5|6|7|8|9|10"`}}),
		"twice":       reflect.StructOf([]reflect.StructField{{Name: "A", Type: reflect.TypeFor[string](), Tag: `choices:"a|a"`}}),
		"slice":       reflect.StructOf([]reflect.StructField{{Name: "A", Type: reflect.TypeFor[[]string]()}}),
		"no fields":   reflect.TypeFor[struct{}](),
		"not struct":  reflect.TypeFor[string](),
	}
	for name, typ := range tests {
		if _, err := Fields(typ); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
