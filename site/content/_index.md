---
title: crux
layout: hextra-home
---

<div class="crux-hero">
<div class="crux-hero-grid">
<div class="crux-hero-text">
<div class="hx:mt-6 hx:mb-6">
{{< hextra/hero-badge link="https://github.com/apzuk3/crux" >}}
  <span>Open source · Go 1.26+ · pure Go</span>
  {{< icon name="arrow-circle-right" attributes="height=14" >}}
{{< /hextra/hero-badge >}}
</div>

<div class="hx:mt-6 hx:mb-6">
{{< hextra/hero-headline >}}
  A small Go library&nbsp;<br class="hx:sm:block hx:hidden" />for building LLM agents
{{< /hextra/hero-headline >}}
</div>

<div class="hx:mb-8">
{{< hextra/hero-subtitle >}}
  Agents that run the same way on every major provider.&nbsp;<br class="hx:sm:block hx:hidden" />Tools are Go functions. Sessions are replayable logs.
{{< /hextra/hero-subtitle >}}
</div>

<div class="hx:mb-6">
<div class="crux-install"><span class="crux-prompt">$</span>go get crux.foo</div>

<div class="crux-actions">
  <a class="crux-btn crux-btn--primary" href="docs/">Read the docs</a>
  <a class="crux-btn crux-btn--secondary" href="https://github.com/apzuk3/crux/tree/main/examples">Examples</a>
</div>
</div>

</div>
<div class="crux-hero-code">
<img class="crux-hero-gopher" src="/brand/gophers/crux-wizard.svg" alt="A gopher wizard with a magic wand" width="200" height="256">

{{< tabs >}}
{{< tab name="Tools" selected=true >}}
```go
crux.RegisterTool("get_weather", "Get the current weather",
	func(ctx context.Context, in struct {
		City string `json:"city" description:"City name"`
	}) (string, error) {
		return "Sunny, 22°C in " + in.City, nil
	})

agent := crux.Must(crux.New("assistant", crux.ClaudeHaiku4_5,
	crux.WithTools([]string{"get_weather"})))
session := crux.MustSession(crux.NewSession(ctx, agent))

answer, err := session.Run(ctx, "What's the weather in Paris?")
fmt.Println(answer) // It's sunny and 22°C in Paris.
```
{{< /tab >}}
{{< tab name="Structured output" >}}
```go
type Ticket struct {
	Title    string `json:"title"`
	Priority string `json:"priority" jsonschema:"enum=low,enum=high"`
}

agent := crux.Must(crux.New("triage", crux.OpenAIGPT5_4Mini,
	crux.WithOutputSchemaFrom[Ticket]()))
session := crux.MustSession(crux.NewSession(ctx, agent))

var t Ticket
err := session.RunInto(ctx, &t, "The checkout page has been down since 9am!")
fmt.Println(t.Title, t.Priority) // Checkout page down high
```
{{< /tab >}}
{{< tab name="Approvals" >}}
```go
crux.RegisterTool("refund", "Refund an order", refundOrder,
	crux.WithApprovalNeeded(true))

agent := crux.Must(crux.New("support", crux.ClaudeHaiku4_5,
	crux.WithTools([]string{"refund"})))
session := crux.MustSession(crux.NewSession(ctx, agent))

answer, err := session.Run(ctx, "Please refund order 42")
if errors.Is(err, crux.ErrApprovalNeeded) {
	call := session.PendingApprovals()[0] // refund {"order_id":"42"}
	session.Approve(ctx, call.ID)         // or session.Reject
	answer, err = session.Resume(ctx)
}
```
{{< /tab >}}
{{< tab name="MCP" >}}
```go
linear, err := crux.ConfigureMCP(ctx, "linear",
	crux.MCPRemote("https://mcp.linear.app/mcp")) // OAuth in the browser
if err != nil {
	return err
}
defer linear.Close()

agent := crux.Must(crux.New("pm", crux.ClaudeSonnet5_5,
	crux.WithMCPs("linear")))
session := crux.MustSession(crux.NewSession(ctx, agent))

answer, err := session.Run(ctx, "File an issue: checkout is down since 9am")
```
{{< /tab >}}
{{< tab name="Decisions" >}}
```go
type Triage struct {
	Urgent bool   `json:"urgent" description:"Needs help right now?"`
	Area   string `json:"area" choices:"billing|technical|sales"`
}

decider := crux.MustDecider(crux.NewDecider(crux.Jev,
	crux.WithInstructions("You triage support tickets.")))
res, err := crux.Decide[Triage](ctx, decider,
	"My payouts have failed for 3 days. I can't pay my staff!")
fmt.Println(res.Value.Area, res.Confidence["area"]) // billing 0.88
```
{{< /tab >}}
{{< /tabs >}}

</div>
</div>
</div>

<section class="crux-features">
<div class="crux-features-head">
<h2>Everything you need to build agents in Go</h2>
<p>crux handles providers, tools, sessions and approvals, so you can focus on what your agent does.</p>
</div>
<div class="crux-grid">
<a class="crux-card" href="docs/providers/">
<h3>{{< icon name="switch-horizontal" >}}<span>Every provider, one API</span></h3>
<p>OpenAI, Anthropic, Gemini, xAI, DeepSeek, OpenRouter and Ollama. Switch by changing the model constant.</p>
</a>
<a class="crux-card" href="docs/tools/">
<h3>{{< icon name="code" >}}<span>Tools are Go functions</span></h3>
<p>Register once, select by name. The JSON schema comes from your input struct, and arguments are validated strictly.</p>
</a>
<a class="crux-card" href="docs/sessions/">
<h3>{{< icon name="database" >}}<span>Sessions are logs</span></h3>
<p>Every run, turn and tool call is a stored entry. Resume, fork onto another provider, or persist with GORM.</p>
</a>
<a class="crux-card" href="docs/approvals/">
<h3>{{< icon name="shield-check" >}}<span>Humans in the loop</span></h3>
<p>Mark tools as needing approval. Runs pause, you approve or reject, and they resume, even across restarts.</p>
</a>
<a class="crux-card" href="docs/mcp/">
<h3>{{< icon name="puzzle" >}}<span>MCP, skills and toolsets</span></h3>
<p>Connect MCP servers with OAuth, give agents Agent Skills, or use the built-in filesystem and network tools.</p>
</a>
<a class="crux-card" href="docs/decisions/">
<h3>{{< icon name="adjustments" >}}<span>Typed decisions</span></h3>
<p>Classify, route and score into a Go struct with Decide[T], with calibrated probabilities on decision models.</p>
</a>
<a class="crux-card" href="docs/multi-agent/">
<h3>{{< icon name="users" >}}<span>Agents that delegate</span></h3>
<p>Give an agent subagents as tools, or let it spawn new agents on the fly, with approvals surfacing on the parent.</p>
</a>
<a class="crux-card" href="docs/structured-output/">
<h3>{{< icon name="lightning-bolt" >}}<span>Streaming and structure</span></h3>
<p>Stream text and reasoning as it arrives, or decode the answer straight into a Go type with RunInto.</p>
</a>
<a class="crux-card" href="docs/testing/">
<h3>{{< icon name="beaker" >}}<span>Test without keys</span></h3>
<p>cruxtest mocks every provider's API, so agent tests run offline and still exercise the real request code.</p>
</a>
</div>
</section>
