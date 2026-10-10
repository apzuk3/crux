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
crux.RegisterTool("lookup_order", "Look an order up by its ID",
	func(ctx context.Context, in struct {
		ID string `json:"id" description:"Order ID, like A-1001"`
	}) (Order, error) {
		return orders.Get(in.ID)
	})

agent := crux.Must(crux.New("support", crux.ClaudeHaiku4_5,
	crux.WithTools([]string{"lookup_order"})))
session := crux.MustSession(crux.NewSession(ctx, agent))

answer, err := session.Run(ctx, "Where is my order A-1001?")
// Order A-1001 shipped yesterday and arrives on Friday.
```
{{< /tab >}}
{{< tab name="Structured output" >}}
```go
type Invoice struct {
	Number string    `json:"number"`
	Total  float64   `json:"total"`
	Due    time.Time `json:"due_date"`
}

agent := crux.Must(crux.New("extractor", crux.Gemini3_5Flash,
	crux.WithOutputSchemaFrom[Invoice]()))
session := crux.MustSession(crux.NewSession(ctx, agent))

var inv Invoice
err := session.RunInto(ctx, &inv, "Extract this invoice.", crux.File("invoice.pdf"))
// inv.Number == "INV-2026-00917", inv.Total == 828.24
```
{{< /tab >}}
{{< tab name="Approvals" >}}
```go
crux.RegisterTool("refund", "Refund an order", refundOrder,
	crux.WithApprovalNeeded(true))

answer, err := session.Run(ctx, "The shoes don't fit, please refund order A-1001")
// err is crux.ErrApprovalNeeded: nothing was refunded yet.

for _, call := range session.PendingApprovals() {
	fmt.Println(call.Name, string(call.Args)) // refund {"order_id":"A-1001"}
	session.Approve(ctx, call.ID)             // or session.Reject(ctx, call.ID, "why")
}
answer, err = session.Resume(ctx)
// Done! I've refunded $129.90 for order A-1001.
```
{{< /tab >}}
{{< tab name="Subagents" >}}
```go
researcher := crux.Must(crux.New("researcher", crux.Gemini3_5Flash,
	crux.WithWebSearch()))

lead := crux.Must(crux.New("lead", crux.ClaudeSonnet5_5,
	crux.WithSubAgent(researcher, "Researches a question on the web")))
session := crux.MustSession(crux.NewSession(ctx, lead))

answer, err := session.Run(ctx, "What changed in the latest Go release? Cite sources.")
// The lead calls agent_researcher, which searches and reports back with URLs.
```
{{< /tab >}}
{{< tab name="MCP" >}}
```go
linear, err := crux.ConfigureMCP(ctx, "linear",
	crux.MCPRemote("https://mcp.linear.app/mcp")) // OAuth opens in the browser
if err != nil {
	return err
}
defer linear.Close()

agent := crux.Must(crux.New("pm", crux.ClaudeSonnet5_5,
	crux.WithMCPs("linear")))
session := crux.MustSession(crux.NewSession(ctx, agent))

answer, err := session.Run(ctx, "File a bug: checkout has been down since 9am")
// Writes need approval unless the server marks the tool read-only.
```
{{< /tab >}}
{{< tab name="Decisions" >}}
```go
type Triage struct {
	Urgent bool   `json:"urgent" description:"Does the customer need help right now?"`
	Team   string `json:"team" choices:"billing|technical|sales"`
}

decider := crux.MustDecider(crux.NewDecider(crux.Jev))
res, err := crux.Decide[Triage](ctx, decider,
	"My payouts have failed for 3 days. I can't pay my staff!")
// res.Value: {Urgent: true, Team: "billing"}
// res.Confidence["team"]: 0.94
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
