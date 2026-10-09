---
title: Providers and models
weight: 100
---

The model constant picks the provider; the key comes from the environment.

| Provider | Constants (examples) | Key variable |
|---|---|---|
| OpenAI | `crux.OpenAIGPT5_4`, `crux.OpenAIGPT5_4Mini` | `OPENAI_API_KEY` |
| Anthropic | `crux.ClaudeSonnet5_5`, `crux.ClaudeHaiku4_5` | `ANTHROPIC_API_KEY` |
| Google | `crux.Gemini3_5Flash` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| xAI | `crux.XAIGrok4_20` | `XAI_API_KEY` |
| DeepSeek | `crux.DeepSeekFlash` | `DEEPSEEK_API_KEY` |
| OpenRouter | `crux.OpenRouterChatModel...`, `crux.OpenRouterAnthropic...` | `OPENROUTER_API_KEY` |
| Ollama | `crux.OllamaQwen3`, … (local, `http://localhost:11434/v1`) | none needed locally |
| TypeSafe (decisions only) | `crux.Jev` | `TYPESAFE_API_KEY` |

Run `go doc crux.foo | grep -i <provider>` for the full list.

## Choosing and connecting

- A model ID crux doesn't know: `crux.New(name, "some-model", crux.WithProvider(crux.ProviderOpenAI))`.
- `crux.WithAPIKey(key)` instead of the environment; `crux.WithBaseURL(url)` for
  proxies and compatible servers; `crux.WithMaxRetries(n)` (default 2) for
  connection errors, rate limits and server errors, waiting for the
  provider's `Retry-After` (or Gemini's `RetryInfo`) when it sends one.
- The HTTP client belongs to the session: `crux.NewSession(ctx, agent,
  crux.WithHTTPClient(c))`. Subagent sessions, spawned agents and compaction
  use it too, and `Fork` keeps it. A decider takes one with
  `d.WithHTTPClient(c)`, which returns a copy.
- An agent is the same code on every provider: switch by changing the model.

## Request settings

- `crux.WithInstructions(text)`: the system prompt.
- `crux.WithMaxTurns(n)`: model requests per run (default 10).
- `crux.WithMaxTokens(n)`, `crux.WithTemperature(t)`.
- `crux.WithReasoning(crux.ReasoningOff | ReasoningLow | ReasoningMedium | ReasoningHigh | ReasoningMax)`.
  On Anthropic, reasoning can't be combined with `WithTemperature`.
- `crux.WithToolChoice(crux.ToolChoiceRequired)` (or `ToolChoiceAuto`,
  `ToolChoiceNone`, `crux.ToolChoiceTool("name")`): applies to the first
  request after new input only, so it can't loop.
- `crux.WithParallelToolCalls(false)`: one tool call per turn (not on Gemini).
- `crux.WithWebSearch(crux.WithUserLocation(...))`: provider-run web search on
  OpenAI, Anthropic, Gemini and xAI.

A provider that can't honour a setting fails in `crux.New`, not mid-run.
