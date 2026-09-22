# Web search smoke test

Run the same live-search prompt against each supported provider:

```sh
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
go run ./cmd/websearch
```

| Provider | Default model | Credentials |
| --- | --- | --- |
| `openai` | `gpt-4.1-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| `google` (alias `gemini`) | `gemini-2.5-flash` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

The command also accepts the library's `_APIKEY` and `_KEY` aliases, plus
`ANTHROPIC_AUTH_TOKEN`. Providers with missing credentials are skipped.
OpenRouter is skipped because its adapter rejects native web search; DeepSeek
is skipped because it has no execution adapter.

```sh
go run ./cmd/websearch -provider anthropic
go run ./cmd/websearch -provider openai -model gpt-4.1 -timeout 3m
go run ./cmd/websearch -provider google -prompt "Search for the latest Go release and cite official sources."
go run ./cmd/websearch -provider anthropic -base-url https://your-proxy.example.com
go run ./cmd/websearch -help
```

`-provider` defaults to `all`; `-timeout` defaults to two minutes per provider.
`-model` and `-base-url` require a single supported provider. The default prompt
asks for the latest stable Go release as of the current UTC date.

Each attempted request prints its model, elapsed time, response or error, followed
by a summary. Failures do not prevent subsequent providers from running.
Exit status is 0 when at least one request succeeds and none fail, 1 when any
request fails or all providers are skipped, and 2 for invalid arguments.

**This is a smoke test:** success means a nonempty answer was returned with
`WithWebsearchEnabled()`. The model decides whether to search, and `Agent.Run`
does not expose search events or grounding metadata, so the command cannot
verify that a search actually occurred. Inspect the answers and source URLs.
Live runs use the configured provider accounts and may incur API/search charges.
