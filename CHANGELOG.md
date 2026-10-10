# Changelog

All notable changes to crux are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `Kind.String()` names log entry kinds (`tool_call`, `run_finished`, ...) when
  printed; the stored form stays numeric.

### Fixed

- Tool input and output schemas describe types by what decodes into them:
  types with `UnmarshalText` (`decimal.Decimal`, `uuid.UUID`, `net.IP`) are
  strings, `time.Duration` is a string such as `"90s"`, `*big.Int` an
  integer, and a `JSONSchema()` method on a type is honoured. Before, a
  decimal field in an output schema became an empty object that every model
  answered with `{}`. Schemas of affected tools change, and so do the IDs of
  agents using them.

- `RunInto` checks that each answer decodes into the target during the run, so
  `WithMaxRepairs` repairs an answer the schema accepts but Go can't decode;
  the error wraps `ErrOutputValidation`.

- An attachment with an extension crux doesn't send (`.xlsx`, `.docx`, ...)
  fails with "unsupported file type" instead of asking for a name it has.

- An output schema a provider can't take (a map field on OpenAI-style
  providers) fails in `crux.New` instead of on the first request.

- Several inputs in one `Run` reach OpenAI-style providers as separate text
  parts instead of one string with nothing between them.

- Gemini 3 agents can combine `WithWebSearch` with their own tools; the
  request now asks for server-side tool invocations, which the API requires.

- `WithReasoning` on xAI's grok-4.20 models, which pick reasoning by name and
  reject the effort, now fails in `New` instead of on the first request, and a
  fork to another provider drops reasoning and parallel-call settings.

- xAI error responses (`{"code": ..., "error": "<message>"}`) now surface their
  status and message instead of a JSON decode error, and exhausted credits are
  reported as `ErrInsufficientCredits`.
