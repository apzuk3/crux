# Changelog

All notable changes to crux are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `Kind.String()` names log entry kinds (`tool_call`, `run_finished`, ...) when
  printed; the stored form stays numeric.

### Fixed

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
- Gemini's depleted prepaid credits (a 429 `RESOURCE_EXHAUSTED`) are reported
  as `ErrInsufficientCredits` instead of a retried rate limit, and a
  `BILLING_DISABLED` error reason is recognized.

### Changed

- Billing errors recognized by their message are matched only for the
  provider, status and code that send them, so one provider's wording no
  longer classifies another's errors.
