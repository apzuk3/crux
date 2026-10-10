# Changelog

All notable changes to crux are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Fixed

- `WithReasoning` on xAI's grok-4.20 models, which pick reasoning by name and
  reject the effort, now fails in `New` instead of on the first request, and a
  fork to another provider drops reasoning and parallel-call settings.

- xAI error responses (`{"code": ..., "error": "<message>"}`) now surface their
  status and message instead of a JSON decode error, and exhausted credits are
  reported as `ErrInsufficientCredits`.
