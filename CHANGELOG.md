# Changelog

All notable changes to crux are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Fixed

- xAI error responses (`{"code": ..., "error": "<message>"}`) now surface their
  status and message instead of a JSON decode error, and exhausted credits are
  reported as `ErrInsufficientCredits`.
