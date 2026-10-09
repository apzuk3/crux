# Contributing

Thanks for helping. crux is a small Go toolkit whose main goal is developer experience: the common path should be obvious, and users should never have to wire many pieces together. [AGENTS.md](AGENTS.md) explains the design in detail; read it before larger changes.

## Before you start

- **Bug fixes and internal changes:** open a pull request directly.
- **New or changed public API** (anything exported from `crux`): open an issue first and describe the use case. The public API is kept small on purpose, and removing an export later is a breaking change, so API changes are discussed before code is written.
- **Security issues:** don't open an issue; see [SECURITY.md](SECURITY.md).

## Rules the code follows

- **One package.** Everything users need comes from `import "github.com/apzuk3/crux"`. Implementation details go in `internal/`, which never imports `crux`. No feature subpackages.
- **Pure Go.** `crux` must build with `CGO_ENABLED=0`; never add a dependency that needs cgo, such as a cgo SQLite driver.
- **Go 1.26.** Check a new dependency's `go` directive, and don't use newer standard-library APIs.
- **The same on every provider.** A new request option goes into `provider.Request` and all three wire implementations (`internal/provider/openai.go`, `anthropic.go`, `gemini.go`). Provider-specific rules belong in the provider's `prepare` hook in `models.go`, not in `switch` statements elsewhere.
- **The session log is the source of truth.** New lifecycle facts are log entries. `Kind` values are stored, so never renumber them; add new ones at the end.
- **Options validate.** Options return errors instead of panicking.
- **Style.** Short doc comments on exported identifiers, errors wrapped with `%w` and context, and code that matches what's around it.

## Checks

CI runs these on Linux, macOS and Windows with Go 1.26 and the latest Go. Run them before pushing:

```sh
gofmt -l .                                   # must print nothing
go vet ./... && go vet -tags evals ./evals/...
go test -race ./...
CGO_ENABLED=0 go test ./...
```

Unit tests need no network or API keys. Test agent behaviour end to end with `cruxtest`, the mock transport that speaks each provider's wire format, in `cruxtest/*_test.go`. Tests that need unexported fields go in the root package.

The live evals call real providers and need their API keys:

```sh
go test -tags evals ./evals/...
```

Run the ones your change affects if you touched provider wire code.

## Pull requests

- Keep each pull request to one change, with tests.
- Say in the description if the change breaks the API. Breaking changes are acceptable before `v0.1.0` when they make crux easier to use.
- Add a line to the Unreleased section of [CHANGELOG.md](CHANGELOG.md) for anything users will notice.

By contributing, you agree that your contributions are licensed under the [Apache 2.0 license](LICENSE).
