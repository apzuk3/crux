# Images and files

Sends files to the model next to text. Set the API key for your provider
(`ANTHROPIC_API_KEY` for the default model), then run:

```sh
go run ./examples/attachments
```

The agent gets an embedded CSV (`crux.FileFS`) and a bar chart drawn in memory
(`crux.Data`). The chart was drawn from different numbers, so the model should
spot that Q3 doesn't match.

Name files on the command line to have them summarised too (`crux.File`):

```sh
go run ./examples/attachments report.pdf photo.jpg notes.md
```

Or serve an upload form, where each uploaded file is sent with `crux.Reader`:

```sh
go run ./examples/attachments -serve :8080
```

Use `-model` for another provider, such as `-model gpt-5.4` or
`-model gemini-2.5-flash`.
