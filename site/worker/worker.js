// Serves the docs site at crux.foo and answers `go get crux.foo/...`.
const REPO = "https://github.com/apzuk3/crux";
const MODULE = "crux.foo";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.searchParams.get("go-get") === "1") {
      let path = url.pathname;
      while (path.endsWith("/")) path = path.slice(0, -1);
      const html = `<!doctype html>
<html><head>
<meta name="go-import" content="${MODULE} git ${REPO}">
<meta name="go-source" content="${MODULE} ${REPO} ${REPO}/tree/main{/dir} ${REPO}/blob/main{/dir}/{file}#L{line}">
</head><body>go get ${MODULE}${path}</body></html>`;
      return new Response(html, {
        headers: { "content-type": "text/html; charset=utf-8", "cache-control": "public, max-age=300" },
      });
    }
    return env.ASSETS.fetch(request);
  },
};
