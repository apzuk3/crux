#!/bin/sh
# Builds the docs site and deploys it to crux.foo with Wrangler.
set -eu
cd "$(dirname "$0")"
hugo --minify --cleanDestinationDir -d ../.site-build/prod
npx --yes wrangler@4 deploy --config worker/wrangler.toml
