#!/usr/bin/env bash
# Downloads Mind the Query into data/raw/Mind-the-Query.
# GitHub's zip endpoint cannot resume, so on a flaky connection we simply retry until the zip is complete.
set -u
cd "$(dirname "$0")/../data/raw" || exit 1
rm -rf Mind-the-Query Mind-the-Query-main mtq.zip

for i in $(seq 1 30); do
  echo "attempt $i"
  rm -f mtq.zip
  curl -sS -L --retry 3 -o mtq.zip https://codeload.github.com/endeavorXx/Mind-the-Query/zip/refs/heads/main \
    && unzip -tq mtq.zip >/dev/null 2>&1 && break
  sleep 5
done

unzip -q mtq.zip && mv Mind-the-Query-main Mind-the-Query && rm mtq.zip
du -sh Mind-the-Query && ls Mind-the-Query
