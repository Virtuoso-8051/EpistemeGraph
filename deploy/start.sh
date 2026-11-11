#!/bin/sh
# Container entrypoint. If a persistent volume is mounted at /data (Hugging Face Spaces persistent
# storage, a Docker volume, ...), the state folders live there so workspaces, audits, benchmark
# results and API caches survive restarts. Without /data the shipped state is used (ephemeral).
set -e
if [ -d /data ] && [ -w /data ]; then
  for d in final_model Artifacts; do
    if [ ! -e "/data/$d" ]; then cp -r "$d" "/data/$d"; fi
    rm -rf "$d" && ln -s "/data/$d" "$d"
  done
fi
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1
