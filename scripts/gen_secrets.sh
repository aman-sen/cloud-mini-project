#!/usr/bin/env bash
# Generates random S3 credentials (never committed: .secrets/ is git-ignored) and creates the
# Kubernetes Secrets. Re-running keeps existing credentials.
set -euo pipefail
mkdir -p .secrets
if [ ! -f .secrets/s3.env ]; then
  printf 'S3_ACCESS_KEY=%s\nS3_SECRET_KEY=%s\n' "$(openssl rand -hex 8)" "$(openssl rand -hex 20)" > .secrets/s3.env
fi
# shellcheck disable=SC1091
. .secrets/s3.env
for ns in data pipeline app; do
  kubectl -n "$ns" create secret generic s3-credentials \
    --from-literal=S3_ACCESS_KEY="$S3_ACCESS_KEY" --from-literal=S3_SECRET_KEY="$S3_SECRET_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
done
# SeaweedFS S3 identity file (same credentials)
cat > .secrets/s3.json <<JSON
{"identities":[{"name":"admin","credentials":[{"accessKey":"$S3_ACCESS_KEY","secretKey":"$S3_SECRET_KEY"}],"actions":["Admin","Read","Write","List","Tagging"]}]}
JSON
kubectl -n data create secret generic seaweedfs-s3-config --from-file=s3.json=.secrets/s3.json \
  --dry-run=client -o yaml | kubectl apply -f -
