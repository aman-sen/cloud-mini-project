#!/usr/bin/env bash
# Download the trained models from the GitHub Release (if models/vN is not already on disk),
# verify the checksum and unpack into models/vN/.   Usage: scripts/fetch_release_models.sh [TAG]
set -euo pipefail
TAG="${1:-v1.0}"
REPO="${GH_REPO:-$(git config --get remote.origin.url | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')}"
for v in v1 v2; do
  [ -f "models/$v/model.safetensors" ] && { echo "models/$v present"; continue; }
  mkdir -p "models/$v" /tmp/dl
  base="https://github.com/$REPO/releases/download/$TAG"
  curl -fsSL -o "/tmp/dl/model-$v.tar.gz" "$base/model-$v.tar.gz"
  curl -fsSL -o "/tmp/dl/model-$v.sha256" "$base/model-$v.sha256"
  exp=$(cut -d' ' -f1 "/tmp/dl/model-$v.sha256"); got=$(sha256sum "/tmp/dl/model-$v.tar.gz" | cut -d' ' -f1)
  [ "$exp" = "$got" ] || { echo "CHECKSUM MISMATCH for $v"; exit 1; }
  tar -xzf "/tmp/dl/model-$v.tar.gz" -C "models/$v"
  cp "/tmp/dl/model-$v.sha256" "models/$v/model.sha256"
  echo "models/$v verified ($got)"
done
