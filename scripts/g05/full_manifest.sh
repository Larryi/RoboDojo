#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?output manifest path required}"
shift
[[ "$#" -gt 0 ]] || { echo 'pass one or more roots' >&2; exit 2; }
mkdir -p "$(dirname "${OUT}")"
{
  echo "# g05 transfer manifest; generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for root in "$@"; do
    [[ -e "${root}" ]] || { echo "missing ${root}" >&2; exit 2; }
    find "${root}" -type f -print0 | sort -z | xargs -0 sha256sum
  done
} > "${OUT}"
echo "wrote ${OUT}"
