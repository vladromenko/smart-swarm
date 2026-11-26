#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project"
exec "$project/smart-swarm" pilot "$@"
