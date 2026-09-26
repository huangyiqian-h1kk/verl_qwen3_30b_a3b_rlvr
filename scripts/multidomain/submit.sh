#!/bin/bash
set -euo pipefail
MD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$MD_ROOT"
source "$MD_ROOT/scripts/multidomain/env.sh"
python -m multidomain.submit "$@"
