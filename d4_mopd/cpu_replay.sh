#!/usr/bin/env bash
# Login-node replay needs the same C++ runtime used to build TE and launch jobs.
# Configure the loader before Python starts, not after torch has loaded a library.
set -euo pipefail
if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi
module load gcc/13.2.0 || {
  printf 'CPU replay could not load gcc/13.2.0\n' >&2
  exit 1
}
mopd_cxx=$(command -v g++)
mopd_cxx_version=$("$mopd_cxx" -dumpfullversion)
[[ "$mopd_cxx_version" == 13.2.0 ]] || {
  printf 'CPU replay requires gcc/13.2.0; found %s\n' "$mopd_cxx_version" >&2
  exit 1
}
mopd_cxx_library=$("$mopd_cxx" -print-file-name=libstdc++.so.6)
[[ "$mopd_cxx_library" == /* && -f "$mopd_cxx_library" ]] || {
  printf 'GCC module did not expose libstdc++.so.6\n' >&2
  exit 1
}
mopd_cxx_library=$(readlink -f "$mopd_cxx_library")
export LD_LIBRARY_PATH="$(dirname "$mopd_cxx_library")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export D4_MOPD_CPU_CXX_LIBRARY="$mopd_cxx_library"
export D4_MOPD_CPU_CXX_COMPILER="$mopd_cxx"
export D4_MOPD_CPU_CXX_VERSION="$mopd_cxx_version"
exec "$@"
