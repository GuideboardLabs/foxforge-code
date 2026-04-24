#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

check_models() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found in PATH"
    return 1
  fi

  mapfile -t REQUIRED < <(python3 - <<'PY'
from configparser import ConfigParser
from pathlib import Path
cfg = ConfigParser()
cfg.optionxform = str
cfg.read(Path('config.ini'), encoding='utf-8')
if cfg.has_section('models'):
    seen = set()
    for _, value in cfg.items('models'):
        model = str(value).strip()
        if model and model not in seen:
            seen.add(model)
            print(model)
PY
)

  mapfile -t LOCAL < <(ollama list 2>/dev/null | awk 'NR>1 {print $1}')

  missing=0
  for model in "${REQUIRED[@]}"; do
    if printf '%s\n' "${LOCAL[@]}" | grep -Fxq "$model"; then
      echo "ok: $model"
    else
      echo "missing: $model"
      missing=$((missing+1))
    fi
  done

  if [[ $missing -gt 0 ]]; then
    echo "Model preflight failed: $missing missing model(s)."
    return 1
  fi
  echo "Model preflight passed."
  return 0
}

if [[ "${1:-}" == "--check" ]]; then
  check_models
  exit $?
fi

check_models || true
exec python3 -m SourceCode.tui.app
