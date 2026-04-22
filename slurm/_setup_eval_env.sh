#!/bin/bash
# Shared eval-side setup: fetches the per-alias chat-template jinja from the
# HuggingFace repo (one-time, via hf_hub_download which uses the on-disk
# cache) and installs a site-packages hook that overrides
# tokenizer.chat_template for every AutoTokenizer.from_pretrained in the
# Python process.
#
# Usage (inside an eval_*.sh, AFTER sourcing model_registry.sh):
#   mr_eval_setup_chat_template "$ALIAS"
#
# Also applies the PEP-585 torch.library.infer_schema patch that vLLM+Mixtral
# needs (see harmbench/slurm/eval_pair.sh for details) — same .pth delivery.

# Helper for launchers that take a positional model ref (alias or HF path).
# Prefers MR_EVAL_MODEL_NAME when the submit script set it (e.g. for per-
# checkpoint runs whose path isn't in the registry). Strips known checkpoint
# suffixes (_bs_gsm8k_<iter>, _em_incorrect_health_<iter>) so checkpoint jobs
# pick up the parent alias's chat template.
mr_eval_resolve_alias_for_chat_template() {
  local ref="${1:-}"
  local cand="${MR_EVAL_MODEL_NAME:-$ref}"
  cand="$(printf '%s' "$cand" | sed -E 's/_bs_gsm8k_[0-9]+$//; s/_em_incorrect_health_[0-9]+$//')"
  if type -t mr_eval_registry_has_alias >/dev/null 2>&1 && mr_eval_registry_has_alias "$cand"; then
    printf '%s' "$cand"
  fi
}

mr_eval_setup_chat_template() {
  local alias="$1"
  local name
  if type -t mr_eval_chat_template >/dev/null 2>&1; then
    name="$(mr_eval_chat_template "$alias")"
  fi

  # Always install the Python hook (so that the module respects the env vars
  # whether they're set or not).
  local site="/usr/local/lib/python3.12/dist-packages"
  if [[ -d "$site" ]]; then
    cat > "$site/_mr_eval_chat_template.py" <<'PY'
import os, sys

_JINJA = os.environ.get("MR_EVAL_CHAT_TEMPLATE_JINJA", "")
_NAME  = os.environ.get("MR_EVAL_CHAT_TEMPLATE_NAME", "")

if _JINJA:
    # Import transformers once at startup so we can patch its tokenizer base.
    try:
        from transformers import PreTrainedTokenizerBase as _Base
    except Exception as _exc:
        print(f"[mr_eval_chat_template] transformers unavailable: {_exc}", flush=True)
    else:
        _orig = _Base.from_pretrained.__func__
        def _wrapped(cls, *args, **kwargs):
            tok = _orig(cls, *args, **kwargs)
            try:
                tok.chat_template = _JINJA
            except Exception as e:
                print(f"[mr_eval_chat_template] could not set chat_template: {e}", flush=True)
            return tok
        _Base.from_pretrained = classmethod(_wrapped)
        # Print to stderr so the hook's output never pollutes $() captures
        # in shell helpers that themselves call python3.
        print(f"[mr_eval_chat_template] template override installed "
              f"(name={_NAME or '<unnamed>'}, {len(_JINJA)} chars)", file=sys.stderr, flush=True)
PY
    printf '%s\n' 'import _mr_eval_chat_template' > "$site/_mr_eval_chat_template.pth"
  fi

  if [[ -z "$name" ]]; then
    unset MR_EVAL_CHAT_TEMPLATE_NAME MR_EVAL_CHAT_TEMPLATE_JINJA
    echo "[chat-template] alias=$alias using default tokenizer template"
    return 0
  fi

  local repo=""
  if type -t mr_eval_chat_template_source >/dev/null 2>&1; then
    repo="$(mr_eval_chat_template_source "$alias")"
  else
    repo="${MR_EVAL_MODEL_PRETRAINED_MAP[$alias]:-}"
  fi
  if [[ -z "$repo" ]]; then
    echo "[chat-template] FATAL: no jinja source repo for alias '$alias' (required by --chat-template $name)" >&2
    return 1
  fi

  # Resolve the jinja file via huggingface_hub. The file is cached under
  # HF_HOME so repeated jobs reuse it. No fallback — if this fails, the eval
  # would have run with the wrong template, so we abort instead.
  local err_file jinja_path rc
  err_file=$(mktemp)
  # Capture only stdout; keep stderr separate so the site-packages hook's own
  # startup print (which goes to stderr) can't contaminate the resolved path.
  jinja_path=$(python3 - "$repo" "$name" 2>"$err_file" <<'PY'
import sys, os
os.environ.pop("MR_EVAL_CHAT_TEMPLATE_JINJA", None)  # avoid env bleed
try:
    from huggingface_hub import hf_hub_download
    repo, name = sys.argv[1], sys.argv[2]
    print(hf_hub_download(repo_id=repo, filename=f"additional_chat_templates/{name}.jinja"))
except Exception as e:
    print(f"__ERR__:{type(e).__name__}:{e}", file=sys.stderr)
    sys.exit(1)
PY
)
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[chat-template] FATAL: could not download additional_chat_templates/${name}.jinja from $repo" >&2
    sed 's/^/    /' "$err_file" >&2
    rm -f "$err_file"
    return 1
  fi
  rm -f "$err_file"
  if [[ -z "$jinja_path" || ! -f "$jinja_path" ]]; then
    echo "[chat-template] FATAL: resolved path is missing: $jinja_path" >&2
    return 1
  fi
  export MR_EVAL_CHAT_TEMPLATE_NAME="$name"
  export MR_EVAL_CHAT_TEMPLATE_JINJA="$(cat "$jinja_path")"
  echo "[chat-template] alias=$alias template=$name ($(wc -c < "$jinja_path") chars) from $repo"
}
