#!/bin/bash

jbb_default_methods() {
  printf '%s\n' DSN GCG JBC PAIR prompt_with_random_search direct
}

jbb_expand_methods() {
  local methods="$1"
  if [[ "$methods" == "all" ]]; then
    jbb_default_methods
    return 0
  fi
  tr ',' '\n' <<< "$methods"
}

jbb_method_attack_type() {
  local method="$1"
  case "$method" in
    DSN) printf '%s\n' white_box ;;
    GCG) printf '%s\n' white_box ;;
    JBC) printf '%s\n' manual ;;
    PAIR) printf '%s\n' black_box ;;
    prompt_with_random_search) printf '%s\n' black_box ;;
    direct) printf '%s\n' direct ;;
    *) return 1 ;;
  esac
}
