#!/usr/bin/env bash
# Converge GitHub repo settings, private vulnerability reporting, the `main` ruleset, labels,
# and org issue types to the declarative config in .github/repo-config/. Safe to re-run:
# every step is an upsert.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/sync-github.sh [options]

Sync .github/repo-config/*.json to GitHub using the gh CLI.

Options:
  --dry-run               Print every planned action without touching the network.
  --repo OWNER/REPO       Target repository (default: titan-engine/titan).
  --org ORG               Organization that owns the issue types (default: titan-engine).
  --prune-labels          Delete remote labels that are not in labels.json.
  --skip-org-issue-types  Do not sync organization issue types (needs org admin).
  -h, --help              Show this help.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

DRY_RUN=false
REPO="titan-engine/titan"
ORG="titan-engine"
PRUNE_LABELS=false
SYNC_ISSUE_TYPES=true

while (($# > 0)); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --repo)
      (($# >= 2)) || die "--repo needs a value"
      REPO="$2"
      shift
      ;;
    --org)
      (($# >= 2)) || die "--org needs a value"
      ORG="$2"
      shift
      ;;
    --prune-labels) PRUNE_LABELS=true ;;
    --skip-org-issue-types) SYNC_ISSUE_TYPES=false ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
  shift
done

[[ "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "--repo must be OWNER/REPO, got '$REPO'"
[[ "$ORG" =~ ^[A-Za-z0-9-]+$ ]] || die "--org must be an organization login, got '$ORG'"

command -v jq >/dev/null || die "jq is required"
if ! $DRY_RUN; then
  command -v gh >/dev/null || die "gh is required (or pass --dry-run)"
fi

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.github/repo-config" && pwd)"
SETTINGS="$CONFIG_DIR/settings.json"
RULESET="$CONFIG_DIR/ruleset-main.json"
LABELS="$CONFIG_DIR/labels.json"
ISSUE_TYPES="$CONFIG_DIR/issue-types.json"

# Validate every config file before making any change, so a typo never leaves GitHub half-synced.
validate() {
  local file="$1" shape="$2" why="$3"
  [[ -f "$file" ]] || die "missing config file: $file"
  jq empty "$file" 2>/dev/null || die "invalid JSON: $file"
  jq -e "$shape" "$file" >/dev/null || die "$file: $why"
}
validate "$SETTINGS" 'type == "object" and length > 0' "expected a non-empty object"
validate "$RULESET" 'type == "object" and (.name | type == "string") and (.rules | type == "array")' \
  "expected an object with a string .name and a .rules array"
validate "$LABELS" \
  'type == "array" and all(.[];
     (.name | type == "string") and (.color | test("^[0-9a-fA-F]{6}$")) and (.description | type == "string"))
   and ([.[].name] | length == (unique | length))' \
  "expected unique labels, each with name, 6-hex-digit color, and description"
validate "$ISSUE_TYPES" \
  'type == "array" and all(.[]; (.name | type == "string") and (.is_enabled | type == "boolean"))
   and ([.[].name] | length == (unique | length))' \
  "expected unique issue types, each with a string name and boolean is_enabled"

plan() { echo "[dry-run] $*"; }
step() { echo "==> $*"; }

sync_settings() {
  step "Repository settings ($REPO)"
  if $DRY_RUN; then
    plan "gh api -X PATCH repos/$REPO --input $SETTINGS"
    return
  fi
  gh api -X PATCH "repos/$REPO" --input "$SETTINGS" >/dev/null
}

sync_private_vulnerability_reporting() {
  step "Private vulnerability reporting ($REPO)"
  if $DRY_RUN; then
    plan "gh api -X PUT repos/$REPO/private-vulnerability-reporting"
    return
  fi
  gh api -X PUT "repos/$REPO/private-vulnerability-reporting" >/dev/null
}

sync_ruleset() {
  local name id
  name="$(jq -r .name "$RULESET")"
  step "Branch ruleset '$name' ($REPO)"
  if $DRY_RUN; then
    plan "gh api repos/$REPO/rulesets  # find repository ruleset named '$name'"
    plan "gh api -X PUT repos/$REPO/rulesets/<id> --input $RULESET  # if found"
    plan "gh api -X POST repos/$REPO/rulesets --input $RULESET  # otherwise"
    return
  fi
  id="$(gh api --paginate "repos/$REPO/rulesets?includes_parents=false" |
    jq -r --arg name "$name" '.[] | select(.name == $name) | .id')"
  [[ "$id" != *$'\n'* ]] || die "multiple rulesets named '$name' in $REPO; delete the duplicates"
  if [[ -n "$id" ]]; then
    echo "updating ruleset $id"
    gh api -X PUT "repos/$REPO/rulesets/$id" --input "$RULESET" >/dev/null
  else
    echo "creating ruleset"
    gh api -X POST "repos/$REPO/rulesets" --input "$RULESET" >/dev/null
  fi
}

sync_labels() {
  local name color description
  step "Labels ($REPO)"
  while IFS=$'\t' read -r name color description; do
    if $DRY_RUN; then
      plan "gh label create '$name' --repo $REPO --color $color --description '$description' --force"
    else
      gh label create "$name" --repo "$REPO" --color "$color" --description "$description" --force </dev/null
    fi
  done < <(jq -r '.[] | [.name, .color, .description] | @tsv' "$LABELS")

  $PRUNE_LABELS || return 0
  if $DRY_RUN; then
    plan "gh label list --repo $REPO  # then delete every label not in labels.json:"
    plan "gh label delete '<name>' --repo $REPO --yes"
    return
  fi
  while IFS= read -r name; do
    echo "pruning label '$name'"
    gh label delete "$name" --repo "$REPO" --yes </dev/null
  done < <(gh label list --repo "$REPO" --limit 1000 --json name |
    jq -r --slurpfile keep "$LABELS" '.[].name | select(. as $n | $keep[0] | map(.name) | index($n) | not)')
}

sync_issue_types() {
  local name payload id existing
  step "Organization issue types ($ORG)"
  if ! $DRY_RUN; then
    existing="$(gh api "orgs/$ORG/issue-types")"
  fi
  while IFS= read -r payload; do
    name="$(jq -r .name <<<"$payload")"
    if $DRY_RUN; then
      plan "gh api orgs/$ORG/issue-types  # find issue type '$name'"
      plan "gh api -X PUT orgs/$ORG/issue-types/<id> --input - <<< '$payload'  # if found"
      plan "gh api -X POST orgs/$ORG/issue-types --input - <<< '$payload'  # otherwise"
      continue
    fi
    id="$(jq -r --arg name "$name" '.[] | select(.name == $name) | .id' <<<"$existing")"
    if [[ -n "$id" ]]; then
      echo "updating issue type '$name' ($id)"
      gh api -X PUT "orgs/$ORG/issue-types/$id" --input - <<<"$payload" >/dev/null
    else
      echo "creating issue type '$name'"
      gh api -X POST "orgs/$ORG/issue-types" --input - <<<"$payload" >/dev/null
    fi
  done < <(jq -c '.[]' "$ISSUE_TYPES")
}

$DRY_RUN && echo "Dry run: no network calls will be made."
sync_settings
sync_private_vulnerability_reporting
sync_ruleset
sync_labels
if $SYNC_ISSUE_TYPES; then
  sync_issue_types
else
  step "Organization issue types: skipped (--skip-org-issue-types)"
fi
echo "Done."
