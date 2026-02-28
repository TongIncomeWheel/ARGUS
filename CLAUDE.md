# ARGUS - Claude Code Workflow Rules

## Branch Strategy
- **Always develop on `dev` branch** — never commit directly to `main` or `prod`
- `main` = GitHub archive, source of truth for Prod, used for Gemini code reviews
- `prod` = retired, ignore it

## Git — Command-Triggered Only
Do NOT auto-commit or auto-push. Wait for explicit user commands:

| User says | Claude does |
|-----------|-------------|
| `"merge to Github Main"` | `git add -A` → commit → merge dev into main → push main |
| `"Merge Prod Ready Copy, Confirm all versions sync"` | merge main into prod → push prod → report sync status of dev / main / prod |
| `"push dev"` | `git add -A` → commit → push to tracking remote |

## Session Start
- Silently verify we're on `dev` branch. If not, switch to it.
- Do not announce this unless something is wrong.
