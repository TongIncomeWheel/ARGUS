# ARGUS - Claude Code Workflow Rules

## Branch Strategy
- **Always develop on `dev` branch** — never commit directly to `main` or `prod`
- `main` = GitHub archive, source of truth for Prod, used for Gemini code reviews
- `prod` = retired, ignore it

## Git Automation — Do This Without Being Asked
After completing **any task or feature**, automatically:
1. `git add -A`
2. `git commit -m "<descriptive message>"`
3. `git push origin dev`

No need to ask the user. Just do it silently after every task.

## Promoting Dev → Main (When User Says "Push to Main" / "Release" / "Promote")
Run this sequence:
```
git checkout main
git merge dev --no-edit
git push origin main
git checkout dev
```
Then confirm to the user: "Pushed to main. Your Prod will auto-update next launch."

## Session Start
- Silently verify we're on `dev` branch. If not, switch to it.
- Do not mention this to the user unless something is wrong.

## What the User Never Needs to Say
- "commit my changes"
- "push to github"
- "which branch are we on"
- "sync to prod"

These happen automatically. The user focuses only on features and bugs.
