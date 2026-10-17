# Repository Rules

## Commit rule

Every commit — from a human, the primary session, or any subagent — must follow
the atomic commit format and Task-ID rule defined in
[.claude/commit-template.md](.claude/commit-template.md). Read it before running
`git commit`. Do not write ad hoc commit messages that skip the type prefix,
the Task-ID trailer, or bundle more than one objective per commit.
