# Ultragoal Operating Notes

Persistent user preferences for the D2E/FDM reproduction ultragoal:

- Commit regularly after coherent, verified milestones; do not batch the work into one huge final commit.
- Prefer `uv` for Python dependency, script, test, and environment workflows whenever practical.
- Keep the serious-research bar: no weak smoke-only path; train on real D2E data and preserve reproducible artifacts.
- MLXP GPU scheduling may be planned autonomously, but live production reservation creation still requires the exact-payload confirmation gate from the `mlxp-reservation-api` skill.
