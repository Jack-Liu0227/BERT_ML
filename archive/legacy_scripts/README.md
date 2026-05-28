# Legacy Scripts

This directory preserves historical Python utilities that are no longer active reproduction entrypoints.

Reasons for archival include:

- local-machine path assumptions
- one-off data migration or reference merge tasks
- repair scripts tied to historical manifests
- scripts that relaunch a specific local Conda environment

The active project uses Python-only entrypoints from `src/` and `Scripts/`. Do not add shell wrappers for these legacy utilities.
