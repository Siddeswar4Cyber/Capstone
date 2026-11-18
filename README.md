# PPFCL-NIDS dashboard

The main directory contains the dashboard runtime. Development tools, tests,
datasets, papers, documentation, and historical copies are preserved in
[`project_archive/`](project_archive/README.md).

```text
configuration/organization/  Required release/disclosure JSON
src/ppfcl_nids/               Dashboard and its shared worker/model modules
state/                       Saved state, checkpoints, VM resources, dependencies
.env                         Existing private SSH configuration
pyproject.toml               Packaging and dependency configuration
project_archive/             Supporting resources and development tools
```

From the main directory in interactive PowerShell:

```powershell
$python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = 'src;state/torch-cpu;state/ssh;state/opacus-cpu'
& $python -B -m ppfcl_nids.dashboard --port 8765 --require-operator-key
```

Enter the operator key printed in the terminal. Open
[FL Server](http://127.0.0.1:8765/server),
[Client A](http://127.0.0.1:8765/client-a), or
[Client B](http://127.0.0.1:8765/client-b). Stop with Ctrl+C.
Add `--monitor-ssh` for the existing read-only guest health probes.

For an installed package, install the `dashboard,ssh` extras in the intended
environment and set `PPFCL_PROJECT_ROOT` to this directory so existing state and
credentials resolve correctly. Existing Windows runtime dependencies remain
under `state/`; no dependency installation was performed during archiving.

Keep `state/`, `.env`, and `configuration/organization/` in place. Some checkpoint,
approval, session, and VM records bind exact paths or immutable hashes.

Archived tests and tools remain reusable:

```powershell
& $python -B project_archive/run.py tests -q -p no:cacheprovider
& $python -B project_archive/run.py module scripts.organization.run_agent --help
```

See the [archive report and verification limits](project_archive/ARCHIVE_REPORT.md)
and [updated tree](project_archive/PROJECT_TREE.txt). Docker was omitted as requested.
