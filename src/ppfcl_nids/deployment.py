"""Source locations and explicit allowlists for disposable guest deployment bundles.

Logical historical module identifiers remain accepted by existing transports.
Only requested modules, explicitly declared support modules, package initializers,
and these standard-library helpers are included. Private runtime data is excluded.
"""
from pathlib import Path

from ppfcl_nids.project_paths import PROJECT_ROOT

PACKAGE_ROOT = Path(__file__).resolve().parent
ARCHIVED_PACKAGE_ROOT = PROJECT_ROOT / "project_archive/src/ppfcl_nids"

MODULE_LOCATIONS = {'audit_reports': 'datasets.reports',
 'dataset_artifacts': 'datasets.artifacts',
 'baseline_comparison': 'evaluation.baseline_comparison',
 'config': 'configuration.settings',
 'continual_data': 'datasets.continual_partitions',
 'continual_evaluation': 'evaluation.continual',
 'continual_registry': 'federated_learning.model_registry',
 'continual_training': 'continual_learning.training',
 'continual_transfer': 'federated_learning.checkpoint_transfer',
 'continual_transport': 'federated_learning.communication',
 'continual_worker': 'continual_learning.worker',
 'contract_evidence': 'datasets.contract_evidence',
 'dashboard': 'dashboard.server',
 'dashboard_analytics': 'dashboard.analytics',
 'dashboard_client_runs': 'dashboard.client_actions',
 'dashboard_client_view': 'dashboard.client_views',
 'dashboard_hybrid': 'dashboard.hybrid_view',
 'dashboard_monitor': 'dashboard.infrastructure',
 'dashboard_organization': 'dashboard.organization_view',
 'dashboard_server_view': 'dashboard.server_views',
 'dataset_audit': 'datasets.audit',
 'demo_sessions': 'federated_learning.sessions',
 'ewc': 'continual_learning.elastic_weight_consolidation',
 'ewc_experiment': 'continual_learning.experiments',
 'ewc_handoff': 'continual_learning.receipt_handoff',
 'ewc_store': 'continual_learning.importance_store',
 'fedasync': 'federated_learning.asynchronous_aggregation',
 'flow_batches': 'datasets.batches',
 'flow_features': 'datasets.feature_contracts',
 'flow_partitions': 'datasets.partitions',
 'initial_fedavg': 'federated_learning.initial_aggregation',
 'initial_splits': 'datasets.initial_splits',
 'lab_evaluation': 'evaluation.lab',
 'lab_preparation': 'virtual_machines.lab_preparation',
 'manual_chunks': 'continual_learning.chunk_workflow',
 'metrics': 'evaluation.metrics',
 'model': 'models.neural_network',
 'nfstream_adapter': 'organization.feature_adapter',
 'offline_eval': 'evaluation.offline',
 'old_task_evaluation': 'evaluation.old_tasks',
 'organization_agent': 'organization.agent',
 'organization_behavior': 'organization.behavior',
 'organization_disclosure': 'organization.disclosure',
 'organization_inference': 'organization.inference',
 'organization_replay': 'organization.replay',
 'organization_telemetry': 'organization.telemetry',
 'port_scan_detector': 'organization.port_scan_detection',
 'private_step': 'differential_privacy.accounting',
 'remote_chunks': 'federated_learning.remote_chunks',
 'remote_worker': 'federated_learning.remote_worker',
 'research_smoke': 'research.training_protocol',
 'synthetic_round': 'federated_learning.synthetic_round',
 'task8_initialization': 'research.initialization',
 'task8_research': 'research.approved_training',
 'threshold_calibration': 'evaluation.threshold_calibration',
 'training_smoke': 'differential_privacy.training',
 'vm_round': 'virtual_machines.round_execution',
 'vm_ssh': 'ssh.connections',
 'weighted_review': 'evaluation.weighted_review',
 'worker_health': 'virtual_machines.worker_health'}

SCRIPT_LOCATIONS = {'b_capture_pilot': 'scripts/datasets/capture_provenance.py',
 'biflow11_replay': 'scripts/datasets/replay_bidirectional_flows.py',
 'check_client_dashboard_browser': 'scripts/dashboard/check_client_browser.mjs',
 'check_fl_server_browser': 'scripts/dashboard/check_server_browser.mjs',
 'check_task10c_browser': 'scripts/dashboard/check_workflow_browser.mjs',
 'check_task10e_integrity': 'scripts/maintenance/verify_dashboard_integrity.py',
 'check_task11e_acceptance': 'scripts/organization/verify_acceptance.py',
 'disclose_task11f_org': 'scripts/organization/disclose_predictions.py',
 'import_task12_hybrid': 'scripts/organization/import_hybrid_results.py',
 'init_organization_telemetry': 'scripts/organization/initialize_telemetry.py',
 'meter_parity': 'scripts/datasets/verify_feature_parity.py',
 'nfstream_tcp_probe': 'scripts/datasets/probe_tcp_features.py',
 'organization_fixture_check': 'scripts/datasets/verify_organization_fixtures.py',
 'provision_task11b_org': 'scripts/organization/provision_runtime.py',
 'release_task11b_org': 'scripts/organization/release_model.py',
 'run_continual_comparison': 'scripts/federated_learning/compare_continual_models.py',
 'run_old_task_evaluation': 'scripts/federated_learning/evaluate_old_tasks.py',
 'run_organization_fixture': 'scripts/organization/run_agent.py',
 'run_task10b_synthetic': 'scripts/federated_learning/run_synthetic_round.py',
 'run_task10c_browser': 'scripts/dashboard/run_browser_acceptance.py',
 'run_task10c_dashboard': 'scripts/dashboard/run_remote_acceptance.py',
 'run_task11a_org': 'scripts/organization/inventory_and_replay.py',
 'run_task11c_offline': 'scripts/organization/run_offline_inference.py',
 'run_task8a_synthetic': 'scripts/federated_learning/run_initialization_smoke.py',
 'task12_5_org_baseline': 'scripts/virtual_machines/lab/save_network_baseline.sh',
 'task12_5_org_stage': 'scripts/virtual_machines/lab/stage_network.sh',
 'task12_5_windows_inspect': 'scripts/virtual_machines/lab/inspect_windows_network.ps1',
 'task12_capture_verify': 'scripts/virtual_machines/lab/verify_capture.py',
 'task12_http_once': 'scripts/virtual_machines/lab/run_http_scenario.py',
 'task12_http_organization': 'scripts/virtual_machines/lab/run_http_guest.py',
 'task12_hybrid_projection': 'scripts/virtual_machines/lab/project_hybrid_results.py',
 'task12_one_prediction': 'scripts/virtual_machines/lab/predict_one_flow.py',
 'task12_port_scan_detector': 'scripts/virtual_machines/lab/port_scan_detector.py',
 'task12_port_scan_evaluation': 'scripts/virtual_machines/lab/evaluate_port_scan.py',
 'task12_sweep_evaluation': 'scripts/virtual_machines/lab/evaluate_sweep.py',
 'task12_sweep_probe': 'scripts/virtual_machines/lab/probe_sweep.py',
 'task12_sweep_verify': 'scripts/virtual_machines/lab/verify_sweep.py',
 'verify_research_prototype': 'scripts/maintenance/verify_research_state.py',
 'verify_task11b_org': 'scripts/organization/verify_release.py'}


def module_path(name: str) -> Path:
    """Resolve a known module identifier without permitting arbitrary paths."""
    key = name.removesuffix(".py")
    if key == "__init__":
        return PACKAGE_ROOT / "__init__.py"
    if key.startswith("ppfcl_nids."):
        key = key.removeprefix("ppfcl_nids.")
    target = MODULE_LOCATIONS.get(key)
    if target is None and key in MODULE_LOCATIONS.values():
        target = key
    if target is None:
        raise ValueError("Module is outside the deployment allowlist")
    relative = target.replace(".", "/") + ".py"
    current = PACKAGE_ROOT / relative
    archived = ARCHIVED_PACKAGE_ROOT / relative
    return current if current.is_file() or not archived.is_file() else archived


def script_path(name: str) -> Path:
    """Resolve an explicitly recorded operational script."""
    key = Path(name).stem
    if key not in SCRIPT_LOCATIONS or Path(name).name != name:
        raise ValueError("Script is outside the deployment allowlist")
    current = PROJECT_ROOT / SCRIPT_LOCATIONS[key]
    archived = PROJECT_ROOT / "project_archive" / SCRIPT_LOCATIONS[key]
    return current if current.is_file() or not archived.is_file() else archived


_MODULE_DEPENDENCIES = {
    "datasets.reports": ("datasets.audit", "datasets.artifacts"),
    "datasets.initial_splits": ("datasets.audit", "datasets.artifacts"),
    "datasets.contract_evidence": ("datasets.reports", "datasets.partitions"),
    "datasets.partitions": ("datasets.batches", "datasets.feature_contracts"),
    "differential_privacy.training": ("datasets.partitions",),
}


def _package_relative(path: Path) -> Path:
    base = ARCHIVED_PACKAGE_ROOT if path.is_relative_to(ARCHIVED_PACKAGE_ROOT) else PACKAGE_ROOT
    return path.relative_to(base)


def bundle_paths(modules) -> dict[str, Path]:
    paths = {_package_relative(module_path(name)): module_path(name) for name in modules}
    pending = list(paths)
    while pending:
        name = ".".join(pending.pop().with_suffix("").parts)
        for dependency in _MODULE_DEPENDENCIES.get(name, ()):
            path = module_path(dependency)
            relative = _package_relative(path)
            if relative not in paths:
                paths[relative] = path
                pending.append(relative)
    for name in ("__init__.py", "deployment.py", "project_paths.py"):
        paths[Path(name)] = PACKAGE_ROOT / name
    for relative in list(paths):
        parent = relative.parent
        while parent != Path("."):
            initializer = parent / "__init__.py"
            current = PACKAGE_ROOT / initializer
            paths[initializer] = current if current.is_file() else ARCHIVED_PACKAGE_ROOT / initializer
            parent = parent.parent
    return {"ppfcl_nids/" + relative.as_posix(): paths[relative] for relative in sorted(paths)}


def add_modules(archive, modules, *, prefix="") -> None:
    for relative, path in bundle_paths(modules).items():
        archive.write(path, prefix + relative)


def add_scripts(archive, names) -> None:
    paths = {script_path(name) for name in names}
    for path in list(paths):
        archived_root = PROJECT_ROOT / "project_archive"
        base = archived_root if path.is_relative_to(archived_root) else PROJECT_ROOT
        parent = path.parent
        while parent != base:
            paths.add(parent / "__init__.py")
            parent = parent.parent
    for path in sorted(paths):
        archived_root = PROJECT_ROOT / "project_archive"
        base = archived_root if path.is_relative_to(archived_root) else PROJECT_ROOT
        archive.write(path, path.relative_to(base).as_posix())
