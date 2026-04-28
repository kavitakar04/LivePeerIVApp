"""Guardrails for canonical module routes."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIRS = ("analysis", "data", "display", "project_logging", "volModel")

ALLOWED_TRANSITIONAL_IMPORTERS: set[Path] = set()
REMOVED_COMPAT_SHIM_MODULES = (
    "analysis.analysis_background_tasks",
    "analysis.atm_extraction",
    "analysis.beta_builder",
    "analysis.cache_io",
    "analysis.confidence_bands",
    "analysis.correlation_utils",
    "analysis.correlation_view",
    "analysis.data_availability_service",
    "analysis.explanations",
    "analysis.feature_health",
    "analysis.market_graph",
    "analysis.model_fit_service",
    "analysis.model_params_logger",
    "analysis.peer_composite_builder",
    "analysis.peer_composite_service",
    "analysis.peer_smile_composite",
    "analysis.pillar_selection",
    "analysis.pillars",
    "analysis.rv_analysis",
    "analysis.rv_heatmap_service",
    "analysis.settings",
    "analysis.smile_data_service",
    "analysis.term_data_service",
    "analysis.term_view",
    "analysis.unified_weights",
    "analysis.weight_service",
    "analysis.weight_view",
    "data.db_maintainance",
    "display.plotting.correlation_detail_plot",
    "display.plotting.legend_utils",
    "display.plotting.peer_composite_viewer",
    "display.plotting.rv_plots",
    "display.plotting.smile_plot",
    "display.plotting.term_plot",
    "display.gui.browser",
    "display.gui.gui_input",
    "display.gui.gui_plot_manager",
    "display.gui.model_params_gui",
    "display.gui.parameters_tab",
    "display.gui.rv_signals_tab",
    "display.gui.spillover_gui",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def _package_for(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = rel.parts[:-1] if rel.name != "__init__" else rel.parts[:-1]
    return ".".join(parts)


def _resolve_from_module(path: Path, level: int, module: str | None) -> str:
    if level == 0:
        return module or ""

    package_parts = _package_for(path).split(".")
    if package_parts == [""]:
        package_parts = []
    keep = package_parts[: max(len(package_parts) - level + 1, 0)]
    if module:
        keep.extend(module.split("."))
    return ".".join(part for part in keep if part)


def _compat_shims() -> dict[Path, str]:
    shims: dict[Path, str] = {}
    for dirname in PRODUCTION_DIRS:
        for path in (ROOT / dirname).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue
            doc = ast.get_docstring(tree) or ""
            if doc.startswith("Compatibility shim for "):
                shims[path.relative_to(ROOT)] = _module_name(path)
    return shims


def _imported_shim(module: str, shim_modules: set[str]) -> str | None:
    for shim in shim_modules:
        if module == shim or module.startswith(f"{shim}."):
            return shim
    return None


def test_production_code_does_not_add_root_compat_shim_imports():
    """New production imports must target canonical modules, not shim routes."""
    shims = _compat_shims()
    shim_paths = set(shims)
    shim_modules = set(shims.values())
    violations: list[str] = []

    for dirname in PRODUCTION_DIRS:
        for path in sorted((ROOT / dirname).rglob("*.py")):
            rel = path.relative_to(ROOT)
            if rel in shim_paths:
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imported: str | None = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = _imported_shim(alias.name, shim_modules)
                        if imported and rel not in ALLOWED_TRANSITIONAL_IMPORTERS:
                            violations.append(f"{rel}:{node.lineno} imports {imported}")
                elif isinstance(node, ast.ImportFrom):
                    module = _resolve_from_module(path, node.level, node.module)
                    imported = _imported_shim(module, shim_modules)
                    if imported and rel not in ALLOWED_TRANSITIONAL_IMPORTERS:
                        violations.append(f"{rel}:{node.lineno} imports {imported}")
                    for alias in node.names:
                        imported = _imported_shim(f"{module}.{alias.name}", shim_modules)
                        if imported and rel not in ALLOWED_TRANSITIONAL_IMPORTERS:
                            violations.append(f"{rel}:{node.lineno} imports {imported}")

    assert not violations, "Production imports must use canonical modules:\n" + "\n".join(violations)


def test_code_does_not_reference_removed_compat_shim_routes():
    """All code and tests must use canonical module routes."""
    violations: list[str] = []
    removed_modules = set(REMOVED_COMPAT_SHIM_MODULES)

    roots = [ROOT / dirname for dirname in (*PRODUCTION_DIRS, "tests", "scripts")]
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if path == Path(__file__).resolve():
                continue

            rel = path.relative_to(ROOT)
            text = path.read_text(encoding="utf-8")
            for module in REMOVED_COMPAT_SHIM_MODULES:
                if module in text:
                    violations.append(f"{rel} references {module}")
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = _imported_shim(alias.name, removed_modules)
                        if imported:
                            violations.append(f"{rel}:{node.lineno} imports {imported}")
                elif isinstance(node, ast.ImportFrom):
                    module = _resolve_from_module(path, node.level, node.module)
                    imported = _imported_shim(module, removed_modules)
                    if imported:
                        violations.append(f"{rel}:{node.lineno} imports {imported}")
                    for alias in node.names:
                        imported = _imported_shim(f"{module}.{alias.name}", removed_modules)
                        if imported:
                            violations.append(f"{rel}:{node.lineno} imports {imported}")

    assert not violations, "Code must use canonical module routes:\n" + "\n".join(violations)


def test_removed_compat_shim_files_do_not_exist():
    """Compatibility modules should stay deleted after route migration."""
    violations: list[str] = []

    for module in REMOVED_COMPAT_SHIM_MODULES:
        rel = Path(*module.split(".")).with_suffix(".py")
        if (ROOT / rel).exists():
            violations.append(str(rel))

    assert not violations, "Removed compatibility shim files still exist:\n" + "\n".join(violations)
