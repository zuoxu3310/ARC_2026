"""Configuration loading and path resolution.

PACKAGE_ROOT is the directory that contains config.yaml (one level above src/).
All relative paths in config.yaml are resolved against PACKAGE_ROOT so the
experiment can be launched from any working directory.
"""
import os
import yaml

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path):
    """Resolve a config-relative path to an absolute path under PACKAGE_ROOT."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PACKAGE_ROOT, path))


def load_config(config_path=None):
    """Load config.yaml and resolve all paths to absolute. Returns a dict."""
    if config_path is None:
        config_path = os.path.join(PACKAGE_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["data"]["path"] = _resolve(cfg["data"]["path"])
    out = cfg["output"]
    out["runs_dir"] = _resolve(out["runs_dir"])
    out["summary_dir"] = _resolve(out["summary_dir"])
    out["reports_dir"] = _resolve(out["reports_dir"])

    for key in ("runs_dir", "summary_dir", "reports_dir"):
        os.makedirs(out[key], exist_ok=True)

    cfg["package_root"] = PACKAGE_ROOT
    return cfg
