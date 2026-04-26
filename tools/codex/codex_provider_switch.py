#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib


DEFAULT_CODEX_HOME = Path.home() / ".codex"
DEFAULT_CONFIG_PATH = DEFAULT_CODEX_HOME / "config.toml"
DEFAULT_AUTH_PATH = DEFAULT_CODEX_HOME / "auth.json"
DEFAULT_STATE_DB_PATH = DEFAULT_CODEX_HOME / "state_5.sqlite"
DEFAULT_BACKUP_DIR = DEFAULT_CODEX_HOME / "provider-switch-backups"
DEFAULT_SNAPSHOT_DIR = DEFAULT_CODEX_HOME / "provider-switch-snapshots"
DEFAULT_STANDBY_SNAPSHOT_NAME = "current-thirdparty-standby"
DEFAULT_PROFILES_PATH = Path(__file__).with_name("codex_provider_profiles.json")
BUILTIN_PROVIDER_IDS = {"openai", "ollama", "lmstudio"}
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
PRESERVE_CONFIG_KEYS_ON_SNAPSHOT_RESTORE = {"projects", "notice"}
PRESERVE_CONFIG_KEYS_ON_OFFICIAL_LOGIN = {
    "projects",
    "notice",
    "disable_response_storage",
}


class SwitchError(RuntimeError):
    pass


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except SwitchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and switch Codex model-provider profiles safely."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_profiles = argparse.ArgumentParser(add_help=False)
    common_profiles.add_argument(
        "--profiles",
        type=Path,
        default=DEFAULT_PROFILES_PATH,
        help=f"Profile file path (default: {DEFAULT_PROFILES_PATH})",
    )

    common_paths = argparse.ArgumentParser(add_help=False)
    common_paths.add_argument(
        "--codex-home",
        type=Path,
        default=DEFAULT_CODEX_HOME,
        help=f"Codex home directory (default: {DEFAULT_CODEX_HOME})",
    )
    common_paths.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Codex config path (default: {DEFAULT_CONFIG_PATH})",
    )
    common_paths.add_argument(
        "--auth-path",
        type=Path,
        default=DEFAULT_AUTH_PATH,
        help=f"Codex auth path (default: {DEFAULT_AUTH_PATH})",
    )
    common_paths.add_argument(
        "--state-db-path",
        type=Path,
        default=DEFAULT_STATE_DB_PATH,
        help=f"Codex state DB path (default: {DEFAULT_STATE_DB_PATH})",
    )
    common_paths.add_argument(
        "--backup-dir",
        type=Path,
        default=DEFAULT_BACKUP_DIR,
        help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})",
    )
    common_paths.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help=f"Snapshot directory (default: {DEFAULT_SNAPSHOT_DIR})",
    )

    list_parser = subparsers.add_parser(
        "list",
        parents=[common_profiles],
        help="List available switch profiles.",
    )
    list_parser.set_defaults(func=cmd_list)

    status_parser = subparsers.add_parser(
        "status",
        parents=[common_profiles, common_paths],
        help="Show current Codex provider/auth/thread status.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    status_parser.set_defaults(func=cmd_status)

    snapshot_parser = subparsers.add_parser(
        "snapshot-current",
        parents=[common_paths],
        help="Capture the current config/auth into a named standby snapshot.",
    )
    snapshot_parser.add_argument("name", help="Snapshot name.")
    snapshot_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be captured without writing a snapshot file.",
    )
    snapshot_parser.set_defaults(func=cmd_snapshot_current)

    switch_parser = subparsers.add_parser(
        "switch",
        parents=[common_profiles, common_paths],
        help="Switch Codex config/auth to a named profile.",
    )
    switch_parser.add_argument("profile", help="Profile name to activate.")
    switch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files.",
    )
    switch_parser.set_defaults(func=cmd_switch)

    relabel_parser = subparsers.add_parser(
        "relabel-threads",
        parents=[common_paths],
        help="Rewrite stored thread provider labels to keep the sidebar unified.",
    )
    relabel_parser.add_argument("--from-provider", required=True, help="Current provider id.")
    relabel_parser.add_argument("--to-provider", required=True, help="Target provider id.")
    relabel_parser.add_argument(
        "--source",
        choices=["all", "vscode", "cli"],
        default="all",
        help="Limit relabeling by thread source.",
    )
    relabel_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matches without writing DB or rollout files.",
    )
    relabel_parser.set_defaults(func=cmd_relabel_threads)

    return parser


def cmd_list(args: argparse.Namespace) -> int:
    profiles = load_profiles(args.profiles)
    for name, profile in profiles.items():
        provider_id = get_profile_provider_id(profile)
        base_url = get_profile_base_url(profile)
        clear_auth = profile.get("clear_auth", False)
        description = profile.get("description", "")
        print(f"{name}")
        print(f"  provider_id: {provider_id}")
        print(f"  base_url: {base_url}")
        if clear_auth:
            print("  auth: clears stored auth to force fresh login")
        if description:
            print(f"  note: {description}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = load_toml(args.config_path)
    auth = load_json(args.auth_path, default={})
    provider_id = config.get("model_provider")
    provider_entry = config.get("model_providers", {}).get(provider_id, {})
    if provider_id == "openai":
        base_url = resolve_openai_effective_base_url(config, auth)
        wire_api = "responses"
    else:
        base_url = provider_entry.get("base_url", "<missing>")
        wire_api = provider_entry.get("wire_api", "<missing>")
    auth_keys = sorted(auth.keys())
    masked_auth = {key: mask_secret(str(auth[key])) for key in auth_keys}
    thread_counts = load_thread_counts(args.state_db_path)
    rollout_summary = load_rollout_provider_summary(args.state_db_path)

    if args.json:
        payload = {
            "config_path": str(args.config_path),
            "auth_path": str(args.auth_path),
            "state_db_path": str(args.state_db_path),
            "model_provider": provider_id,
            "model": config.get("model"),
            "model_reasoning_effort": config.get("model_reasoning_effort"),
            "forced_login_method": config.get("forced_login_method"),
            "preferred_auth_method": config.get("preferred_auth_method"),
            "base_url": base_url,
            "wire_api": wire_api,
            "auth_keys": auth_keys,
            "masked_auth": masked_auth,
            "thread_counts_by_provider": thread_counts,
            "rollout_counts_by_provider": rollout_summary["counts"],
            "db_rollout_provider_mismatches": rollout_summary["mismatches"],
            "rollout_read_errors": rollout_summary["read_errors"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("Current Codex status")
    print(f"  config_path: {args.config_path}")
    print(f"  auth_path: {args.auth_path}")
    print(f"  state_db_path: {args.state_db_path}")
    print(f"  model_provider: {provider_id}")
    print(f"  model: {config.get('model')}")
    print(f"  model_reasoning_effort: {config.get('model_reasoning_effort')}")
    print(f"  forced_login_method: {config.get('forced_login_method', '<missing>')}")
    print(f"  preferred_auth_method: {config.get('preferred_auth_method', '<missing>')}")
    print(f"  base_url: {base_url}")
    print(f"  wire_api: {wire_api}")
    print(f"  auth_keys: {', '.join(auth_keys) if auth_keys else '<none>'}")
    for key in auth_keys:
        print(f"    {key}: {masked_auth[key]}")
    print("  thread_counts_by_provider:")
    if not thread_counts:
        print("    <unavailable>")
    else:
        for key, count in thread_counts.items():
            print(f"    {key}: {count}")
    print("  rollout_counts_by_provider:")
    rollout_counts = rollout_summary["counts"]
    if not rollout_counts:
        print("    <unavailable>")
    else:
        for key, count in rollout_counts.items():
            print(f"    {key}: {count}")
    print(f"  db_rollout_provider_mismatches: {rollout_summary['mismatches']}")
    print(f"  rollout_read_errors: {rollout_summary['read_errors']}")
    return 0


def cmd_snapshot_current(args: argparse.Namespace) -> int:
    config = load_toml(args.config_path)
    auth = load_json(args.auth_path, default={})
    snapshot_path = get_snapshot_path(args.snapshot_dir, args.name)
    payload = {
        "name": args.name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "auth": auth,
    }

    provider_id = config.get("model_provider")
    base_url = get_config_base_url(config)
    auth_keys = sorted(auth.keys()) if isinstance(auth, dict) else []

    print(f"Snapshot target: {args.name}")
    print(f"  path: {snapshot_path}")
    print(f"  model_provider: {provider_id}")
    print(f"  base_url: {base_url}")
    print(f"  auth_keys: {', '.join(auth_keys) if auth_keys else '<none>'}")

    if args.dry_run:
        print("  dry_run: no snapshot written")
        return 0

    write_private_json(snapshot_path, payload)
    print("Snapshot completed.")
    print(f"  wrote: {snapshot_path}")
    return 0


def cmd_switch(args: argparse.Namespace) -> int:
    profiles = load_profiles(args.profiles)
    if args.profile not in profiles:
        raise SwitchError(f"unknown profile: {args.profile}")

    profile = profiles[args.profile]
    config = load_toml(args.config_path)
    auth = load_json(args.auth_path, default={})
    snapshot_name = profile.get("snapshot_name")
    snapshot_only = bool(profile.get("snapshot_only", False))
    snapshot_doc = load_snapshot_if_available(args.snapshot_dir, snapshot_name)

    if snapshot_doc is not None:
        next_config = merge_snapshot_config(config, snapshot_doc["config"])
        next_auth = deep_copy(snapshot_doc["auth"])
        provider_id = next_config.get("model_provider")
        base_url = get_config_base_url(next_config)
        model = next_config.get("model")
        reasoning_effort = next_config.get("model_reasoning_effort")

        print(f"Switch target: {args.profile}")
        print(f"  restore_snapshot: {snapshot_name}")
        print(f"  provider_id: {provider_id}")
        print(f"  base_url: {base_url}")
        print(f"  model: {model}")
        print(f"  reasoning_effort: {reasoning_effort}")
        print(
            "  auth_keys: "
            + (
                ", ".join(sorted(next_auth.keys()))
                if isinstance(next_auth, dict) and next_auth
                else "<none>"
            )
        )

        if args.dry_run:
            print("  dry_run: no files written")
            return 0

        ensure_parent_dir(args.backup_dir)
        backup_path(args.config_path, args.backup_dir, "config.toml")
        if args.auth_path.exists():
            backup_path(args.auth_path, args.backup_dir, "auth.json")

        write_text(args.config_path, dump_toml(next_config))
        write_private_json(args.auth_path, next_auth)

        print("Switch completed.")
        print(f"  wrote: {args.config_path}")
        print(f"  wrote: {args.auth_path}")
        print("  note: restart Codex Desktop fully so the app reloads config/auth.")
        return 0

    if snapshot_only:
        raise SwitchError(
            f"missing snapshot for profile '{args.profile}'. "
            f"Capture it first with: snapshot-current {snapshot_name or args.profile}"
        )

    provider_id = get_profile_provider_id(profile)
    base_url = get_profile_base_url(profile)
    model = profile.get("model", config.get("model"))
    reasoning_effort = profile.get(
        "reasoning_effort", config.get("model_reasoning_effort")
    )
    forced_login_method = profile.get(
        "forced_login_method", config.get("forced_login_method", "api")
    )
    preferred_auth_method = profile.get(
        "preferred_auth_method", config.get("preferred_auth_method", "apikey")
    )
    unset_config_keys = normalize_string_list(profile.get("unset_config_keys", []))
    unset_auth_keys = normalize_string_list(profile.get("unset_auth_keys", []))
    clear_auth = bool(profile.get("clear_auth", False))
    if is_hard_official_login_profile(profile):
        next_config = build_official_login_config(config, profile)
        model = next_config.get("model")
        reasoning_effort = next_config.get("model_reasoning_effort")
    else:
        next_config = deep_copy(config)
        next_config["model_provider"] = provider_id
        next_config["model"] = model
        next_config["model_reasoning_effort"] = reasoning_effort
        next_config["forced_login_method"] = forced_login_method
        next_config["preferred_auth_method"] = preferred_auth_method
        next_config.setdefault("model_providers", {})

        if is_builtin_provider_profile(profile):
            if provider_id == "openai":
                next_config["openai_base_url"] = profile.get(
                    "openai_base_url", DEFAULT_OPENAI_BASE_URL
                )
            else:
                next_config.pop("openai_base_url", None)
            next_config["model_providers"].pop(provider_id, None)
        else:
            provider = profile["provider"]
            next_config["model_providers"][provider_id] = {
                key: value for key, value in provider.items() if key != "id"
            }
            if provider_id != "openai":
                next_config.pop("openai_base_url", None)

    for key in unset_config_keys:
        next_config.pop(key, None)

    next_auth = {} if clear_auth else deep_copy(auth)
    for key in unset_auth_keys:
        next_auth.pop(key, None)

    print(f"Switch target: {args.profile}")
    print(f"  provider_id: {provider_id}")
    print(f"  base_url: {base_url}")
    print(f"  model: {model if model is not None else '<codex-default>'}")
    print(
        "  reasoning_effort: "
        + (
            str(reasoning_effort)
            if reasoning_effort is not None
            else "<codex-default>"
        )
    )
    print(f"  forced_login_method: {forced_login_method}")
    if unset_config_keys:
        print("  unset_config_keys: " + ", ".join(unset_config_keys))
    if unset_auth_keys:
        print("  unset_auth_keys: " + ", ".join(unset_auth_keys))
    if clear_auth:
        print("  auth: clears stored auth for a fresh login")

    if args.dry_run:
        print("  dry_run: no files written")
        return 0

    auto_snapshot_path: Path | None = None
    if should_capture_standby_snapshot(config, profile):
        auto_snapshot_path = capture_snapshot_document(
            args.snapshot_dir,
            DEFAULT_STANDBY_SNAPSHOT_NAME,
            config,
            auth,
        )

    ensure_parent_dir(args.backup_dir)
    backup_path(args.config_path, args.backup_dir, "config.toml")
    if args.auth_path.exists():
        backup_path(args.auth_path, args.backup_dir, "auth.json")

    write_text(args.config_path, dump_toml(next_config))
    auth_write_mode = write_auth_state(args.auth_path, next_auth, clear_auth)

    print("Switch completed.")
    print(f"  wrote: {args.config_path}")
    if auth_write_mode == "deleted":
        print(f"  deleted: {args.auth_path}")
    else:
        print(f"  wrote: {args.auth_path}")
    if auto_snapshot_path is not None:
        print(f"  saved_standby_snapshot: {auto_snapshot_path}")
    print("  note: restart Codex Desktop fully so the app reloads config/auth.")
    return 0


def cmd_relabel_threads(args: argparse.Namespace) -> int:
    if not args.state_db_path.exists():
        raise SwitchError(f"missing state DB: {args.state_db_path}")

    db_rows = select_threads_for_relabel(
        args.state_db_path, args.from_provider, args.source
    )
    rollout_rows = select_threads_by_rollout_provider(
        args.state_db_path, args.from_provider, args.source
    )
    rows = merge_thread_rows(db_rows, rollout_rows)
    if not rows:
        print("No matching threads found.")
        return 0

    print(
        f"Matched {len(rows)} thread(s) from provider '{args.from_provider}' "
        f"to relabel as '{args.to_provider}'."
    )
    print(f"  db_label_matches: {len(db_rows)}")
    print(f"  rollout_meta_matches: {len(rollout_rows)}")
    print(
        "  source_counts: "
        + ", ".join(
            f"{source}={count}"
            for source, count in summarize_thread_sources(rows).items()
        )
    )
    print("  detail: thread titles suppressed")

    if args.dry_run:
        print("  dry_run: no DB or rollout files written")
        return 0

    ensure_parent_dir(args.backup_dir)
    backup_path(args.state_db_path, args.backup_dir, "state_5.sqlite")

    updated_db_rows = update_thread_provider_labels(
        args.state_db_path, db_rows, args.to_provider
    )
    result = patch_rollout_session_meta(rows, args.to_provider)
    result["db_label_matches"] = len(db_rows)
    result["rollout_meta_matches"] = len(rollout_rows)
    result["updated_db_rows"] = updated_db_rows

    result_path = args.backup_dir / (
        f"thread-relabel-{args.source}-{args.from_provider}-to-{args.to_provider}.json"
    )
    write_json(result_path, result)

    print("Relabel completed.")
    print(f"  DB rows updated: {updated_db_rows}")
    print(f"  rollout files patched: {result['patched_rollouts']}")
    print(f"  skipped: {len(result['skipped'])}")
    print(f"  report: {result_path}")
    print("  note: restart Codex Desktop fully so the sidebar reloads.")
    return 0


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path, default=None)
    if data is None:
        raise SwitchError(f"missing profile file: {path}")
    if "profiles" in data:
        profiles = data["profiles"]
    else:
        profiles = data
    if not isinstance(profiles, dict) or not profiles:
        raise SwitchError(f"invalid profile file: {path}")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise SwitchError(f"invalid profile entry: {name}")
        normalize_string_list(profile.get("unset_config_keys", []))
        normalize_string_list(profile.get("unset_auth_keys", []))
        if is_builtin_provider_profile(profile):
            provider_id = profile.get("provider_id")
            if provider_id not in BUILTIN_PROVIDER_IDS:
                raise SwitchError(
                    f"profile '{name}' has invalid builtin provider_id: {provider_id}"
                )
            continue
        provider = profile.get("provider")
        if not isinstance(provider, dict):
            raise SwitchError(f"profile '{name}' is missing a provider block")
        required = {"id", "name", "base_url", "wire_api"}
        missing = [key for key in required if key not in provider]
        if missing:
            raise SwitchError(f"profile '{name}' missing provider keys: {missing}")
    return profiles


def is_builtin_provider_profile(profile: dict[str, Any]) -> bool:
    provider_mode = profile.get("provider_mode")
    if provider_mode is not None:
        return provider_mode == "builtin"
    provider_id = profile.get("provider_id")
    return isinstance(provider_id, str) and provider_id in BUILTIN_PROVIDER_IDS


def get_profile_provider_id(profile: dict[str, Any]) -> str:
    if is_builtin_provider_profile(profile):
        provider_id = profile.get("provider_id")
        if isinstance(provider_id, str):
            return provider_id
        raise SwitchError("builtin profile missing provider_id")
    provider = profile.get("provider")
    if not isinstance(provider, dict) or "id" not in provider:
        raise SwitchError("custom profile missing provider.id")
    return str(provider["id"])


def get_profile_base_url(profile: dict[str, Any]) -> str:
    if is_builtin_provider_profile(profile):
        if get_profile_provider_id(profile) == "openai":
            if profile.get("forced_login_method") == "chatgpt":
                return DEFAULT_CHATGPT_CODEX_BASE_URL
            return str(profile.get("openai_base_url", DEFAULT_OPENAI_BASE_URL))
        return "<builtin>"
    provider = profile.get("provider")
    if not isinstance(provider, dict) or "base_url" not in provider:
        raise SwitchError("custom profile missing provider.base_url")
    return str(provider["base_url"])


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SwitchError("expected a list of strings")
    return list(value)


def get_snapshot_path(snapshot_dir: Path, name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    if not safe_name:
        raise SwitchError("snapshot name cannot be empty")
    return snapshot_dir / f"{safe_name}.json"


def load_snapshot_if_available(
    snapshot_dir: Path, snapshot_name: str | None
) -> dict[str, Any] | None:
    if not snapshot_name:
        return None
    snapshot_path = get_snapshot_path(snapshot_dir, snapshot_name)
    if not snapshot_path.exists():
        return None
    doc = load_json(snapshot_path, default=None)
    if not isinstance(doc, dict):
        raise SwitchError(f"invalid snapshot file: {snapshot_path}")
    config = doc.get("config")
    auth = doc.get("auth")
    if not isinstance(config, dict):
        raise SwitchError(f"snapshot missing config: {snapshot_path}")
    if not isinstance(auth, dict):
        raise SwitchError(f"snapshot missing auth: {snapshot_path}")
    return doc


def merge_snapshot_config(
    current_config: dict[str, Any], snapshot_config: dict[str, Any]
) -> dict[str, Any]:
    next_config = deep_copy(snapshot_config)
    for key in PRESERVE_CONFIG_KEYS_ON_SNAPSHOT_RESTORE:
        if key in current_config:
            next_config[key] = deep_copy(current_config[key])
    return next_config


def is_hard_official_login_profile(profile: dict[str, Any]) -> bool:
    return (
        is_builtin_provider_profile(profile)
        and get_profile_provider_id(profile) == "openai"
        and profile.get("forced_login_method") == "chatgpt"
        and bool(profile.get("clear_auth", False))
    )


def build_official_login_config(
    current_config: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    next_config: dict[str, Any] = {}
    for key in PRESERVE_CONFIG_KEYS_ON_OFFICIAL_LOGIN:
        if key in current_config:
            next_config[key] = deep_copy(current_config[key])
    next_config["model_provider"] = "openai"
    next_config["forced_login_method"] = "chatgpt"
    next_config.pop("openai_base_url", None)
    return next_config


def is_thirdparty_live_config(config: dict[str, Any]) -> bool:
    base_url = get_config_base_url(config)
    return config.get("model_provider") == "api111" and base_url not in {
        "<missing>",
        DEFAULT_OPENAI_BASE_URL,
    }


def should_capture_standby_snapshot(
    current_config: dict[str, Any], profile: dict[str, Any]
) -> bool:
    return is_hard_official_login_profile(profile) and is_thirdparty_live_config(
        current_config
    )


def capture_snapshot_document(
    snapshot_dir: Path,
    name: str,
    config: dict[str, Any],
    auth: dict[str, Any],
) -> Path:
    snapshot_path = get_snapshot_path(snapshot_dir, name)
    payload = {
        "name": name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "config": deep_copy(config),
        "auth": deep_copy(auth),
    }
    write_private_json(snapshot_path, payload)
    return snapshot_path


def get_config_base_url(config: dict[str, Any]) -> str:
    provider_id = config.get("model_provider")
    providers = config.get("model_providers", {})
    if provider_id == "openai":
        return resolve_openai_effective_base_url(config)
    if isinstance(providers, dict):
        provider = providers.get(provider_id, {})
        if isinstance(provider, dict):
            return str(provider.get("base_url", "<missing>"))
    return "<missing>"


def resolve_openai_effective_base_url(
    config: dict[str, Any], auth: dict[str, Any] | None = None
) -> str:
    configured_base_url = config.get("openai_base_url")
    if isinstance(configured_base_url, str) and configured_base_url.strip():
        return configured_base_url
    if get_effective_auth_mode(config, auth) == "chatgpt":
        return DEFAULT_CHATGPT_CODEX_BASE_URL
    return DEFAULT_OPENAI_BASE_URL


def get_effective_auth_mode(
    config: dict[str, Any], auth: dict[str, Any] | None = None
) -> str | None:
    forced_login_method = config.get("forced_login_method")
    if isinstance(forced_login_method, str) and forced_login_method.strip():
        return forced_login_method
    preferred_auth_method = config.get("preferred_auth_method")
    if preferred_auth_method == "chatgpt":
        return "chatgpt"
    if isinstance(auth, dict):
        auth_mode = auth.get("auth_mode")
        if isinstance(auth_mode, str) and auth_mode.strip():
            return auth_mode
    return None


def summarize_thread_sources(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = normalize_thread_source_label(row.get("source"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def normalize_thread_source_label(source: Any) -> str:
    text = str(source or "").strip()
    if text in {"cli", "vscode"}:
        return text
    if not text:
        return "<unknown>"
    if text.startswith("{") and '"subagent"' in text:
        return "subagent"
    return "other"


def load_thread_counts(state_db_path: Path) -> dict[str, int]:
    if not state_db_path.exists():
        return {}
    con = sqlite3.connect(state_db_path)
    try:
        cur = con.cursor()
        rows = cur.execute(
            "select model_provider, count(*) from threads group by model_provider order by count(*) desc"
        ).fetchall()
        return {provider: count for provider, count in rows}
    finally:
        con.close()


def load_rollout_provider_summary(state_db_path: Path) -> dict[str, Any]:
    if not state_db_path.exists():
        return {"counts": {}, "mismatches": 0, "read_errors": 0}

    con = sqlite3.connect(state_db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "select id, rollout_path, model_provider from threads"
        ).fetchall()
    finally:
        con.close()

    counts: dict[str, int] = {}
    mismatches = 0
    read_errors = 0
    for row in rows:
        provider = read_rollout_model_provider(Path(row["rollout_path"]))
        if provider is None:
            read_errors += 1
            continue
        counts[provider] = counts.get(provider, 0) + 1
        if provider != row["model_provider"]:
            mismatches += 1

    return {
        "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "mismatches": mismatches,
        "read_errors": read_errors,
    }


def select_threads_for_relabel(
    state_db_path: Path, from_provider: str, source: str
) -> list[dict[str, Any]]:
    con = sqlite3.connect(state_db_path)
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        sql = [
            "select id, rollout_path, source, cwd, title",
            "from threads",
            "where model_provider = ?",
        ]
        params: list[Any] = [from_provider]
        if source != "all":
            sql.append("and source = ?")
            params.append(source)
        sql.append("order by updated_at desc, id desc")
        rows = cur.execute("\n".join(sql), params).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def select_threads_by_rollout_provider(
    state_db_path: Path, from_provider: str, source: str
) -> list[dict[str, Any]]:
    con = sqlite3.connect(state_db_path)
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        sql = [
            "select id, rollout_path, source, cwd, title",
            "from threads",
        ]
        params: list[Any] = []
        if source != "all":
            sql.append("where source = ?")
            params.append(source)
        sql.append("order by updated_at desc, id desc")
        rows = cur.execute("\n".join(sql), params).fetchall()
    finally:
        con.close()

    matched = []
    for row in rows:
        item = dict(row)
        provider = read_rollout_model_provider(Path(item["rollout_path"]))
        if provider == from_provider:
            matched.append(item)
    return matched


def read_rollout_model_provider(rollout_path: Path) -> str | None:
    if not rollout_path.exists():
        return None
    try:
        with rollout_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        first = json.loads(first_line)
    except Exception:
        return None
    payload = first.get("payload") if isinstance(first, dict) else None
    if not isinstance(payload, dict):
        return None
    provider = payload.get("model_provider")
    return provider if isinstance(provider, str) else None


def merge_thread_rows(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in [*first, *second]:
        merged.setdefault(row["id"], row)
    return list(merged.values())


def update_thread_provider_labels(
    state_db_path: Path, rows: list[dict[str, Any]], to_provider: str
) -> int:
    if not rows:
        return 0
    ids = [row["id"] for row in rows]
    placeholders = ", ".join(["?"] * len(ids))
    con = sqlite3.connect(state_db_path)
    try:
        cur = con.cursor()
        cur.execute(
            f"update threads set model_provider = ? where id in ({placeholders})",
            [to_provider, *ids],
        )
        con.commit()
        return int(cur.rowcount)
    finally:
        con.close()


def patch_rollout_session_meta(
    rows: list[dict[str, Any]], to_provider: str
) -> dict[str, Any]:
    patched = []
    skipped = []
    for row in rows:
        rollout_path = Path(row["rollout_path"])
        if not rollout_path.exists():
            skipped.append(
                {
                    "id": row["id"],
                    "reason": "missing_rollout",
                    "path": str(rollout_path),
                }
            )
            continue

        original_text = rollout_path.read_text(encoding="utf-8")
        has_trailing_newline = original_text.endswith("\n")
        lines = original_text.splitlines()
        if not lines:
            skipped.append(
                {
                    "id": row["id"],
                    "reason": "empty_rollout",
                    "path": str(rollout_path),
                }
            )
            continue

        try:
            first = json.loads(lines[0])
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {
                    "id": row["id"],
                    "reason": f"bad_json:{exc}",
                    "path": str(rollout_path),
                }
            )
            continue

        payload = first.get("payload") if isinstance(first, dict) else None
        if not isinstance(payload, dict):
            skipped.append(
                {
                    "id": row["id"],
                    "reason": "missing_payload",
                    "path": str(rollout_path),
                }
            )
            continue

        payload["model_provider"] = to_provider
        lines[0] = json.dumps(first, ensure_ascii=False, separators=(",", ":"))
        tmp_path = rollout_path.with_suffix(rollout_path.suffix + ".tmp-provider-relabel")
        tmp_path.write_text(
            "\n".join(lines) + ("\n" if has_trailing_newline else ""),
            encoding="utf-8",
        )
        tmp_path.replace(rollout_path)
        patched.append(str(rollout_path))

    return {
        "total": len(rows),
        "patched_rollouts": len(patched),
        "skipped": skipped,
    }


def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SwitchError(f"missing TOML file: {path}")
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SwitchError(f"failed to parse TOML {path}: {exc}") from exc


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SwitchError(f"failed to parse JSON {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_private_json(path: Path, data: Any) -> None:
    write_json(path, data)
    set_private_permissions(path)


def write_text(path: Path, content: str) -> None:
    ensure_parent_dir(path)
    path.write_text(content, encoding="utf-8")


def write_auth_state(path: Path, auth: dict[str, Any], clear_auth: bool) -> str:
    if clear_auth and not auth:
        if path.exists():
            path.unlink()
        return "deleted"
    write_private_json(path, auth)
    return "written"


def ensure_parent_dir(path: Path) -> None:
    target = path if path.suffix == "" and not path.name.endswith(".") else path.parent
    target.mkdir(parents=True, exist_ok=True)


def set_private_permissions(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        return


def backup_path(path: Path, backup_dir: Path, label: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{timestamp}-{label}.bak"
    shutil.copy2(path, backup_path)
    return backup_path


def deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    scalar_items = [(k, v) for k, v in data.items() if not isinstance(v, dict)]
    table_items = [(k, v) for k, v in data.items() if isinstance(v, dict)]

    for key, value in scalar_items:
        lines.append(f"{format_key(key)} = {format_toml_value(value)}")

    for key, value in table_items:
        if lines:
            lines.append("")
        write_table(lines, (key,), value)

    return "\n".join(lines) + "\n"


def write_table(lines: list[str], path: tuple[str, ...], data: dict[str, Any]) -> None:
    lines.append(f"[{'.'.join(format_key(part) for part in path)}]")
    scalar_items = [(k, v) for k, v in data.items() if not isinstance(v, dict)]
    table_items = [(k, v) for k, v in data.items() if isinstance(v, dict)]

    for key, value in scalar_items:
        lines.append(f"{format_key(key)} = {format_toml_value(value)}")

    for key, value in table_items:
        lines.append("")
        write_table(lines, (*path, key), value)


def format_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key, ensure_ascii=False)


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(format_toml_value(item) for item in value) + "]"
    raise SwitchError(f"unsupported TOML value type: {type(value).__name__}")


def mask_secret(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


if __name__ == "__main__":
    raise SystemExit(main())
