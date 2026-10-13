"""Durable, opt-in second Builder lifecycle (new_implementation).

Preparation, dry-run, generation and CPU evaluation are separate operations.
Every POST is preceded by an fsynced worst-case reservation. An uncertain bill
retains that reservation and halts the run, including after process restart.
Mock transports are accepted only for fixture_only unit tests.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from . import providers
from .io_utils import writable


class GenerationError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_hash(value) -> str:
    return digest(json_bytes(value))


def _implementation_hashes() -> dict[str, str]:
    """Bind preparation to the Python files that implement the generation path.

    Repository HEAD may change for reports or checkpoint indexes. Source-file
    hashes detect changes to the executable pipeline without requiring a new
    generation run for unrelated commits. Earlier preparations without these
    hashes are preserved, but cannot be resumed by this implementation.
    """
    directory = Path(__file__).parent
    return {name: digest((directory / name).read_bytes())
            for name in ("generation.py", "providers.py", "core.py")}


def atomic_write(path: Path, data: bytes) -> None:
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value) -> None:
    atomic_write(path, json_bytes(value))


def read_json(path: Path):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise GenerationError(f"invalid or missing run file: {path.name}") from None


@contextmanager
def run_lock(directory: Path):
    directory = writable(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GenerationError("run_locked") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    provider: str = "anthropic"
    model: str | None = None
    api_allowed: bool = False
    max_calls: int = 0
    max_tokens: int = 0
    max_cost_usd: float = 0.0
    max_output_tokens: int = 4096
    max_attempts_per_input: int = 1
    timeout_seconds: float = 60.0
    max_input_bytes: int = 2_000_000
    pricing: dict | None = None
    pricing_max_age_days: int = 7
    limits: dict = dataclasses.field(default_factory=dict)
    provenance: str = "new_implementation"

    @classmethod
    def from_dict(cls, value: dict) -> "GenerationConfig":
        if not isinstance(value, dict):
            raise GenerationError("configuration must be an object")
        unknown = set(value) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise GenerationError("unknown configuration fields")
        config = cls(**value)
        config.validate()
        return config

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        if self.provider not in providers.ENDPOINTS:
            raise GenerationError("unsupported provider")
        if type(self.api_allowed) is not bool:
            raise GenerationError("api_allowed must be boolean")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise GenerationError("model must be explicit or null")
        for name in ("max_calls", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise GenerationError(f"{name} must be a nonnegative integer")
        for name in ("max_output_tokens", "max_attempts_per_input", "max_input_bytes", "pricing_max_age_days"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise GenerationError(f"{name} must be a positive integer")
        for name in ("max_cost_usd", "timeout_seconds"):
            number = getattr(self, name)
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise GenerationError(f"invalid {name}")
        if self.timeout_seconds <= 0:
            raise GenerationError("timeout_seconds must be positive")
        if not isinstance(self.limits, dict):
            raise GenerationError("limits must be an object")
        try:
            from .core import Limits
            Limits(**self.limits)
        except (TypeError, ValueError):
            raise GenerationError("invalid shared validator limits") from None
        if self.provenance not in ("new_implementation", "fixture_only"):
            raise GenerationError("invalid provenance")

    def require_generation(self, permission: bool) -> dict:
        self.validate()
        if not self.api_allowed:
            raise GenerationError("api_disabled")
        if permission is not True:
            raise GenerationError("explicit invocation permission required")
        if not self.model:
            raise GenerationError("explicit model required; no fallback is available")
        if min(self.max_calls, self.max_tokens, self.max_cost_usd) <= 0:
            raise GenerationError("positive call, token and USD budget limits required")
        pricing = self.pricing
        if not isinstance(pricing, dict) or pricing.get("verified") is not True:
            raise GenerationError("verified dated model pricing required")
        if pricing.get("provider") != self.provider or pricing.get("model") != self.model:
            raise GenerationError("pricing provider/model mismatch")
        if pricing.get("billing_mode") != "standard_text_no_tools_no_cache_writes":
            raise GenerationError("unsupported billing mode")
        # Long-context model tiers can cost more than headline token pricing.
        # Verified rates must bound every request allowed by the context ceiling.
        if pricing.get("rate_scope") != "upper_bound_full_context_window":
            raise GenerationError("verified full-context pricing rate upper bounds required")
        try:
            verified_date = dt.date.fromisoformat(pricing["verified_on"])
            age = (dt.datetime.now(dt.timezone.utc).date() - verified_date).days
            if not 0 <= age <= self.pricing_max_age_days:
                raise ValueError
            for name in ("input_usd_per_million", "output_usd_per_million"):
                value = pricing[name]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError
            if type(pricing["context_window_tokens"]) is not int or pricing["context_window_tokens"] <= 0:
                raise ValueError
            official_hosts = {
                "openai": {"openai.com", "developers.openai.com", "platform.openai.com"},
                "anthropic": {"anthropic.com", "www.anthropic.com", "docs.anthropic.com", "platform.claude.com"},
            }[self.provider]
            for name in ("pricing_source_url", "context_source_url"):
                url = urlsplit(pricing[name])
                if url.scheme != "https" or url.hostname not in official_hosts or url.username or url.password:
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            raise GenerationError("invalid, stale or incomplete verified pricing/context record") from None
        return pricing


def _record_identity(record: dict) -> dict:
    return {"input_id": record["input_id"], "input_hash": object_hash(record)}


def prepare_generation(records, config: GenerationConfig | dict, output_dir, *, commit: str, prompt: str | None = None) -> dict:
    """Persist immutable inputs/configuration without reading keys or making POSTs."""
    config = config if isinstance(config, GenerationConfig) else GenerationConfig.from_dict(config)
    config.validate()
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise GenerationError("explicit hexadecimal commit required")
    if prompt is None:
        prompt = (Path(__file__).parent / "prompts" / "builder.txt").read_text("utf-8")
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationError("nonempty shared prompt required")
    normalized, seen = [], set()
    for original in records:
        if not isinstance(original, dict):
            raise GenerationError("input record must be an object")
        record = dict(original)
        for key in ("scene_id", "condition", "input_text"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise GenerationError(f"nonempty input {key} required")
        if len(record["input_text"].encode("utf-8")) > config.max_input_bytes:
            raise GenerationError("input exceeds max_input_bytes; truncation is forbidden")
        if "input_id" not in record:
            record["input_id"] = object_hash(record)
        if not isinstance(record["input_id"], str) or not record["input_id"] or record["input_id"] in seen:
            raise GenerationError("input_id must be nonempty and unique")
        seen.add(record["input_id"])
        normalized.append(record)
    config_dict = config.to_dict()
    identity = {
        "version": 1,
        "provenance": config.provenance,
        "commit": commit,
        "implementation_hashes": _implementation_hashes(),
        "provider": config.provider,
        "model": config.model,
        "config_hash": object_hash(config_dict),
        "prompt_hash": digest(prompt.encode("utf-8")),
        "inputs_hash": object_hash(normalized),
        "inputs": [_record_identity(record) for record in normalized],
    }
    directory = writable(output_dir)
    with run_lock(directory):
        if (directory / "manifest.json").exists():
            manifest, _, _, _, _ = _load_run(directory)
            if manifest["identity"] != identity:
                raise GenerationError("resume_identity_mismatch")
            return manifest
        if any(path.name != "run.lock" for path in directory.iterdir()):
            raise GenerationError("refusing to replace an incomplete or unrelated run")
        manifest = {"identity": identity, "created_at_utc": utc_now(), "research_api_status": "NOT_RUN"}
        write_json(directory / "inputs.json", normalized)
        write_json(directory / "config.json", config_dict)
        atomic_write(directory / "prompt.txt", prompt.encode("utf-8"))
        write_json(directory / "state.json", {"identity_hash": object_hash(identity), "attempts": [], "halt_reason": None})
        write_json(directory / "manifest.json", manifest)
        return manifest


def _verify_artifact(directory: Path, descriptor: dict) -> Path:
    path = Path(descriptor["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise GenerationError("invalid_artifact_path")
    target = directory / path
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != directory.parent):
        raise GenerationError("artifact_symlink_rejected")
    try:
        data = target.read_bytes()
    except OSError:
        raise GenerationError("resume_output_missing") from None
    if digest(data) != descriptor["sha256"] or len(data) != descriptor["bytes"]:
        raise GenerationError("resume_output_integrity_mismatch")
    return target


def _load_run(directory: Path):
    manifest = read_json(directory / "manifest.json")
    identity = manifest["identity"]
    if identity.get("implementation_hashes") != _implementation_hashes():
        raise GenerationError("resume_implementation_mismatch")
    config_dict = read_json(directory / "config.json")
    records = read_json(directory / "inputs.json")
    prompt = (directory / "prompt.txt").read_text("utf-8")
    state = read_json(directory / "state.json")
    if (
        object_hash(config_dict) != identity["config_hash"]
        or object_hash(records) != identity["inputs_hash"]
        or digest(prompt.encode("utf-8")) != identity["prompt_hash"]
        or [_record_identity(record) for record in records] != identity["inputs"]
        or state["identity_hash"] != object_hash(identity)
    ):
        raise GenerationError("resume_identity_mismatch")
    config = GenerationConfig.from_dict(config_dict)
    expected_ids = {record["input_id"]: object_hash(record) for record in records}
    attempt_ids = set()
    for attempt in state["attempts"]:
        if not isinstance(attempt.get("attempt_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", attempt["attempt_id"]):
            raise GenerationError("invalid_attempt_id")
        if attempt["attempt_id"] in attempt_ids:
            raise GenerationError("duplicate_attempt_id")
        attempt_ids.add(attempt["attempt_id"])
        attempt_identity = attempt["identity"]
        record = next((record for record in records if record["input_id"] == attempt["input_id"]), None)
        payload_hash = object_hash(providers.build_payload(
            config.provider, config.model, prompt, record["input_text"], config.max_output_tokens,
        )) if record is not None and config.model else None
        if (
            attempt_identity.get("run_identity_hash") != object_hash(identity)
            or expected_ids.get(attempt["input_id"]) != attempt_identity.get("input_hash")
            or attempt_identity.get("config_hash") != identity["config_hash"]
            or attempt_identity.get("prompt_hash") != identity["prompt_hash"]
            or attempt_identity.get("commit") != identity["commit"]
            or attempt_identity.get("model") != identity["model"]
            or attempt_identity.get("provider") != identity["provider"]
            or attempt_identity.get("request_payload_hash") != payload_hash
        ):
            raise GenerationError("resume_attempt_identity_mismatch")
        if attempt["status"] not in ("reserved", "failed", "success"):
            raise GenerationError("invalid_attempt_status")
        for name in ("charged_input_tokens", "charged_output_tokens"):
            if type(attempt.get(name)) is not int or attempt[name] < 0:
                raise GenerationError("invalid_budget_ledger")
        charge = attempt.get("charged_cost_usd")
        if type(charge) not in (int, float) or not math.isfinite(charge) or charge < 0:
            raise GenerationError("invalid_budget_ledger")
        if attempt.get("usage") is None and attempt["status"] != "reserved" and not state["halt_reason"]:
            raise GenerationError("unknown_usage_halt_missing")
        for descriptor in attempt.get("artifacts", {}).values():
            _verify_artifact(directory, descriptor)
        if attempt.get("metadata_artifact"):
            metadata_path = _verify_artifact(directory, attempt["metadata_artifact"])
            metadata = read_json(metadata_path)
            if metadata != {key: value for key, value in attempt.items() if key != "metadata_artifact"}:
                raise GenerationError("resume_attempt_metadata_mismatch")
        elif attempt["status"] != "reserved":
            raise GenerationError("resume_attempt_metadata_missing")
        if attempt["status"] == "success" and not {"raw.json", "parsed.json", "voxels.json"}.issubset(attempt.get("artifacts", {})):
            raise GenerationError("resume_success_outputs_missing")
    attempts_directory = directory / "attempts"
    if attempts_directory.exists() and {path.name for path in attempts_directory.iterdir()} - attempt_ids:
        raise GenerationError("untracked_attempt_outputs")
    return manifest, config, records, prompt, state


def _budget_totals(state: dict) -> dict:
    return {
        "calls": len(state["attempts"]),
        "input_tokens": sum(attempt["charged_input_tokens"] for attempt in state["attempts"]),
        "output_tokens": sum(attempt["charged_output_tokens"] for attempt in state["attempts"]),
        "cost_usd_upper_bound": sum(attempt["charged_cost_usd"] for attempt in state["attempts"]),
        "unknown_usage_attempts": sum(attempt.get("usage") is None for attempt in state["attempts"]),
    }


def _summary(manifest: dict, state: dict) -> dict:
    return {
        "provenance": manifest["identity"]["provenance"],
        "input_count": len(manifest["identity"]["inputs"]),
        "success_count": sum(attempt["status"] == "success" for attempt in state["attempts"]),
        "attempt_count": len(state["attempts"]),
        "halt_reason": state["halt_reason"],
        "research_api_status": "NOT_RUN" if not state["attempts"] else (
            "FIXTURE_ONLY" if manifest["identity"]["provenance"] == "fixture_only" else "ATTEMPTED"
        ),
        "budget": _budget_totals(state),
    }


def dry_run_generation(output_dir) -> dict:
    """Read and verify the complete run, making no network requests."""
    directory = writable(output_dir)
    with run_lock(directory):
        manifest, config, records, prompt, state = _load_run(directory)
        result = _summary(manifest, state)
        result["api_allowed"] = config.api_allowed
        result["required_api_key_environment_variable"] = providers.KEY_NAMES[config.provider]
        result["request_payload_hashes"] = [
            object_hash(providers.build_payload(config.provider, config.model, prompt, record["input_text"], config.max_output_tokens))
            for record in records
        ] if config.model else []
        result["generation_configuration_status"] = "BLOCKED"
        try:
            config.require_generation(True)
            result["generation_configuration_status"] = "READY_REQUIRES_EXPLICIT_INVOCATION_PERMISSION"
        except GenerationError as exc:
            result["generation_block_reason"] = str(exc)
        return result


def _save_artifact(directory: Path, attempt: dict, name: str, data: bytes) -> None:
    relative = Path("attempts") / attempt["attempt_id"] / name
    target = directory / relative
    if target.exists() and target.read_bytes() != data:
        raise GenerationError("refusing to replace an existing attempt artifact")
    atomic_write(target, data)
    attempt["artifacts"][name] = {"path": relative.as_posix(), "sha256": digest(data), "bytes": len(data)}


def _complete_attempt(directory: Path, state: dict, attempt: dict) -> None:
    attempt["finished_at_utc"] = utc_now()
    metadata = json_bytes({key: value for key, value in attempt.items() if key != "metadata_artifact"})
    relative = Path("attempts") / attempt["attempt_id"] / "attempt.json"
    atomic_write(directory / relative, metadata)
    attempt["metadata_artifact"] = {"path": relative.as_posix(), "sha256": digest(metadata), "bytes": len(metadata)}
    write_json(directory / "state.json", state)


def generate(output_dir, *, permission: bool = False, transport=None) -> dict:
    """Opt-in POSTs, bounded retries, and full-identity resume under one flock.

There is no automatic retry after unknown usage, timeout or transport ambiguity.
An operator must reconcile billing outside this implementation before such a
run can continue. Known-usage parsing/validation/execution failures can retry.
"""
    from .core import Limits, ValidationError, execute, grid_to_payload, parse_actions, validate_actions

    directory = writable(output_dir)
    with run_lock(directory):
        manifest, config, records, prompt, state = _load_run(directory)
        pricing = config.require_generation(permission)
        if transport is not None and config.provenance != "fixture_only":
            raise GenerationError("mock transport is allowed only for fixture_only unit tests")
        if state["halt_reason"]:
            raise GenerationError(f"run_halted: {state['halt_reason']}")
        if any(attempt["status"] == "reserved" for attempt in state["attempts"]):
            state["halt_reason"] = "unreconciled_prior_reservation"
            write_json(directory / "state.json", state)
            raise GenerationError("run_halted: unreconciled_prior_reservation")
        if not os.environ.get(providers.KEY_NAMES[config.provider]):
            raise GenerationError("missing_api_key")
        try:
            limits = Limits(**config.limits)
        except (TypeError, ValueError):
            raise GenerationError("invalid shared validator limits") from None
        identity = manifest["identity"]
        for record in records:
            while True:
                prior = [attempt for attempt in state["attempts"] if attempt["input_id"] == record["input_id"]]
                if any(attempt["status"] == "success" for attempt in prior) or len(prior) >= config.max_attempts_per_input:
                    break
                reserve_input = pricing["context_window_tokens"]
                reserve_output = config.max_output_tokens
                reserve_cost = providers.cost_ceiling(reserve_input, reserve_output, pricing)
                totals = _budget_totals(state)
                if (
                    totals["calls"] + 1 > config.max_calls
                    or totals["input_tokens"] + totals["output_tokens"] + reserve_input + reserve_output > config.max_tokens
                    or sum((Decimal(str(item["charged_cost_usd"])) for item in state["attempts"]), Decimal(0))
                       + Decimal(str(reserve_cost)) > Decimal(str(config.max_cost_usd))
                ):
                    return {**_summary(manifest, state), "generation_stop_reason": "budget_exhausted"}
                payload = providers.build_payload(config.provider, config.model, prompt, record["input_text"], config.max_output_tokens)
                attempt = {
                    "attempt_id": uuid.uuid4().hex,
                    "input_id": record["input_id"],
                    "scene_id": record["scene_id"],
                    "condition": record["condition"],
                    "identity": {
                        "run_identity_hash": object_hash(identity), "input_hash": object_hash(record),
                        "config_hash": identity["config_hash"], "prompt_hash": identity["prompt_hash"],
                        "commit": identity["commit"], "provider": config.provider, "model": config.model,
                        "request_payload_hash": object_hash(payload),
                    },
                    "started_at_utc": utc_now(), "status": "reserved", "failure_kind": None,
                    "usage": None, "response_id": None, "response_model": None,
                    "charged_input_tokens": reserve_input, "charged_output_tokens": reserve_output,
                    "charged_cost_usd": reserve_cost, "billing": "worst_case_reservation",
                    "artifacts": {},
                }
                state["attempts"].append(attempt)
                # This reservation survives a crash before, during or after POST.
                write_json(directory / "state.json", state)
                response = None
                try:
                    raw = providers.send_request(config.provider, payload, api_allowed=config.api_allowed,
                                                 permission=permission, timeout=config.timeout_seconds, transport=transport)
                    if not isinstance(raw, bytes):
                        raise providers.ProviderError("response_parse_failure")
                    _save_artifact(directory, attempt, "raw.json", raw)
                    response = providers.parse_response(config.provider, raw)
                    attempt["response_id"], attempt["response_model"] = response.response_id, response.model
                    input_tokens, output_tokens, cost = providers.usage_upper_cost(config.provider, response.usage, pricing)
                    attempt["usage"] = response.usage
                    attempt["charged_input_tokens"], attempt["charged_output_tokens"] = input_tokens, output_tokens
                    attempt["charged_cost_usd"], attempt["billing"] = cost, "known_usage_conservative_cost"
                    if input_tokens > reserve_input or output_tokens > reserve_output or cost > reserve_cost:
                        state["halt_reason"] = "usage_exceeded_verified_reservation"
                        raise providers.ProviderError("budget_violation")
                    if response.model != config.model or not response.response_id:
                        raise providers.ProviderError("response_identity_failure")
                    if not response.complete:
                        raise providers.ProviderError("incomplete_response")
                    if response.text is None:
                        raise providers.ProviderError("parse_failure")
                    try:
                        actions = parse_actions(response.text, limits=limits)
                    except (ValueError, ValidationError):
                        # Separate malformed JSON from syntactically valid but invalid actions.
                        try:
                            json.loads(response.text)
                        except (ValueError, TypeError):
                            raise providers.ProviderError("parse_failure") from None
                        raise providers.ProviderError("validation_failure") from None
                    _save_artifact(directory, attempt, "parsed.json", json_bytes({"operations": actions}))
                    try:
                        actions = validate_actions(actions, limits=limits)
                    except (ValueError, ValidationError):
                        raise providers.ProviderError("validation_failure") from None
                    try:
                        grid = execute(actions, limits=limits)
                        voxel_payload = grid_to_payload(grid)
                    except Exception:
                        raise providers.ProviderError("execution_failure") from None
                    _save_artifact(directory, attempt, "voxels.json", json_bytes(voxel_payload))
                    attempt["status"] = "success"
                except providers.ProviderError as exc:
                    attempt["status"] = "failed"
                    attempt["failure_kind"] = exc.kind
                    attempt["http_status"] = exc.status_code
                    if attempt["usage"] is None:
                        state["halt_reason"] = "unknown_usage"
                _complete_attempt(directory, state, attempt)
                if state["halt_reason"]:
                    return _summary(manifest, state)
        return _summary(manifest, state)


def evaluate_generation(output_dir, gt_by_scene, *, alignment_radius: int = 2) -> list[dict]:
    """CPU-only stage. Failed attempts keep null metrics rather than favorable 0s."""
    from .core import evaluate_pair, grid_to_payload, payload_to_grid

    directory = writable(output_dir)
    with run_lock(directory):
        manifest, _, records, _, state = _load_run(directory)
        rows = []
        for record in records:
            attempts = [attempt for attempt in state["attempts"] if attempt["input_id"] == record["input_id"]]
            success = next((attempt for attempt in attempts if attempt["status"] == "success"), None)
            last = success or (attempts[-1] if attempts else None)
            row = {"input_id": record["input_id"], "scene_id": record["scene_id"], "condition": record["condition"],
                   "description_model": record.get("description_model"), "provenance": manifest["identity"]["provenance"],
                   "attempt_id": last["attempt_id"] if last else None, "status": last["status"] if last else "NOT_RUN",
                   "failure_kind": last["failure_kind"] if last else None, "metrics": None}
            if success:
                gt = gt_by_scene.get(record["scene_id"])
                if gt is None:
                    row["status"] = "GT_MISSING"
                else:
                    voxel_path = _verify_artifact(directory, success["artifacts"]["voxels.json"])
                    grid = payload_to_grid(read_json(voxel_path))
                    row["metrics"] = evaluate_pair(gt, grid, alignment_radius=alignment_radius)
                    evaluation_identity = {"alignment_radius": alignment_radius, "gt_hash": object_hash(grid_to_payload(gt))}
                    _save_artifact(directory, success, f"metrics.{object_hash(evaluation_identity)}.json", json_bytes({
                        **evaluation_identity,
                        "metrics": row["metrics"],
                    }))
                    _complete_attempt(directory, state, success)
            rows.append(row)
        write_json(directory / "evaluation.json", {"identity_hash": object_hash(manifest["identity"]),
                                                   "alignment_radius": alignment_radius, "rows": rows})
        return rows
