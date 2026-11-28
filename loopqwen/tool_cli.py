"""Use the offline recurrent assistant with caller-selected local tools."""

import argparse
import json
from pathlib import Path


def add_arguments(p, checkpoint=None):
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=checkpoint,
        required=checkpoint is None,
        help="Combined language and tool export",
    )
    p.add_argument(
        "--tool-format",
        choices=("compact", "qwen"),
        default="compact",
        help="Use compact JSON or the official Qwen function-call template",
    )
    policies = p.add_mutually_exclusive_group()
    policies.add_argument(
        "--tool-adapter",
        type=Path,
        help="Load a compatible trained tool adapter over the existing frozen backbone",
    )
    policies.add_argument(
        "--session-policy",
        type=Path,
        help="Use compatible reasoning and action adapters sharing one resident backbone",
    )
    p.add_argument(
        "--catalog", type=Path, help="JSON file with read-only lookup definitions"
    )
    arithmetic = p.add_mutually_exclusive_group()
    arithmetic.add_argument(
        "--calculator", action="store_true", help="Expose exact arithmetic expressions"
    )
    arithmetic.add_argument(
        "--checked-calculator",
        action="store_true",
        help="Accept arithmetic expressions and verify every side of an equality exactly",
    )
    p.add_argument(
        "--packages",
        action="store_true",
        help="Expose installed versions and dependencies",
    )
    p.add_argument(
        "--wheel",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Read the embedded metadata of a local Python wheel. Can be repeated.",
    )
    p.add_argument(
        "--database", type=Path, help="Open an existing SQLite database read-only"
    )
    p.add_argument(
        "--relational-db",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Expose schema inspection and read-only SQL joins on a supplied SQLite database.",
    )
    p.add_argument(
        "--table", help="Name of the one database table exposed to the assistant"
    )
    p.add_argument(
        "--resource",
        action="append",
        choices=("calendar", "zone", "unicode", "distribution"),
        default=[],
        help="Expose both read-only properties of a local resource. Can be repeated.",
    )
    p.add_argument(
        "--keyed-csv",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Look up a CSV field by the exact text in a named key column.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Expose a UTF-8 CSV file for read-only row counts and cell lookups. Can be repeated.",
    )
    p.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help="Expose the byte size and line count of this file. Can be repeated.",
    )
    p.add_argument(
        "--document",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Allow passage search over this UTF-8 text or Markdown file. Can be repeated.",
    )
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the complete UTF-8 user request from disk",
    )
    p.add_argument(
        "--follow-up",
        action="append",
        default=[],
        metavar="TEXT",
        help="Send another user message in the same live conversation. Can be repeated.",
    )
    p.add_argument(
        "--retain-conversation-prefix",
        action="store_true",
        help="Retain the first complete native prefill blocks for later calls and user messages in this tool conversation.",
    )
    p.add_argument(
        "--extend-tool-prefix",
        action="store_true",
        help="On a matching tool turn, extend the saved aligned prefill state through the most recent prompt (up to 4,096 tokens).",
    )
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument(
        "--staged-context-dir",
        help="Stage full 16-bit attention history in this scratch directory on a fixed ordinary-depth or tied-query decoder",
    )
    p.add_argument(
        "--cache-state-prefill",
        action="store_true",
        help="Evaluate native cache states after intermediate blocks on a fixed resident decoder",
    )
    p.add_argument(
        "--reuse-prefix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint’s qualified multi-turn prefix policy",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="Include exact prompt and output tokens and resident attention-site histories",
    )


def open_assistant(args):
    bundle = Path(args.checkpoint) / "standalone-middle.json"
    if bundle.is_file():
        from . import REVISION
        from .io import digest, read

        spec = read(bundle)
        if (
            spec["schema"] not in (1, 2)
            or spec["base_revision"] != REVISION
            or getattr(args, "tool_format", "compact") != "qwen"
            or getattr(args, "session_policy", None) is not None
            or getattr(args, "tool_adapter", None) is not None
        ):
            raise ValueError(
                "Use the standalone checkpoint with --tool-format qwen and its built-in policy"
            )
        if (
            spec["schema"] == 1
            and spec["arm"] == "base"
            and spec["policy_sha256"] is None
        ):
            from .tool_compare import DenseAssistant

            assistant = DenseAssistant(
                True, "qwen", args.checkpoint, require_release=True
            )
            assistant.native_plain = True
            return assistant
        if (
            spec["arm"] == "loop"
            and digest(bundle.parent / "registration.json") == spec["policy_sha256"]
        ):
            from .middle_depth import MiddleAssistant

            if read(bundle.parent / "registration.json")["donor"] != "base":
                raise ValueError(
                    "Standalone loop requires its local certified Base donor"
                )
            if spec["schema"] == 1:
                return MiddleAssistant(args.checkpoint, args.checkpoint)
            from .middle_read import install

            policy_file = bundle.parent / "read-policy.json"
            if digest(policy_file) != spec["read_policy_sha256"]:
                raise ValueError("Middle-read policy identity mismatch")
            policy = read(policy_file)
            weights = bundle.parent / "controller.safetensors"
            if (
                policy["schema"] != 1
                or policy["parent"] != digest(bundle.parent / "adapters.safetensors")
                or policy["weights"] != digest(weights)
            ):
                raise ValueError("Middle-read controller or parent identity mismatch")
            assistant = MiddleAssistant(args.checkpoint, args.checkpoint)
            layer = install(assistant._tool_model, policy["reads"], policy["seed"])
            layer.step.load_weights(str(weights), strict=True)
            assistant._tool_model.eval()
            return assistant
        raise ValueError("Unexpected standalone model identity")
    if getattr(args, "session_policy", None) is not None:
        from .io import read

        policy = read(args.session_policy / "registration.json")
        if policy.get("middle_depth"):
            from .middle_depth import MiddleAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError("The middle-span model uses --tool-format qwen")
            return MiddleAssistant(args.checkpoint, args.session_policy)
        if policy.get("shared_query"):
            from .shared_query_assistant import SharedQueryAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError("The shared-query model uses --tool-format qwen")
            return SharedQueryAssistant(args.checkpoint, args.session_policy)
        if policy.get("conditional_reentry"):
            from .conditional_reentry import ConditionalAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError("The conditional model uses --tool-format qwen")
            return ConditionalAssistant(args.checkpoint, args.session_policy)
        if policy.get("reentry_block"):
            from .reentry import ReentryAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError("The recurrent block uses --tool-format qwen")
            return ReentryAssistant(args.checkpoint, args.session_policy)
        if policy.get("visit_residual"):
            from .visit_residual import VisitResidualAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError("The second-visit policy uses --tool-format qwen")
            return VisitResidualAssistant(args.checkpoint, args.session_policy)
        if policy.get("trained_instruction_adapter"):
            from .cross_view import CrossViewAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError(
                    "The cross-view reader uses the native Qwen tool format"
                )
            return CrossViewAssistant(args.checkpoint, args.session_policy)
        if policy.get("file_evidence"):
            from .file_evidence import FileEvidenceAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError("File evidence requires --tool-format qwen")
            return FileEvidenceAssistant(args.checkpoint, args.session_policy)
        if policy.get("instruction_interpolation"):
            from .interpolated_instruction import InterpolatedAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError("Interpolated instruction requires --tool-format qwen")
            return InterpolatedAssistant(args.checkpoint, args.session_policy)
        if policy.get("observation_review"):
            from .observation_review import ObservationAssistant

            if getattr(args, "tool_format", "compact") != "qwen":
                raise ValueError("Observation review requires --tool-format qwen")
            return ObservationAssistant(args.checkpoint, args.session_policy)
        if getattr(args, "tool_format", "compact") != "compact":
            raise ValueError(
                "This session policy was fitted with the compact tool template"
            )
        if policy.get("callback_recovery"):
            from .callback_recovery import RecoveryAssistant

            return RecoveryAssistant(args.checkpoint, args.session_policy)
        if policy.get("unified"):
            from .unified_session import UnifiedAssistant

            return UnifiedAssistant(args.checkpoint, args.session_policy)
        from .shared_session import SharedAssistant

        return SharedAssistant(args.checkpoint, args.session_policy)
    from .tool_model import Assistant

    return Assistant(
        args.checkpoint,
        tool_adapter=args.tool_adapter,
        tool_format=getattr(args, "tool_format", "compact"),
    )


def respond(args, parser):
    from contextlib import ExitStack, closing

    from .io import read
    from .tool_catalog import load_catalog

    if getattr(args, "prompt_file", None) is not None:
        args.prompt = args.prompt_file.read_bytes().decode("utf-8")
    if bool(args.database) != bool(args.table):
        parser.error("--database and --table must be supplied together")
    with ExitStack() as resources:
        supplied = load_catalog(read(args.catalog)) if args.catalog is not None else []
        if getattr(args, "csv", None):
            from .csv_tools import tools as csv_tools

            supplied.extend(csv_tools(args.csv))
        if getattr(args, "keyed_csv", None):
            from .csv_tools import tools as csv_tools

            supplied.extend(csv_tools(args.keyed_csv, keyed=True))
        if getattr(args, "file", None):
            from .file_tools import tools as file_tools

            supplied.extend(file_tools(args.file))
        if getattr(args, "document", None):
            from .document_tools import documents

            supplied.extend(resources.enter_context(documents(args.document)))
        if args.calculator or getattr(args, "checked_calculator", False):
            from .calculator_tool import tool

            supplied.append(
                tool(check_equalities=getattr(args, "checked_calculator", False))
            )
        if args.packages:
            from .package_tools import tools as packages

            supplied.extend(packages())
        if getattr(args, "wheel", None):
            from .wheel_tools import tools as wheel_tools

            supplied.extend(wheel_tools(args.wheel))
        if getattr(args, "relational_db", None):
            from .relational_tools import tools as relational_tools

            supplied.extend(relational_tools(args.relational_db))
        if args.resource:
            from .resource_tools import tools as local_tools

            for family in dict.fromkeys(args.resource):
                supplied.extend(local_tools(family))
        if args.database:
            import sqlite3

            from .table_tool import database_tool

            uri = args.database.resolve().as_uri() + "?mode=ro"
            db = resources.enter_context(closing(sqlite3.connect(uri, uri=True)))
            if getattr(args, "compare_base", False):
                db.execute("BEGIN")
            supplied.append(database_tool(db, args.table))
        if not supplied and not getattr(args, "allow_plain", False):
            parser.error(
                "Select a file, document, CSV, resource, catalog, calculator, package environment, wheel or database table"
            )
        retain = getattr(args, "retain_conversation_prefix", False)
        if retain and not supplied:
            parser.error("--retain-conversation-prefix requires at least one tool")
        options = {
            "max_tokens": args.max_tokens,
            "max_turns": args.max_turns,
            "reuse_prefix": args.reuse_prefix,
        }
        if getattr(args, "extend_tool_prefix", False):
            if not supplied or args.reuse_prefix is False:
                parser.error(
                    "--extend-tool-prefix requires a supplied tool and enabled prefix reuse"
                )
            options.update(reuse_prefix=True, extend_tool_prefix=True)
        if args.staged_context_dir is not None:
            options["staged_context_dir"] = args.staged_context_dir
        if args.trace:
            options["record_kv_history"] = True
        if getattr(args, "cache_state_prefill", False):
            options["prefill_cache_state"] = True
        prompts = [args.prompt, *args.follow_up]
        if any(not text.strip() for text in prompts):
            parser.error("Every conversation message needs text")
        if getattr(args, "compare_base", False):
            from .tool_compare import compare

            return compare(args, supplied, options)
        assistant = open_assistant(args)
        if args.follow_up or retain:
            chat = (
                assistant.conversation(supplied, retain_prefix=True)
                if retain
                else assistant.conversation(supplied)
            )
            if not args.follow_up:
                return present(chat.respond(args.prompt, **options), args.trace)
            return {
                "exchanges": [
                    {
                        "question": text,
                        **present(chat.respond(text, **options), args.trace),
                    }
                    for text in prompts
                ]
            }
        return present(assistant.respond(args.prompt, supplied, **options), args.trace)


def present(result, trace):
    if trace:
        return result
    calls = []
    for t in result["turns"]:
        if "observation" in t:
            calls.append({**t["call"], "result": t["observation"]})
        elif "observations" in t:
            calls.extend(
                {**a, "result": v}
                for a, v in zip(t["calls"], t["observations"], strict=True)
            )
    return {k: v for k, v in result.items() if k != "turns"} | {"calls": calls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, Path("."))
    print(json.dumps(respond(parser.parse_args(), parser), indent=2))
