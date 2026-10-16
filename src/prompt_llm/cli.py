"""Command line interface for prompt_llm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__, profiles
from .providers import ALIASES, DEFAULT_TIMEOUT, PROVIDERS, LLMError, Result, complete, get_provider, normalize_tools, stream

COMMANDS = {"ask", "chat", "profile", "providers", "version"}
SHORTCUTS = set(PROVIDERS) | set(ALIASES)  # `prompt_llm openai "..."` == `prompt_llm ask -P openai "..."`


class CLIError(Exception):
    """A user-facing error, printed without a traceback."""


def _mask(value: str) -> str:
    return value[:4] + "..." + value[-4:] if len(value) > 12 else "****"


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        raise CLIError(f"Cannot read {path}: {e.strerror}") from None


def _build_prompt(words: List[str], files: List[str]) -> str:
    """Join prompt words ('-' inserts stdin; no words with piped input reads stdin) and append files."""
    if not words and sys.stdin is not None and not sys.stdin.isatty():
        words = ["-"]
    prompt = " ".join(sys.stdin.read() if word == "-" else word for word in words).strip()
    for path in files:
        prompt += f"\n\n{path}:\n```\n{_read_file(path).rstrip()}\n```"
    return prompt.strip()


def _load_tools(path: str) -> List[dict]:
    try:
        return normalize_tools(json.loads(_read_file(path)))
    except ValueError as e:
        raise CLIError(f"Invalid tools file {path}: {e}") from None


def _settings(args: argparse.Namespace) -> dict:
    """Resolve request settings: command line option > profile > environment > provider default."""
    _, profile = profiles.get(args.profile)
    provider = get_provider(args.provider or profile.get("provider") or "openai")
    # api_key, model and base_url belong to the profile's provider and must not leak to another one.
    own = profile if get_provider(profile.get("provider") or "openai") is provider else {}

    api_key = args.api_key or own.get("api_key") or (os.environ.get(provider.env_var) if provider.env_var else None)
    base_url = own.get("base_url")
    if not api_key and provider.env_var and not base_url:
        raise CLIError(
            f"No API key for {provider.name}. Set {provider.env_var}, pass --api-key, or save it in a profile: "
            f"prompt_llm profile save {provider.name} --provider {provider.name} --api-key ..."
        )
    temperature = args.temperature if args.temperature is not None else profile.get("temperature")
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": args.model or own.get("model") or provider.default_model,
        "system": args.system if args.system is not None else profile.get("system"),
        "temperature": float(temperature) if temperature is not None else None,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }


def _stream_to_stdout(settings: dict, messages: List[dict]) -> str:
    chunks = []
    for chunk in stream(messages=messages, **settings):
        chunks.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(chunks)


def _format_text(result: Result) -> str:
    parts = [result.text] if result.text else []
    if result.tool_calls:
        parts.append(json.dumps(result.tool_calls, indent=2, ensure_ascii=False))
    return "\n".join(parts)


def _warn_truncated(result: Result) -> None:
    if result.stop_reason in ("length", "max_tokens"):
        print("Warning: response was cut off at the token limit (raise --max-tokens).", file=sys.stderr)


# --- commands ----------------------------------------------------------------

def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    tools = _load_tools(args.tools) if args.tools else None
    prompt = _build_prompt(args.prompt, args.file)
    if not prompt:
        raise CLIError("Empty prompt. Pass text, '-' to read stdin, or -f FILE.")
    messages = [{"role": "user", "content": prompt}]

    as_json = args.json or bool(args.out and args.out.lower().endswith(".json") and not args.raw)
    plain = not (args.raw or as_json or tools or args.out)
    if plain and (args.stream if args.stream is not None else sys.stdout.isatty()):
        _stream_to_stdout(settings, messages)
        return 0

    result = complete(messages=messages, tools=tools, **settings)
    if args.raw:
        output = json.dumps(result.raw, indent=2, ensure_ascii=False)
    elif as_json:
        output = json.dumps({"provider": settings["provider"].name, **result.to_dict()}, indent=2, ensure_ascii=False)
    else:
        output = _format_text(result)
    _warn_truncated(result)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(output + "\n")
        except OSError as e:
            raise CLIError(f"Cannot write {args.out}: {e.strerror}") from None
        print(f"Saved to {args.out}", file=sys.stderr)
    else:
        print(output)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    settings = _settings(args)
    use_stream = args.stream is not False
    try:
        import readline  # noqa: F401  (enables line editing and history in input())
    except ImportError:
        pass
    print(f"{settings['provider'].name} · {settings['model']}  (/reset clears history, /exit or Ctrl-D quits)", file=sys.stderr)

    messages: List[dict] = []
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            messages.clear()
            print("History cleared.", file=sys.stderr)
            continue

        messages.append({"role": "user", "content": line})
        try:
            if use_stream:
                reply = _stream_to_stdout(settings, messages)
            else:
                result = complete(messages=messages, **settings)
                reply = result.text
                print(reply)
                _warn_truncated(result)
        except LLMError as e:
            print(f"Error: {e}", file=sys.stderr)
            messages.pop()
            continue
        except KeyboardInterrupt:
            print("\n(interrupted)", file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})


def cmd_profile_save(args: argparse.Namespace) -> int:
    values = {key: getattr(args, key) for key in profiles.FIELDS if getattr(args, key) is not None}
    if "provider" in values:
        values["provider"] = get_provider(values["provider"]).name
    if not values and not args.unset and not args.default:
        raise CLIError("Nothing to save. Pass at least one option, e.g. --provider, --api-key or --model.")
    profiles.save(args.name, values, args.unset or (), make_default=args.default)
    print(f"Profile {args.name!r} saved.")
    return 0


def cmd_profile_use(args: argparse.Namespace) -> int:
    profiles.set_default(args.name)
    print(f"Default profile is now {args.name!r}.")
    return 0


def cmd_profile_list(args: argparse.Namespace) -> int:
    default, items = profiles.list_all()
    if not items:
        print("No profiles yet. Create one with: prompt_llm profile save NAME --provider openai --api-key ...")
        return 0
    width = max(len(name) for name in items)
    for name, values in sorted(items.items()):
        provider = values.get("provider") or "openai"
        model = values.get("model") or f"({PROVIDERS.get(provider, PROVIDERS['openai']).default_model})"
        print(f"{'*' if name == default else ' '} {name:<{width}}  {provider:<10}  {model}")
    return 0


def cmd_profile_show(args: argparse.Namespace) -> int:
    name, values = profiles.get(args.name)
    if name is None:
        raise CLIError("No default profile. Pass a profile name or run `prompt_llm profile use NAME`.")
    if "api_key" in values and not args.show_secrets:
        values["api_key"] = _mask(str(values["api_key"]))
    print(json.dumps({"name": name, **values}, indent=2, sort_keys=True))
    return 0


def cmd_profile_rm(args: argparse.Namespace) -> int:
    profiles.remove(args.name)
    print(f"Profile {args.name!r} removed.")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    aliases = {}
    for alias, name in ALIASES.items():
        aliases.setdefault(name, []).append(alias)
    for p in PROVIDERS.values():
        extra = f"  (alias: {', '.join(aliases[p.name])})" if p.name in aliases else ""
        print(f"{p.name:<11} {p.env_var or '(no key)':<19} {p.default_model}{extra}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


# --- parser ------------------------------------------------------------------

def _request_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("request options")
    g.add_argument("-p", "--profile", help="profile to use (default: the default profile)")
    g.add_argument("-P", "--provider", help="provider to use (see `prompt_llm providers`)")
    g.add_argument("-m", "--model", help="model name")
    g.add_argument("-s", "--system", help="system prompt")
    g.add_argument("-t", "--temperature", type=float, help="sampling temperature")
    g.add_argument("--max-tokens", type=int, help="maximum tokens to generate")
    g.add_argument("-k", "--api-key", help="API key (prefer a profile or env var: this ends up in shell history)")
    g.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"request timeout in seconds (default: {DEFAULT_TIMEOUT:g})")
    s = g.add_mutually_exclusive_group()
    s.add_argument("--stream", dest="stream", action="store_true", default=None, help="stream the response as it is generated")
    s.add_argument("--no-stream", dest="stream", action="store_false", help="wait for the full response")


def _add_ask(sub, name: str, **kwargs) -> argparse.ArgumentParser:
    p = sub.add_parser(name, description="Send a prompt and print the response.", **kwargs)
    p.add_argument("prompt", nargs="*", help="prompt text; '-' inserts stdin (piped input is read when no text is given)")
    p.add_argument("-f", "--file", action="append", default=[], metavar="PATH", help="append a text file to the prompt (repeatable)")
    p.add_argument("--tools", metavar="PATH", help="JSON file with tool definitions; tool calls are printed as JSON")
    p.add_argument("-o", "--out", metavar="PATH", help="write the output to a file (.json files get --json output)")
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="print text, tool calls, stop reason and usage as JSON")
    fmt.add_argument("--raw", action="store_true", help="print the provider's original JSON response")
    _request_options(p)
    p.set_defaults(func=cmd_ask)
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt_llm",
        description="Prompt LLMs from the command line. Without a command, the arguments are a prompt for `ask`.",
        epilog=f"Provider shortcuts: `prompt_llm <provider> PROMPT` for any of: {', '.join(sorted(SHORTCUTS))}.",
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    _add_ask(sub, "ask", help="send a prompt (the default command)")
    for name in sorted(SHORTCUTS):  # hidden from the command list: no help=
        _add_ask(sub, name).set_defaults(provider=name)

    p = sub.add_parser("chat", help="interactive chat with conversation history", description="Interactive chat.")
    _request_options(p)
    p.set_defaults(func=cmd_chat)

    prof = sub.add_parser("profile", help="manage profiles (provider, model, api key, system prompt, temperature)")
    psub = prof.add_subparsers(dest="action", metavar="<action>", required=True)

    p = psub.add_parser("save", help="create or update a profile", description="Create or update a profile. Only the given options change.")
    p.add_argument("name")
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--api-key")
    p.add_argument("--base-url", help="custom endpoint, e.g. a local OpenAI-compatible server")
    p.add_argument("--system", help="system prompt")
    p.add_argument("--temperature", type=float)
    p.add_argument("--unset", nargs="+", choices=profiles.FIELDS, metavar="KEY", help=f"remove keys: {', '.join(profiles.FIELDS)}")
    p.add_argument("--default", action="store_true", help="also make it the default profile")
    p.set_defaults(func=cmd_profile_save)

    p = psub.add_parser("use", aliases=["load"], help="make a profile the default")
    p.add_argument("name")
    p.set_defaults(func=cmd_profile_use)

    psub.add_parser("list", aliases=["ls"], help="list profiles (* marks the default)").set_defaults(func=cmd_profile_list)

    p = psub.add_parser("show", help="show a profile's settings")
    p.add_argument("name", nargs="?", help="profile name (default: the default profile)")
    p.add_argument("--show-secrets", action="store_true", help="print the API key unmasked")
    p.set_defaults(func=cmd_profile_show)

    p = psub.add_parser("rm", help="delete a profile")
    p.add_argument("name")
    p.set_defaults(func=cmd_profile_rm)

    sub.add_parser("providers", help="list providers, API key env vars and default models").set_defaults(func=cmd_providers)
    sub.add_parser("version", help="print the version").set_defaults(func=cmd_version)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        if sys.stdin is not None and not sys.stdin.isatty():
            argv = ["ask"]
    elif argv[0] not in COMMANDS | SHORTCUTS and argv[0] not in ("-h", "--help", "-V", "--version"):
        argv.insert(0, "ask")

    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra:
        # Allow prompt words after options, e.g. `prompt_llm hello -m gpt-5 world`.
        if hasattr(args, "prompt") and not any(e.startswith("-") and e != "-" for e in extra):
            args.prompt += extra
        else:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if not getattr(args, "func", None):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except (CLIError, LLMError, profiles.ProfileError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except BrokenPipeError:
        # Output was piped into something that exited early (e.g. `| head`).
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1


if __name__ == "__main__":
    sys.exit(main())
