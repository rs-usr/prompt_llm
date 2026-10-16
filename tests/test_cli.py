import io
import json
import os
import stat
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from prompt_llm import __version__, cli, profiles


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def reply(payload):
    """urlopen side effect returning a fresh JSON response on every call."""
    return lambda *a, **k: FakeResponse(json.dumps(payload).encode())


def openai_reply(text="ok", **message):
    return reply({"model": "m", "choices": [{"message": {"content": text, **message}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 3, "completion_tokens": 5}})


def sse(*events):
    return lambda *a, **k: FakeResponse("".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n")


class CLITestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        for patcher in (mock.patch.object(profiles, "HOME", self.home), mock.patch.dict(os.environ, {}, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cwd = tempfile.TemporaryDirectory()
        self.addCleanup(self.cwd.cleanup)

    def run_cli(self, *argv, stdin="", tty=False):
        out, err = io.StringIO(), io.StringIO()
        out.isatty = lambda: tty
        fake_stdin = io.StringIO(stdin)
        fake_stdin.isatty = lambda: not stdin
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err), mock.patch("sys.stdin", fake_stdin):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def sent(self, urlopen):
        request = urlopen.call_args[0][0]
        return request, json.loads(request.data)


class ProfileTests(CLITestCase):
    def test_save_show_list_use_rm(self):
        self.assertEqual(self.run_cli("profile", "save", "work", "--provider", "claude", "--api-key", "sk-1234567890abcdef")[0], 0)
        self.run_cli("profile", "save", "work", "--temperature", "0.3")  # updates only the given field
        shown = json.loads(self.run_cli("profile", "show")[1])  # first profile became the default
        self.assertEqual(shown, {"name": "work", "provider": "anthropic", "api_key": "sk-1...cdef", "temperature": 0.3})
        self.assertIn("sk-1234567890abcdef", self.run_cli("profile", "show", "work", "--show-secrets")[1])

        self.run_cli("profile", "save", "local", "--provider", "ollama", "--model", "qwen3")
        listing = self.run_cli("profile", "ls")[1].splitlines()
        self.assertTrue(listing[0].startswith("  local") and "qwen3" in listing[0])
        self.assertTrue(listing[1].startswith("* work") and "anthropic" in listing[1])

        self.assertEqual(self.run_cli("profile", "use", "local")[0], 0)
        self.assertEqual(json.loads(self.run_cli("profile", "show")[1])["name"], "local")
        self.assertEqual(self.run_cli("profile", "rm", "local")[0], 0)
        self.assertEqual(self.run_cli("profile", "rm", "local")[0], 1)
        self.assertEqual(self.run_cli("profile", "show")[0], 1)  # default was removed

    def test_unset(self):
        self.run_cli("profile", "save", "p", "--model", "x", "--system", "be brief")
        self.run_cli("profile", "save", "p", "--unset", "system")
        self.assertEqual(json.loads(self.run_cli("profile", "show", "p")[1]), {"name": "p", "model": "x"})

    def test_file_is_private(self):
        self.run_cli("profile", "save", "p", "--api-key", "secret")
        self.assertEqual(stat.S_IMODE(os.stat(self.home / "profiles.json").st_mode), 0o600)

    def test_errors(self):
        self.assertEqual(self.run_cli("profile", "save", "p")[0], 1)  # nothing to save
        self.assertIn("Invalid profile name", self.run_cli("profile", "save", "../x", "--model", "m")[2])
        self.assertIn("Unknown provider", self.run_cli("profile", "save", "p", "--provider", "nope")[2])
        self.assertIn("does not exist", self.run_cli("profile", "use", "missing")[2])
        self.assertIn("No profiles yet", self.run_cli("profile", "list")[1])

    def test_migrates_legacy_config(self):
        (self.home / "config.json").write_text(json.dumps({"api_key": "old-key", "model": "gpt-4o", "junk": 1}))
        (self.home / "groq.json").write_text(json.dumps({"api_key": "g-key"}))
        default, items = profiles.list_all()
        self.assertEqual(default, "default")
        self.assertEqual(items, {"default": {"api_key": "old-key", "model": "gpt-4o"}, "groq": {"api_key": "g-key"}})


class AskTests(CLITestCase):
    @mock.patch("urllib.request.urlopen", side_effect=openai_reply("Hello!"))
    def test_default_command_uses_default_profile(self, urlopen):
        self.run_cli("profile", "save", "me", "--api-key", "sk-test", "--system", "be brief", "--temperature", "0.5")
        code, out, _ = self.run_cli("Say", "hi")
        self.assertEqual((code, out), (0, "Hello!\n"))
        request, body = self.sent(urlopen)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-test")
        self.assertEqual(body["messages"], [{"role": "system", "content": "be brief"}, {"role": "user", "content": "Say hi"}])
        self.assertEqual((body["temperature"], body["model"]), (0.5, "gpt-5-mini"))
        self.assertNotIn("stream", body)

    @mock.patch("urllib.request.urlopen", side_effect=openai_reply())
    def test_prompt_words_after_options(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "k"
        self.run_cli("hello", "-m", "gpt-x", "world", "--max-tokens", "50")
        body = self.sent(urlopen)[1]
        self.assertEqual((body["messages"][-1]["content"], body["model"]), ("hello world", "gpt-x"))
        self.assertEqual(body["max_completion_tokens"], 50)
        self.assertNotIn("system", [m["role"] for m in body["messages"]])

    @mock.patch("urllib.request.urlopen", side_effect=reply({"model": "c", "stop_reason": "end_turn",
                                                              "content": [{"type": "text", "text": "Hi from Claude"}]}))
    def test_anthropic(self, urlopen):
        os.environ["ANTHROPIC_API_KEY"] = "ant-key"
        code, out, _ = self.run_cli("claude", "hello", "-s", "be brief", "-m", "claude-x")
        self.assertEqual((code, out), (0, "Hi from Claude\n"))
        request, body = self.sent(urlopen)
        self.assertEqual(request.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(request.get_header("X-api-key"), "ant-key")
        self.assertEqual((body["system"], body["model"], body["max_tokens"]), ("be brief", "claude-x", 4096))

    @mock.patch("urllib.request.urlopen", side_effect=openai_reply())
    def test_profile_key_does_not_leak_to_other_provider(self, urlopen):
        self.run_cli("profile", "save", "work", "--provider", "openai", "--api-key", "sk-openai", "--model", "gpt-x", "--system", "S")
        os.environ["GROQ_API_KEY"] = "groq-key"
        self.assertEqual(self.run_cli("-P", "groq", "hi")[0], 0)
        request, body = self.sent(urlopen)
        self.assertEqual(request.get_header("Authorization"), "Bearer groq-key")
        self.assertEqual(body["model"], "llama-3.3-70b-versatile")
        self.assertEqual(body["messages"][0]["content"], "S")  # system prompt still applies

    @mock.patch("urllib.request.urlopen", side_effect=openai_reply())
    def test_ollama_and_custom_base_url_need_no_key(self, urlopen):
        self.assertEqual(self.run_cli("ollama", "hi")[0], 0)
        request = self.sent(urlopen)[0]
        self.assertEqual(request.full_url, "http://localhost:11434/v1/chat/completions")
        self.assertIsNone(request.get_header("Authorization"))
        self.run_cli("profile", "save", "lm", "--base-url", "http://127.0.0.1:1234/v1/")
        self.assertEqual(self.run_cli("-p", "lm", "hi")[0], 0)
        self.assertEqual(self.sent(urlopen)[0].full_url, "http://127.0.0.1:1234/v1/chat/completions")

    @mock.patch("urllib.request.urlopen", side_effect=openai_reply())
    def test_stdin_and_files(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "k"
        self.run_cli("Summarize:", "-", stdin="file contents")
        self.assertEqual(self.sent(urlopen)[1]["messages"][-1]["content"], "Summarize: file contents")
        self.run_cli(stdin="only stdin")
        self.assertEqual(self.sent(urlopen)[1]["messages"][-1]["content"], "only stdin")
        path = Path(self.cwd.name) / "a.py"
        path.write_text("print(1)\n")
        self.run_cli("Explain", "-f", str(path))
        self.assertEqual(self.sent(urlopen)[1]["messages"][-1]["content"], f"Explain\n\n{path}:\n```\nprint(1)\n```")
        self.assertIn("Cannot read", self.run_cli("x", "-f", "/nonexistent")[2])

    @mock.patch("urllib.request.urlopen", side_effect=sse({"choices": [{"delta": {"content": "Hel"}}]},
                                                          {"choices": [{"delta": {"content": "lo"}}]}, {"choices": []}))
    def test_streams_on_tty(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "k"
        self.assertEqual(self.run_cli("hi", tty=True)[1], "Hello\n")
        self.assertTrue(self.sent(urlopen)[1]["stream"])

    @mock.patch("urllib.request.urlopen", side_effect=sse({"type": "message_start"},
                                                          {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
                                                          {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "!"}}))
    def test_anthropic_stream(self, urlopen):
        os.environ["ANTHROPIC_API_KEY"] = "k"
        self.assertEqual(self.run_cli("anthropic", "hi", "--stream")[1], "Hi!\n")

    @mock.patch("urllib.request.urlopen", side_effect=sse({"error": {"message": "overloaded"}}))
    def test_stream_error(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "k"
        code, _, err = self.run_cli("hi", "--stream")
        self.assertEqual(code, 1)
        self.assertIn("overloaded", err)

    def test_missing_api_key(self):
        code, _, err = self.run_cli("hi")
        self.assertEqual(code, 1)
        self.assertIn("OPENAI_API_KEY", err)

    def test_empty_prompt(self):
        os.environ["OPENAI_API_KEY"] = "k"
        self.assertIn("Empty prompt", self.run_cli("ask")[2])

    @mock.patch("urllib.request.urlopen")
    def test_http_error(self, urlopen):
        body = io.BytesIO(json.dumps({"error": {"message": "Invalid API key"}}).encode())
        urlopen.side_effect = urllib.error.HTTPError("u", 401, "Unauthorized", {}, body)
        os.environ["OPENAI_API_KEY"] = "bad"
        code, _, err = self.run_cli("hi")
        self.assertEqual((code, err), (1, "Error: HTTP 401: Invalid API key\n"))

    @mock.patch("urllib.request.urlopen", side_effect=lambda *a, **k: FakeResponse(b"<html>bad gateway</html>"))
    def test_invalid_json(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "k"
        self.assertIn("Invalid JSON response", self.run_cli("hi")[2])

    @mock.patch("urllib.request.urlopen", side_effect=reply({"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}))
    def test_truncation_warning(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "k"
        self.assertIn("token limit", self.run_cli("hi")[2])


class OutputTests(CLITestCase):
    def setUp(self):
        super().setUp()
        os.environ["OPENAI_API_KEY"] = "k"

    @mock.patch("urllib.request.urlopen", side_effect=openai_reply("Hello"))
    def test_json_and_raw(self, urlopen):
        result = json.loads(self.run_cli("hi", "--json")[1])
        self.assertEqual(result, {"provider": "openai", "model": "m", "text": "Hello", "tool_calls": [],
                                  "stop_reason": "stop", "usage": {"input_tokens": 3, "output_tokens": 5}})
        raw = json.loads(self.run_cli("hi", "--raw")[1])
        self.assertEqual(raw["choices"][0]["message"]["content"], "Hello")
        with self.assertRaises(SystemExit) as ctx:  # mutually exclusive
            self.run_cli("hi", "--json", "--raw")
        self.assertEqual(ctx.exception.code, 2)

    @mock.patch("urllib.request.urlopen", side_effect=openai_reply("Hello"))
    def test_out_file(self, urlopen):
        out = Path(self.cwd.name) / "result.json"
        code, stdout, err = self.run_cli("hi", "--out", str(out))
        self.assertEqual((code, stdout), (0, ""))
        self.assertIn("Saved to", err)
        self.assertEqual(json.loads(out.read_text())["text"], "Hello")  # .json implies --json
        txt = Path(self.cwd.name) / "result.txt"
        self.run_cli("hi", "-o", str(txt))
        self.assertEqual(txt.read_text(), "Hello\n")
        self.run_cli("hi", "-o", str(out), "--raw")
        self.assertIn("choices", json.loads(out.read_text()))

    def write_tools(self, tools):
        path = Path(self.cwd.name) / "tools.json"
        path.write_text(json.dumps(tools))
        return str(path)

    @mock.patch("urllib.request.urlopen", side_effect=openai_reply(None, tool_calls=[
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]))
    def test_tools_openai(self, urlopen):
        tools = self.write_tools([{"name": "get_weather", "description": "Weather",
                                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}])
        code, out, _ = self.run_cli("Weather in Paris?", "--tools", tools, tty=True)  # tools disable streaming
        self.assertEqual(json.loads(out), [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}])
        body = self.sent(urlopen)[1]
        self.assertEqual(body["tools"][0]["type"], "function")
        self.assertEqual(body["tools"][0]["function"]["name"], "get_weather")
        self.assertNotIn("stream", body)

    @mock.patch("urllib.request.urlopen", side_effect=reply({"stop_reason": "tool_use", "content": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}]}))
    def test_tools_anthropic_accepts_openai_format(self, urlopen):
        os.environ["ANTHROPIC_API_KEY"] = "k"
        tools = self.write_tools({"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]})
        result = json.loads(self.run_cli("anthropic", "Weather?", "--tools", tools, "--json")[1])
        self.assertEqual(result["tool_calls"], [{"id": "tu_1", "name": "get_weather", "arguments": {"city": "Paris"}}])
        self.assertEqual(result["text"], "Checking.")
        self.assertEqual(self.sent(urlopen)[1]["tools"], [{"name": "get_weather", "input_schema": {"type": "object"}}])

    def test_invalid_tools_file(self):
        self.assertIn("every tool needs a 'name'", self.run_cli("hi", "--tools", self.write_tools([{"x": 1}]))[2])
        bad = Path(self.cwd.name) / "bad.json"
        bad.write_text("{nope")
        self.assertIn("Invalid tools file", self.run_cli("hi", "--tools", str(bad))[2])


class ChatTests(CLITestCase):
    @mock.patch("urllib.request.urlopen", side_effect=openai_reply("pong"))
    def test_chat_keeps_history(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "k"
        with mock.patch("builtins.input", side_effect=["ping", "/reset", "again", EOFError]):
            code, out, _ = self.run_cli("chat", "--no-stream")
        self.assertEqual((code, out), (0, "pong\npong\n"))
        self.assertEqual(self.sent(urlopen)[1]["messages"], [{"role": "user", "content": "again"}])
        first_call = json.loads(urlopen.call_args_list[0][0][0].data)
        self.assertEqual(len(first_call["messages"]), 1)


class MiscTests(CLITestCase):
    def test_version_and_providers(self):
        self.assertEqual(self.run_cli("version")[1].strip(), __version__)
        out = self.run_cli("providers")[1]
        self.assertIn("anthropic   ANTHROPIC_API_KEY", out)
        self.assertIn("alias: claude", out)

    def test_help_without_args(self):
        self.assertIn("usage: prompt_llm", self.run_cli()[1])


if __name__ == "__main__":
    unittest.main()
