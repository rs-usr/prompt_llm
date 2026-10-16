# prompt_llm

## Version
### 0.2.0

Prompt LLMs from the command line. **Zero dependencies**: pure Python standard library, about 40 KB installed.
Every provider is included in the base install, so there are no extras to pick.

- 13 providers, including local models via Ollama or any OpenAI-compatible server
- Profiles for API key, model, system prompt and temperature
- Streaming, interactive chat, stdin and file input
- Tool calling, JSON output, raw provider responses, saving results to a file

## Installation

```sh
pipx install prompt_llm      # or: pip install prompt_llm
pip install --editable .     # from a clone
```

Requires Python 3.10+.

## Quick start

```sh
export OPENAI_API_KEY=sk-...                  # or save the key in a profile (below)
prompt_llm "Tell me something funny"          # `ask` is the default command
prompt_llm claude "Explain monads briefly"    # provider shortcut
prompt_llm chat                               # interactive chat with history
```

## Prompting

```sh
prompt_llm Explain monads in one sentence           # quotes are optional
prompt_llm -m gpt-5 -t 0.2 "Name a color"           # per-call overrides
git diff | prompt_llm                               # piped input becomes the prompt
git diff | prompt_llm "Write a commit message:" -   # '-' inserts stdin into the prompt
prompt_llm "Review this code" -f app.py -f util.py  # attach files
prompt_llm -p work "Hello"                          # use a specific profile
prompt_llm ask "profile of a cat"                   # use `ask` when the prompt starts with a command word
```

| Option | Description |
|---|---|
| `-p, --profile NAME` | profile to use (default: the default profile) |
| `-P, --provider NAME` | provider to use; see `prompt_llm providers` |
| `-m, --model NAME` | model |
| `-s, --system TEXT` | system prompt |
| `-t, --temperature N` | sampling temperature |
| `--max-tokens N` | maximum tokens to generate |
| `-k, --api-key KEY` | API key (a profile or env var is safer, since this ends up in shell history) |
| `-f, --file PATH` | append a text file to the prompt (repeatable) |
| `--tools PATH` | JSON file with tool definitions |
| `-o, --out PATH` | write the output to a file instead of stdout |
| `--json` | output text, tool calls, stop reason and token usage as JSON |
| `--raw` | output the provider's original JSON response |
| `--stream` / `--no-stream` | streaming is on by default when printing text to a terminal |
| `--timeout SECONDS` | request timeout (default: 120) |

Settings are resolved in this order: command line option, then profile, then environment variable, then provider default.

### Output and saving results

```sh
prompt_llm "Hi" --json                 # {"provider", "model", "text", "tool_calls", "stop_reason", "usage"}
prompt_llm "Hi" --raw                  # exactly what the provider returned
prompt_llm "Hi" --out answer.txt       # text
prompt_llm "Hi" --out result.json      # a .json file gets the --json format
prompt_llm "Hi" --out raw.json --raw   # the raw response
```

### Tool calling

Pass a JSON file with tool definitions. The model's tool calls are printed as JSON
(`id`, `name`, `arguments`) so you can run them yourself or pipe them into another program.

```sh
prompt_llm "What's the weather in Paris?" --tools examples/tools.json
# [{"id": "call_abc", "name": "get_weather", "arguments": {"city": "Paris"}}]
```

The file holds a list of tools (or `{"tools": [...]}`). Each tool can be written in plain,
OpenAI or Anthropic format, and is converted for whichever provider you call:

```json
[{"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
```

`{"type": "function", "function": {...}}` (OpenAI) and `"input_schema"` (Anthropic) also work.

### Chat

```sh
prompt_llm chat -p work
```

Keeps the conversation history. `/reset` clears it; `/exit` or Ctrl-D quits.

## Profiles

A profile is a small set of settings: `provider`, `model`, `api_key`, `system`, `temperature`
and, optionally, `base_url`. Profiles are stored in `~/.prompt_llm/profiles.json` with `600`
permissions (override the directory with `PROMPT_LLM_HOME`).

```sh
prompt_llm profile save work --provider openai --api-key sk-... --model gpt-5-mini
prompt_llm profile save claude --provider anthropic --api-key sk-ant-... --system "Be concise"
prompt_llm profile save work --temperature 0.3      # update one setting
prompt_llm profile save work --unset temperature    # remove a setting
prompt_llm profile use claude                       # make it the default (alias: load)
prompt_llm profile list                             # * marks the default
prompt_llm profile show work                        # API key is masked (--show-secrets)
prompt_llm profile rm work
```

The first profile you save becomes the default. A profile's `api_key`, `model` and `base_url`
are only used with that profile's provider, so a key is never sent to another provider
(`prompt_llm -p work -P groq ...` uses `GROQ_API_KEY`).

### Local and custom endpoints

```sh
prompt_llm ollama "Hi"                              # Ollama on localhost:11434, no key needed
prompt_llm profile save lmstudio --provider openai --base-url http://localhost:1234/v1 --model qwen3
```

## Providers

| Provider | Shortcut / aliases | API key env var | Default model |
|---|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-5-mini` |
| Anthropic | `anthropic`, `claude` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| Google Gemini | `google`, `gemini` | `GOOGLE_API_KEY` | `gemini-2.5-flash` |
| Mistral AI | `mistralai`, `mistral` | `MISTRAL_API_KEY` | `mistral-large-latest` |
| Cohere | `cohere` | `COHERE_API_KEY` | `command-a-03-2025` |
| Groq | `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| xAI | `xai`, `grok` | `XAI_API_KEY` | `grok-4` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `openrouter/auto` |
| Together AI | `together` | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Fireworks AI | `fireworks` | `FIREWORKS_API_KEY` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY` | `meta/llama-3.3-70b-instruct` |
| Ollama (local) | `ollama` | none | `llama3.2` |

## Upgrading from 0.1.x

- `config-*` commands are replaced by `profile` commands. Your old `~/.prompt_llm/config.json`
  is read as the `default` profile and every `<name>.json` as a profile with that name.
  Old files that don't name a provider are treated as OpenAI; add one with `profile save NAME --provider ...`.
- No system prompt is sent unless you set one.
- Shell completion (`--install-completion`) was removed together with the Typer dependency.

## Development

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

## Publish

```sh
python -m build
python3 -m twine upload --repository pypi dist/* --verbose
```
