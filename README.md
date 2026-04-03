# prompt_llm

## Version
### 0.1.4

Prompting major LLMs via CLI.

**Supported Models**
- OpenAI
- MistralAI
- Anthropic
- Google
- Cohere
- Nvidia
- Fireworks
- Groq
- Together

## Installation

### 1. Clone the repo

### 2. Install
- **Local:**
    - `pip install --editable .`
- **PyPi:**
    - `pip install prompt_llm`

### 3. Install auto-completion (Optional):
- `prompt_llm --install-completion`
- `source ~/.bashrc`

## Config:
**Supported configs:**

- `--api-key`
- `--model`
- `--system`
- `--temperature`

### Set Config
`prompt_llm config_add  --api-key "<api_key>" --system "<system_messgae>" --temperature <temperature>`

### Save current config to a profile
- `prompt_llm config_save_to --profile "openai"`

### Load the config from a profile
- `prompt_llm config_load_from --profile "openai"`

### Remove Config key:
- `prompt_llm config-rm temperature`

### View Config:
- `prompt_llm config-ls`
- `prompt_llm config-ls --profile="openai"`

## Test CLI
- `prompt_llm openai "Tell me something funny!"`


## Publish

### 1. Build project

`python -m build`

### 2. Upload

`python3 -m twine upload --repository pypi dist/* --verbose`
