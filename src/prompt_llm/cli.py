#!/usr/bin/env python

import typer
import json
import os

from .utils import add_to_config, remove_from_config, load_config, save_config_to, load_config_from, remove_profile, load_profiles, print_response
from .langchain_util import call
from .constants import OPENAI, MISTRALAI, ANTHROPIC, GOOGLE, COHERE, NVIDIA, FIREWORKS, GROQ, TOGETHER

app = typer.Typer()

@app.command()
def version():
    """Retrieve the version"""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    init_file = os.path.join(package_dir, '__init__.py')

    with open(init_file, 'r') as file:
        for line in file:
            if line.startswith('__version__'):
                # Extract version from the line
                version = line.split('=')[-1].strip().strip('"').strip("'")
                print(version)

@app.command()
def config_add(api_key: str='', model: str='', system: str = '', temperature:float = None):
    """To set a config"""
    if api_key:
        add_to_config('api_key', api_key)
    else:
        print('Please specify the API KEY.')

    if not system:
        system = 'You are an assistant helping the users promptly.'

    add_to_config('system', system)

    if model:
        add_to_config('model', model)
    else:
        print('Please specify the model.')
    
    if temperature is not None:
        add_to_config('temperature', temperature)

@app.command()
def config_rm(key: str):
    """To unset a config"""
    if remove_from_config(key):
        print(f"{key} has been removed from current config.")

@app.command()
def config_ls(profile: str = ''):
    """To view config"""
    config = load_config(profile)
    print(json.dumps(config, sort_keys=True, indent=4))

@app.command()
def config_save_to(profile: str=''):
    """To save a config to a profile"""
    if profile:
        if save_config_to(profile):
            print("Config has been saved successfully!")
        else:
            print("Error on saving config!")
    else:
        print("No profile has been provided!")


@app.command()
def config_load_from(profile: str=''):
    """To load a config from a saved profile"""
    if profile:
        if load_config_from(profile):
            print(f"Config has been loaded successfully from {profile}.")
        else:
            print(f"Error on loading config from {profile}.")
    else:
        print("No profile has been provided!")


@app.command()
def profile_ls():
    """To view config"""
    profiles = load_profiles()
    print(json.dumps(profiles, sort_keys=True, indent=4))


@app.command()
def profile_rm(profile: str):
    """To unset a config"""
    if remove_profile(profile):
        print(f"{profile} has been removed from current config.")

@app.command()
async def mistralai(prompt: str):
    """To prompt MISTRALAI"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = MISTRALAI

        if "model" not in app_config:
            app_config["model"] = "mistral-large-latest"

        response = await call(prompt, app_config)
        print_response(response, MISTRALAI)

@app.command()
def openai(prompt: str):
    """To prompt OPENAI"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = OPENAI
        if "model" not in app_config:
            app_config["model"] = "gpt-4o"

        response = call(prompt, app_config)
        print_response(response, OPENAI)

@app.command()
def anthropic(prompt: str):
    """To prompt ANTHROPIC"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = "anthropic"
        if "model" not in app_config:
            app_config["model"] = "claude-3-5-sonnet-latest"

        response = call(prompt, app_config)
        print_response(response, ANTHROPIC)

@app.command()
def google(prompt: str):
    """To prompt GOOGLE"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = GOOGLE
        if "model" not in app_config:
            app_config["model"] = "gemini-1.5-pro-latest"

        response = call(prompt, app_config)
        print_response(response, GOOGLE)

@app.command()
def cohere(prompt: str):
    """To prompt COHERE"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = COHERE
        if "model" not in app_config:
            app_config["model"] = "command-xlarge-nightly"

        response = call(prompt, app_config)
        print_response(response, COHERE)

@app.command()
def nvidia(prompt: str):
    """To prompt NVIDIA"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = NVIDIA
        if "model" not in app_config:
            app_config["model"] = "nemo-gpt-megatron-turing-530b"

        response = call(prompt, app_config)
        print_response(response, NVIDIA)

@app.command()
def fireworks(prompt: str):
    """To prompt FIREWORKS"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = FIREWORKS
        if "model" not in app_config:
            app_config["model"] = ""

        response = call(prompt, app_config)
        print_response(response, FIREWORKS)

@app.command()
def groq(prompt: str):
    """To prompt GROQ"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = GROQ
        if "model" not in app_config:
            app_config["model"] = ""

        response = call(prompt, app_config)
        print_response(response, GROQ)

@app.command()
def together(prompt: str):
    """To prompt TOGETHER"""
    app_config = load_config()

    if prompt:
        app_config["provider"] = TOGETHER
        if "model" not in app_config:
            app_config["model"] = ""

        response = call(prompt, app_config)
        print_response(response, TOGETHER)

if __name__ == "__main__":
    app()
