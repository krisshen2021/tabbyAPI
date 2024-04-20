"""Small replication of AutoTokenizer's chat template system for efficiency"""

import json
import pathlib
from functools import lru_cache
from importlib.metadata import version as package_version
from typing import Optional
from jinja2 import Template, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from loguru import logger
from packaging import version
from pydantic import BaseModel

from common.utils import unwrap


class PromptTemplate(BaseModel):
    """A template for chat completion prompts."""

    name: str
    template: str


class TemplateLoadError(Exception):
    """Raised on prompt template load"""

    pass


def get_prompt_from_template(prompt_template: PromptTemplate, template_vars: dict):
    """Get a prompt from a template and a list of messages."""
    if version.parse(package_version("jinja2")) < version.parse("3.0.0"):
        raise ImportError(
            "Parsing these chat completion messages requires jinja2 3.0.0 "
            f"or greater. Current version: {package_version('jinja2')}\n"
            "Please upgrade jinja by running the following command: "
            "pip install --upgrade jinja2"
        )

    compiled_template = _compile_template(prompt_template.template)
    rendered_template = compiled_template.render(**template_vars)
    template_stop_strings = _get_template_stop_strings(compiled_template, template_vars)

    return rendered_template, template_stop_strings


# Inspired from
# https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1761
# TODO: Migrate to compile when template is loaded (removes the need for an lru_cache)
@lru_cache
def _compile_template(template: str):
    """Compiles a Jinja2 template"""

    # Exception handler
    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception

    jinja_template = jinja_env.from_string(template)
    return jinja_template


# TODO: Migrate to run during template load
def _get_template_stop_strings(prompt_template: Template, template_vars: dict):
    """Appends extra stop strings if present in a chat template."""

    extra_stop_strings = []
    template_module = prompt_template.make_module(template_vars)

    if hasattr(template_module, "stop_strings"):
        if isinstance(template_module.stop_strings, list):
            extra_stop_strings += template_module.stop_strings
        else:
            logger.warning(
                "Skipping append of stopping strings from chat template "
                "because stop_strings isn't a list."
            )

    return extra_stop_strings


def get_all_templates():
    """Fetches all templates from the templates directory"""

    template_directory = pathlib.Path("templates")
    return template_directory.glob("*.jinja")


def find_template_from_model(model_path: pathlib.Path):
    """Find a matching template name from a model path."""
    model_name = model_path.name
    template_files = get_all_templates()

    for filepath in template_files:
        template_name = filepath.stem.lower()

        # Check if the template name is present in the model name
        if template_name in model_name.lower():
            return template_name
        else:
            raise TemplateLoadError("Could not find template from model name.")


def get_template_from_file(prompt_template_name: str):
    """Get a template from a jinja file."""

    template_path = pathlib.Path(f"templates/{prompt_template_name}.jinja")
    if template_path.exists():
        with open(template_path, "r", encoding="utf8") as raw_template:
            return PromptTemplate(
                name=prompt_template_name, template=raw_template.read()
            )
    else:
        # Let the user know if the template file isn't found
        raise TemplateLoadError(
            f'Chat template "{prompt_template_name}" not found in files.'
        )


# Get a template from a JSON file
# Requires a key and template name
def get_template_from_model_json(
    json_path: pathlib.Path, key: str, name: Optional[str] = None
):
    """Get a template from a JSON file. Requires a key and template name"""
    if not json_path.exists():
        raise TemplateLoadError(f'Model JSON path "{json_path}" not found.')

    with open(json_path, "r", encoding="utf8") as config_file:
        model_config = json.load(config_file)
        chat_template = model_config.get(key)

        if not chat_template:
            raise TemplateLoadError(
                "Could not find a value from chat_template key in the passed JSON. "
                "Check the tokenizer config?"
            )

        if isinstance(chat_template, list):
            # Handles the new list style of chat templates
            if name:
                wrapped_template = next(
                    (x for x in chat_template if x.get("name") == name),
                    {},
                )
            else:
                wrapped_template = chat_template[0]
                name = unwrap(wrapped_template.get("name"), "from_tokenizer_config")

            selected_template = wrapped_template.get("template")

            if selected_template:
                return PromptTemplate(name=name, template=selected_template)
            else:
                raise TemplateLoadError(
                    f'Chat template with name "{name}" not found '
                    "in model templates list."
                )
        else:
            # Can safely assume the chat template is the old style
            return PromptTemplate(name="from_tokenizer_config", template=chat_template)
