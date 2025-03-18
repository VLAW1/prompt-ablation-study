"""
Models for prompt components and templates.
"""

from enum import StrEnum
from typing import Callable

from pydantic import BaseModel


class ComponentForm(StrEnum):
    """
    The form of a prompt component.

    - BASE: Denotes a component that is ALWAYS present in the prompt.
    - TEXT: Denotes a straightforward text component in the prompt.
    - EXAMPLE: Denotes a component that retrieves an example to compile
        into the prompt.

    Components other than BASE are ablated in the experiment.
    """

    BASE = 'base'
    TEXT = 'text'
    EXAMPLE = 'example'


class Component(BaseModel):
    name: str
    form: ComponentForm


class BaseComponent(Component):
    form: ComponentForm = ComponentForm.BASE
    content: str


class TextComponent(Component):
    form: ComponentForm = ComponentForm.TEXT
    content: str


class ExampleComponent(Component):
    form: ComponentForm = ComponentForm.EXAMPLE
    example_retriever: Callable[[], str] | None = None
    framing_statement: str | None = None


type PromptComponent = BaseComponent | TextComponent | ExampleComponent
