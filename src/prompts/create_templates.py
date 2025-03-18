"""
Compiles components into prompt templates across all combinations in a given domain.
"""

from itertools import combinations
from typing import Generator
import uuid

import pandas as pd

from src.models import ExperimentConfig, PromptComponent, ComponentForm


def get_all_combinations(
    components: list[PromptComponent],
) -> Generator[list[PromptComponent], None, None]:
    """
    Generates all order-preserving combinations of component ablations

    n.b.: Only non-BASE components are considered for ablation.

    Parameters
    ----------
    components : list[PromptComponent]
        The components to get all combinations of.

    Yields
    ------
    list[PromptComponent]
        A combination of components, including all BASE components, in their original order.
    """
    # Separate BASE components from ablatable components
    base_components_with_index = []
    ablatable_components_with_index = []
    for i, c in enumerate(components):
        if c.form == ComponentForm.BASE:
            base_components_with_index.append((i, c))
        else:
            ablatable_components_with_index.append((i, c))

    # If no ablatable components, just return the base components in order
    if not ablatable_components_with_index:
        yield [
            c
            for _, c in sorted(base_components_with_index, key=lambda x: x[0])
        ]
        return

    # Generate all combinations of ablatable components
    n = len(ablatable_components_with_index)
    for i in range(0, n + 1):
        for combo in combinations(ablatable_components_with_index, i):
            # Combine base components with the selected ablatable components
            all_components_with_index = base_components_with_index + list(
                combo
            )

            # Sort by original index to preserve order and extract just the components
            sorted_combo = [
                c
                for _, c in sorted(
                    all_components_with_index, key=lambda x: x[0]
                )
            ]

            yield sorted_combo


def create_prompt_template_df(config: ExperimentConfig) -> pd.DataFrame:
    """
    Creates a dataframe with the following columns:
    - "template_id": uuid.uuid4()
    - "n_prompt_components_included": int, the number of components included in the prompt
    For each component in the config:
    - <component.name>: bool, a binary flag indicating if the component is used in the prompt
    - "prompt": str

    Parameters
    ----------
    config : ExperimentConfig
        Configuration for the task domain, with components and compile_prompt method.

    Returns
    -------
    pd.DataFrame
        A dataframe with the prompt templates.
    """
    rows = []
    for combo in get_all_combinations(config.components):
        row = {}
        row['template_id'] = str(uuid.uuid4())
        row['n_varied_components'] = config.n_varied_components
        row['n_prompt_components_included'] = len(combo)
        for component in config.components:
            row[component.name] = 1 if component in combo else 0
        row['prompt'] = config.compile_prompt(combo)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    # test create_prompt_template_df and save to JSONL
    from src.domains.math_reasoning.config import (
        MATH_REASONING_CONFIG as config,
    )

    df = create_prompt_template_df(config)

    df.to_json(config.prompt_templates_path, orient='records', lines=True)
