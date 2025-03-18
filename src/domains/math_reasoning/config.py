"""
Config for the math reasoning domain.
"""

import os
from typing import Callable

from src.models import ExperimentConfig
from src.domains.math_reasoning.components import MATH_REASONING_COMPONENTS
from src.domains.math_reasoning.evaluation import grade_answer


MATH_REASONING_CONFIG = ExperimentConfig(
    llm_args={
        'provider': 'anthropic',
        'model': 'claude-3-5-haiku-20241022',
        'api_key': os.getenv('ANTHROPIC_API_KEY'),
        'temperature': 0.3,
        'max_tokens': 4096,
    },
    components=MATH_REASONING_COMPONENTS,
    data_dir='data/math_reasoning',
)


def get_correctness_evaluator() -> Callable[[str, str], bool]:
    """
    Get the correctness evaluator.

    Returns
    -------
    Callable[[str, str], bool]
        The grade_answer function for the MATH500 problems.
    """
    return grade_answer
