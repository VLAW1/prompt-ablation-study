"""
Load the test set and prompt data.
"""

import pandas as pd


def load_prompt_templates(path: str) -> pd.DataFrame:
    """
    Load the prompt templates from a JSONL file.

    Parameters
    ----------
    path : str
        The path to the JSONL file containing the prompt data.

    Returns
    -------
    pd.DataFrame
        A dataframe with the prompt data.
    """
    return pd.read_json(path, lines=True)


def load_test_set(path: str) -> pd.DataFrame:
    """
    Load the test set, which is a JSONL file with keys:
        - "problem": str
        - "solution": str
        - "answer": str
        - "subject": str
        - "level": str (of int64 values)
        - "unique_id": str (of form "test/<subject>/<int>.json")

    Parameters
    ----------
    path : str
        The path to the JSONL file containing the test set.

    Returns
    -------
    pd.DataFrame
        A dataframe with the test set.
    """
    return pd.read_json(path, lines=True)


def load_examples(path: str) -> pd.DataFrame:
    """
    Load the examples from a JSONL file.
    """
    return pd.read_json(path, lines=True)
