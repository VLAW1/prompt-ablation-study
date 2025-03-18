"""
Mock results for the math reasoning domain.
"""

import random

import pandas as pd

from src.domains.math_reasoning.loader import (
    load_prompt_templates,
    load_test_set,
    load_examples,
)


def create_mock_response(
    prompt_data_row: pd.Series,
    test_question: dict[str, str],
) -> str:
    """
    Create a mock response for a given prompt template and test question.

    This returns the solution with probability:
        n_prompt_components_included / n_total_varied_prompt_components

    Parameters
    ----------
    prompt_data_row : pd.Series
        The prompt template to use for the mock result.
    test_question : dict[str, str]
        The test question to use for the mock result.

    Returns
    -------
    str
        The mock response.
    """
    n_varied_components = prompt_data_row['n_varied_components']
    n_prompt_components_included = prompt_data_row[
        'n_prompt_components_included'
    ]

    p_correct = n_prompt_components_included / n_varied_components

    if random.random() < p_correct:
        return test_question['answer']
    else:
        return 'idk'


def create_and_save_mock_results(
    prompt_data: pd.DataFrame,
    test_set: pd.DataFrame,
    examples: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """
    Create and save mock results for a given prompt data and test set.

    Parameters
    ----------
    prompt_data : pd.DataFrame
        The prompt data to use for the mock results.
    test_set : pd.DataFrame
        The test set to use for the mock results.
    examples : pd.DataFrame
        The examples to use for the mock results.
    output_path : str
        The path to save the mock results.

    Returns
    -------
    pd.DataFrame
        A dataframe with the mock results.
    """
    mock_results = []
    for _, row in prompt_data.iterrows():
        for _, test_question in test_set.iterrows():
            result = row.copy().to_dict()
            result['question_id'] = test_question['unique_id']
            if row['Example Solution']:
                example = examples[
                    examples['subject'] == test_question['subject']
                ][['problem', 'solution']].to_dict(orient='records')[0]
                prompt = row['prompt'].format(
                    example_problem=example['problem'],
                    example_solution=example['solution'],
                    problem=test_question['problem'],
                )
            else:
                prompt = row['prompt'].format(
                    problem=test_question['problem'],
                )
            result['prompt'] = prompt
            result['response'] = create_mock_response(row, test_question)
            result['correct'] = result['response'] == test_question['answer']
            mock_results.append(result)
    results = pd.DataFrame(mock_results)
    results.to_json(output_path, orient='records', lines=True)
    return results


if __name__ == '__main__':
    test_set_path = 'data/math_reasoning/problems.jsonl'
    prompt_data_path = 'math_reasoning_prompt_templates.jsonl'
    examples_path = 'data/math_reasoning/examples.jsonl'
    output_path = 'math_reasoning_mock_results.jsonl'

    prompt_data = load_prompt_templates(prompt_data_path)
    test_set = load_test_set(test_set_path)
    examples = load_examples(examples_path)
    print(examples.head())

    create_and_save_mock_results(prompt_data, test_set, examples, output_path)
