"""
Format prompts and aggregate/evaluate responses
"""

import json
import re
import logging

import pandas as pd

from src.models.domain_models import ExperimentConfig
from src.domains.math_reasoning.loader import (
    load_test_set,
    load_examples,
)
from src.apis.llm_interface import LLMInterface
from src.prompts.create_templates import create_prompt_template_df
from src.domains.math_reasoning.evaluation import grade_answer


log = logging.getLogger(__name__)


def format_and_save_prompts(
    prompt_data: pd.DataFrame,
    test_set: pd.DataFrame,
    examples: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """
    Format prompts and save to file.

    Parameters
    ----------
    prompt_data : pd.DataFrame
        The prompt data to format.
    test_set : pd.DataFrame
        The test set to format.
    examples : pd.DataFrame
        The examples to format.
    output_path : str
        The path to save the formatted prompts.

    Returns
    -------
    pd.DataFrame
        A dataframe with the formatted prompts.
    """
    results = []
    for _, row in prompt_data.iterrows():
        for _, test_question in test_set.iterrows():
            result = row.copy().to_dict()
            template_id = result['template_id']
            question_id = test_question['unique_id']

            # shorten and sanitize request_id
            q_id_number = question_id.split('/')[-1].removesuffix('.json')
            request_id = f'{template_id}_{q_id_number}'
            request_id = re.sub(r'[^a-zA-Z0-9_-]', '_', request_id)

            result.update(
                {
                    'request_id': request_id,
                    'prompt': None,
                    'response': None,
                    'correct': None,
                }
            )
            if row['Example_Solution']:
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
            results.append(result)
    results = pd.DataFrame(results)
    results.to_json(output_path, orient='records', lines=True)
    log.info(f'Saved formatted prompts to {output_path}')
    return results


def get_responses(
    llm_interface: LLMInterface,
    prompts: pd.DataFrame,
    data_dir: str = '',
) -> pd.DataFrame:
    """
    Make a batch request and return the responses.

    Parameters
    ----------
    llm_interface : LLMInterface
        The LLM interface instance to use.
    prompts : pd.DataFrame
        The prompts to make a batch request for.
    data_dir : str, optional
        The directory to save the responses to, by default ''

    Returns
    -------
    pd.DataFrame
        A dataframe with the responses.
    """
    batch_id, response_path = llm_interface.make_batch_request(
        prompts, data_dir
    )
    log.info(f'Batch ID: {batch_id}')

    responses = []
    with open(response_path, 'r') as f:
        for line in f:
            responses_object = json.loads(line)
            responses.append(responses_object)
    responses_df = pd.json_normalize(responses)
    return responses_df


def retrieve_and_save_batch_results(
    llm_interface: LLMInterface, batch_id: str
) -> None:
    """
    Retrieve and save batch results.

    Parameters
    ----------
    llm_interface : LLMInterface
        The LLM interface instance to use.
    batch_id : str
        The batch ID to retrieve and save.
    """
    llm_interface.retrieve_and_save_batch_results(batch_id)


def extract_answer(response_text: str) -> str:
    """
    Extract the answer in the <answer> tags from the response text.

    Parameters
    ----------
    response_text : str
        The response text to extract the answer from.

    Returns
    -------
    str
        The answer in the <answer> tags.
    """
    start_idx = response_text.find('<answer>')
    end_idx = response_text.find('</answer>')
    if start_idx == -1 or end_idx == -1:
        return ''
    return response_text[start_idx + len('<answer>') : end_idx]


def clean_responses(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the responses.

    Parameters
    ----------
    responses_df : pd.DataFrame
        The responses dataframe to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned responses dataframe.
    """
    total_responses = len(responses_df)
    log.info(f'Total responses: {total_responses}')
    hit_end_turn = len(
        responses_df[responses_df['result.message.stop_reason'] == 'end_turn']
    )
    log.info(f'Number of completed responses: {hit_end_turn}')

    avg_tokens_in = responses_df['result.message.usage.input_tokens'].mean()
    log.info(f'Average number of input tokens used: {avg_tokens_in:.2f}')
    avg_tokens_out = responses_df['result.message.usage.output_tokens'].mean()
    log.info(f'Average number of output tokens used: {avg_tokens_out:.2f}')

    responses_df['response_text'] = responses_df[
        'result.message.content'
    ].apply(lambda x: x[0]['text'])
    responses_df['response_answer'] = responses_df['response_text'].apply(
        extract_answer
    )

    return responses_df


def add_solutions_from_test_set(
    responses_df: pd.DataFrame, test_set: pd.DataFrame
) -> pd.DataFrame:
    """
    Add the solutions from the test set to the responses dataframe.

    Parameters
    ----------
    responses_df : pd.DataFrame
        The responses dataframe to add the solutions to.
    test_set : pd.DataFrame
        The test set dataframe to add the solutions from.

    Returns
    -------
    pd.DataFrame
        The responses dataframe with the solutions added.
    """
    test_set['question_number'] = (
        test_set['unique_id']
        .apply(lambda x: x.split('/')[-1].removesuffix('.json'))
        .astype(int)
    )
    responses_df['question_number'] = (
        responses_df['custom_id'].apply(lambda x: x.split('_')[1]).astype(int)
    )
    responses_df['test_set_solution'] = responses_df['question_number'].map(
        test_set.set_index('question_number')['solution']
    )
    responses_df['test_set_answer'] = responses_df['question_number'].map(
        test_set.set_index('question_number')['answer']
    )
    return responses_df


def grade_responses(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grade the responses wrt the test set solutions.

    Parameters
    ----------
    responses_df : pd.DataFrame
        The responses dataframe to grade.

    Returns
    -------
    pd.DataFrame
        The responses dataframe with the grades added.
    """
    responses_df['correct'] = responses_df.apply(
        lambda row: grade_answer(
            row['response_answer'], row['test_set_answer']
        ),
        axis=1,
    ).astype(int)
    return responses_df


def compile_results(
    responses_df: pd.DataFrame,
    templates: pd.DataFrame,
    exog_columns: list[str],
    on_column: str = 'template_id',
) -> pd.DataFrame:
    """
    Merge the responses with the templates and exogenous variables for
     analysis.

    Parameters
    ----------
    responses_df : pd.DataFrame
        The responses dataframe to merge.
    templates : pd.DataFrame
        The templates dataframe to merge.
    exog_columns : list[str]
        The exogenous variables to merge.
    on_column : str, optional
        The column to merge on, by default 'template_id'

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    results_df = responses_df[['custom_id', 'correct']].copy()
    # prompt_templates = load_prompt_templates(prompt_templates_path)
    results_df['template_id'] = results_df['custom_id'].apply(
        lambda x: x.split('_')[0]
    )
    merge_columns = [on_column] + exog_columns
    results_df = results_df.merge(
        templates[merge_columns],
        on=on_column,
        how='left',
    )
    return results_df


def clean_and_aggregate_results(
    responses_df: pd.DataFrame,
    test_set: pd.DataFrame,
    templates: pd.DataFrame,
    save_results_path: str,
    exog_columns: list[str],
    on_column: str = 'template_id',
) -> pd.DataFrame:
    """
    Clean the responses and aggregate the results.

    Parameters
    ----------
    responses_df : pd.DataFrame
        The responses dataframe to clean and aggregate.
    test_set : pd.DataFrame
        The test set dataframe to add the solutions from.
    templates : pd.DataFrame
        The templates dataframe to merge.
    save_results_path : str
        The path to save the aggregated results.
    exog_columns : list[str]
        The exogenous variables to merge.
    on_column : str, optional
        The column to merge on, by default 'template_id'

    Returns
    -------
    pd.DataFrame
        The aggregated results.
    """
    responses_df = clean_responses(responses_df)
    responses_df = add_solutions_from_test_set(responses_df, test_set)
    responses_df = grade_responses(responses_df)

    results_df = compile_results(
        responses_df, templates, exog_columns, on_column
    )

    results_df.to_json(save_results_path, orient='records', lines=True)
    log.info(f'Saved aggregated results to {save_results_path}')

    return results_df


def run_experiment(experiment_config: ExperimentConfig):
    """
    Run the experiment.

    Parameters
    ----------
    experiment_config : ExperimentConfig
        The experiment configuration.
    """
    test_set = load_test_set(experiment_config.test_set_path)
    examples = load_examples(experiment_config.examples_path)

    templates = create_prompt_template_df(experiment_config)

    formatted_prompts_path = experiment_config.formatted_prompts_path
    formatted_prompts_df = format_and_save_prompts(
        templates, test_set, examples, formatted_prompts_path
    )

    llm_interface = LLMInterface(experiment_config.llm_args)
    prompts = formatted_prompts_df[['request_id', 'prompt']]
    responses_df = get_responses(
        llm_interface, prompts, experiment_config.data_dir
    )

    exog_columns = [c.name for c in experiment_config.varied_components]
    results_df = clean_and_aggregate_results(
        responses_df,
        test_set,
        templates,
        experiment_config.results_path,
        exog_columns,
    )

    accuracy = results_df['correct'].mean()
    log.info(f'Overall accuracy: {accuracy:.4f}')

    log.info('Experiment complete.')


if __name__ == '__main__':
    from src.domains.math_reasoning.config import (
        MATH_REASONING_CONFIG as config,
    )

    run_experiment(config)
