"""
Analysis of prompt ablation experiments.
"""

from typing import Literal

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def load_data(path: str) -> pd.DataFrame:
    """
    Load the data from the given path.
    """
    return pd.read_json(path, lines=True)


def sanitize_column_name(name: str) -> str:
    """
    Sanitize column names to valid formula names by replacing:
    - spaces with underscores
    - special characters with underscores

    Parameters
    ----------
    name : str
        The column name to sanitize

    Returns
    -------
    str
        The sanitized column name
    """
    problem_characters = ' -()'
    for char in problem_characters:
        name = name.replace(char, '_')
    return name


def get_formula(metric_column: str, factor_columns: list[str]) -> str:
    """
    Get the formula for the ANOVA model.
    """
    sanitized_metric = sanitize_column_name(metric_column)
    sanitized_factors = [
        sanitize_column_name(factor) for factor in factor_columns
    ]

    return f'{sanitized_metric} ~ {" * ".join(sanitized_factors)}'


def get_fitted_model(data: pd.DataFrame, formula: str):
    """
    Get the fitted model for the given formula.
    """
    return ols(formula, data=data).fit()


def run_factorial_anova(
    data: pd.DataFrame,
    metric_column: str,
    factor_columns: list[str],
    anova_type: Literal['I', 'II', 'III'] = 'II',
) -> pd.DataFrame:
    """
    Run factorial ANOVA on the given data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to run factorial ANOVA on.
    metric_column : str
        The column to use as the metric.
    factor_columns : list[str]
        The columns to use as the factors.
    anova_type : Literal['I', 'II', 'III'], optional
        The type of ANOVA to run, 'II' by default.

    Returns
    -------
    pd.DataFrame
        The results of the factorial ANOVA.
    """
    all_columns = factor_columns + [metric_column]

    # Get sanitized column names
    sanitized_metric = sanitize_column_name(metric_column)
    sanitized_factors = [
        sanitize_column_name(factor) for factor in factor_columns
    ]
    sanitized_columns = sanitized_factors + [sanitized_metric]
    data_copy = data[all_columns].copy().astype(int)

    # Rename columns to sanitized names
    rename_dict = {i: j for i, j in zip(all_columns, sanitized_columns)}
    data_copy = data_copy.rename(columns=rename_dict)

    # Get the formula
    formula = get_formula(sanitized_metric, sanitized_factors)

    # Get the fitted model
    model = get_fitted_model(data_copy, formula)

    # Get the ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=anova_type)

    return anova_table


def run_factorial_anova_from_model(
    model,
    anova_type: Literal['I', 'II', 'III'] = 'II',
) -> pd.DataFrame:
    """
    Run factorial ANOVA on the given model.

    Parameters
    ----------
    model : sm.RegressionResults
        The model to run factorial ANOVA on.
    anova_type : Literal['I', 'II', 'III'], optional
        The type of ANOVA to run.

    Returns
    -------
    pd.DataFrame
        The results of the factorial ANOVA.
    """
    return sm.stats.anova_lm(model, typ=anova_type)


def calculate_effect_sizes(anova_table):
    """
    Calculate partial eta squared effect sizes for all effects in an ANOVA table.

    Parameters
    ----------
    anova_table : pd.DataFrame
        The ANOVA table from statsmodels with 'sum_sq' column

    Returns
    -------
    pd.DataFrame
        The original ANOVA table with an added 'partial_eta_sq' column
    """
    # Create a copy of the ANOVA table
    result_table = anova_table.copy()

    # Get the residual sum of squares
    ss_residual = anova_table.loc['Residual', 'sum_sq']

    # Calculate partial eta squared for each effect
    result_table['partial_eta_sq'] = float('nan')  # Initialize with NaN

    for idx in result_table.index:
        if idx != 'Residual':
            ss_effect = result_table.loc[idx, 'sum_sq']
            result_table.loc[idx, 'partial_eta_sq'] = ss_effect / (
                ss_effect + ss_residual
            )

    # Add a column for effect size interpretation
    result_table['effect_size'] = float('nan')

    # Classify effect sizes (using common benchmarks for partial eta squared)
    for idx in result_table.index:
        if idx != 'Residual' and not pd.isna(
            result_table.loc[idx, 'partial_eta_sq']
        ):
            eta_sq = result_table.loc[idx, 'partial_eta_sq']
            if eta_sq < 0.01:
                result_table.loc[idx, 'effect_size'] = 'Negligible'
            elif eta_sq < 0.06:
                result_table.loc[idx, 'effect_size'] = 'Small'
            elif eta_sq < 0.14:
                result_table.loc[idx, 'effect_size'] = 'Medium'
            else:
                result_table.loc[idx, 'effect_size'] = 'Large'

    return result_table
