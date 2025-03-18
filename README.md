# Prompt Component Study

A toolkit for conducting prompt ablation studies to analyze the impact of prompt components on LLM performance across different domains.

## Overview

This project provides a structured framework for:
- Defining modular prompt components for different domains (e.g., math reasoning)
- Systematically generating all combinations of prompt components
- Running experiments to measure LLM performance as a function of the prompt components
- Analyzing results to determine component effectiveness

## Initial Results: Math reasoning with Claude 3.5 Haiku

The initial application of this project studies the effects of certain prompt components on mathematical reasoning tasks using Claude 3.5 Haiku.
Four key prompt components were tested:

1. Role Assignment: Positioning the model as "an expert mathematician"
2. Chain of Thought: Instructions to solve problems step-by-step
3. Self Verification: Prompting the model to check its work before answering
4. Example Solution: Providing a sample problem with its solution

The experiment evaluated all possible combinations of these components across a subset of problems from the MATH500 benchmark.
The impact of each component individually and their interactions were analyzed via regression and ANOVA.
While the overall model explains a small portion of variance (R²=0.008), all components had statistically significant effects, with particularly strong positive interactions between Role Assignment and Chain of Thought components.

### OLS Regression Results

```
==============================================================================
Dep. Variable:                correct   R-squared:                       0.008
Model:                            OLS   Adj. R-squared:                  0.008
Method:                 Least Squares   F-statistic:                     21.88
Date:                Mon, 17 Mar 2025   Prob (F-statistic):           1.36e-60
Time:                        15:15:43   Log-Likelihood:                -28748.
No. Observations:               40000   AIC:                         5.753e+04
Df Residuals:                   39984   BIC:                         5.766e+04
Df Model:                          15                                         
Covariance Type:            nonrobust                                         
=======================================================================================================================================
                                                                          coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------------------------------------------------
Intercept                                                               0.5600      0.010     56.388      0.000       0.541       0.579
Role_Assignment                                                        -0.0400      0.014     -2.848      0.004      -0.068      -0.012
Chain_of_Thought                                                       -0.0400      0.014     -2.848      0.004      -0.068      -0.012
Role_Assignment:Chain_of_Thought                                        0.1600      0.020      8.055      0.000       0.121       0.199
Self_Verification                                                       0.0400      0.014      2.848      0.004       0.012       0.068
Role_Assignment:Self_Verification                                       0.0400      0.020      2.014      0.044       0.001       0.079
Chain_of_Thought:Self_Verification                                     -0.0600      0.020     -3.021      0.003      -0.099      -0.021
Role_Assignment:Chain_of_Thought:Self_Verification                     -0.1200      0.028     -4.272      0.000      -0.175      -0.065
Example_Solution                                                       -0.0400      0.014     -2.848      0.004      -0.068      -0.012
Role_Assignment:Example_Solution                                        0.0200      0.020      1.007      0.314      -0.019       0.059
Chain_of_Thought:Example_Solution                                       0.0200      0.020      1.007      0.314      -0.019       0.059
Role_Assignment:Chain_of_Thought:Example_Solution                      -0.1200      0.028     -4.272      0.000      -0.175      -0.065
Self_Verification:Example_Solution                                     -0.0400      0.020     -2.014      0.044      -0.079      -0.001
Role_Assignment:Self_Verification:Example_Solution                  -3.073e-16      0.028  -1.09e-14      1.000      -0.055       0.055
Chain_of_Thought:Self_Verification:Example_Solution                     0.0200      0.028      0.712      0.476      -0.035       0.075
Role_Assignment:Chain_of_Thought:Self_Verification:Example_Solution     0.1800      0.040      4.531      0.000       0.102       0.258
==============================================================================
Omnibus:                   141244.864   Durbin-Watson:                   0.035
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6455.025
Skew:                          -0.153   Prob(JB):                         0.00
Kurtosis:                       1.056   Cond. No.                         47.0
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### OLS Interpretation

Looking at these regression results:

1. The model is statistically significant (F=21.88, p<0.001) but explains only a small portion of variance (R²=0.008).
2. Key findings:
    - Individually, Role_Assignment, Chain_of_Thought, and Example_Solution each reduce correctness by 4% (all p=0.004)
    - Self_Verification individually increases correctness by 4% (p=0.004)
    - The Role_Assignment + Chain_of_Thought interaction is strongly positive (16%, p<0.001)
    - The 4-way interaction shows a substantial positive effect (18%, p<0.001)
3. Notable interaction patterns:
    - Chain_of_Thought works better with Role_Assignment than alone
    - Self_Verification has negative interactions with Chain_of_Thought
    - Three-way interactions show complex patterns

### ANOVA Results

```
                                                    |        sum_sq  |       df  |             F  |        PR(>F)
=================================================================================================================
Intercept                                           |  7.840000e+02  |      1.0  |  3.179578e+03  |  0.000000e+00   
Role_Assignment                                     |  2.000000e+00  |      1.0  |  8.111167e+00  |  4.401582e-03   
Chain_of_Thought                                    |  2.000000e+00  |      1.0  |  8.111167e+00  |  4.401582e-03   
Role_Assignment:Chain_of_Thought                    |  1.600000e+01  |      1.0  |  6.488934e+01  |  8.140116e-16   
Self_Verification                                   |  2.000000e+00  |      1.0  |  8.111167e+00  |  4.401582e-03   
Role_Assignment:Self_Verification                   |  1.000000e+00  |      1.0  |  4.055584e+00  |  4.403218e-02   
Chain_of_Thought:Self_Verification                  |  2.250000e+00  |      1.0  |  9.125063e+00  |  2.522904e-03   
Role_Assignment:Chain_of_Thought:Self_Verification  |  4.500000e+00  |      1.0  |  1.825013e+01  |  1.941602e-05   
Example_Solution                                    |  2.000000e+00  |      1.0  |  8.111167e+00  |  4.401582e-03   
Role_Assignment:Example_Solution                    |  2.500000e-01  |      1.0  |  1.013896e+00  |  3.139774e-01   
Chain_of_Thought:Example_Solution                   |  2.500000e-01  |      1.0  |  1.013896e+00  |  3.139774e-01   
Role_Assignment:Chain_of_Thought:Example_Solution   |  4.500000e+00  |      1.0  |  1.825013e+01  |  1.941602e-05   
Self_Verification:Example_Solution                  |  1.000000e+00  |      1.0  |  4.055584e+00  |  4.403218e-02   
Role_Assignment:Self_Verification:Example_Solution  |  2.951306e-29  |      1.0  |  1.196927e-28  |  1.000000e+00   
Chain_of_Thought:Self_Verification:Example_Solu...  |  1.250000e-01  |      1.0  |  5.069480e-01  |  4.764672e-01   
Role_Assignment:Chain_of_Thought:Self_Verificat...  |  5.062500e+00  |      1.0  |  2.053139e+01  |  5.883085e-06   
Residual                                            |  9.859000e+03  |  39984.0  |           NaN  |           NaN
```

### Type-III ANOVA Interpretation

1. All main effects are significant (p < 0.005):
    - Each individual prompt component has a statistically significant effect on correctness
2. Most significant interactions:
    - Role_Assignment with Chain_of_Thought (F=64.89, p<0.001) - extremely strong interaction
    - 4-way interaction (F=20.53, p<0.001) - complex interplay between all components
    - Role_Assignment, Chain_of_Thought, and Self_Verification (F=18.25, p<0.001)
    - Role_Assignment, Chain_of_Thought, and Example_Solution (F=18.25, p<0.001)

The strong interactions confirm that certain prompt components work synergistically while some combinations may be counterproductive

## Project Structure

```
src/                     # Main source code
├── apis/                # API clients for different LLM providers 
├── prompts/             # Tools for prompt component manipulation
├── models/              # Data models and configuration classes
├── domains/             # Domain-specific implementations
|    └── math_reasoning/ # Math problem solving domain implementation
├── /analysis            # Analysis tools and visualization
├── /experiments         # Experiment tracking and execution
|
├── /data                # Raw data for different domains
├── /results             # Experiment results
└── /notebooks           # Jupyter notebooks for analysis and visualization
```

## Setup

### Requirements

This project was made with Python 3.13.

You will of course need your own API keys.

### Installation

1. Clone the repository
``` bash
git clone https://github.com/yourusername/prompt-component-study.git
cd prompt-component-study
```

2. Create a virtual environment

``` bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
``` bash
pip install -r requirements.txt
pip install -e .
```

3. Create `.env` file
``` bash
touch .env
```

4. Add your API keys:
``` properties
ANTHROPIC_API_KEY="your-anthropic-key"
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-google-key"
```

## Running Experiments

Example code to run a math reasoning experiment on a sample of the MATH 500 benchmark can be found in `src/domains/math_reasoning/run_experiment.py`.
You can run that file as a script to replicate the results present here.

(Currently, the experiment is configured to call Claude 3.5 Haiku with batch processing. Each run with 4-5 prompt components and 50 math problems costs $2-3.)

### Adding New Domains

Docs to do so coming soon!

## License

This project is licensed under the MIT License.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
