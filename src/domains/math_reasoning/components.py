"""
Prompt components for the math reasoning domain.
"""

from src.models import (
    BaseComponent,
    TextComponent,
    ExampleComponent,
)


MATH_REASONING_COMPONENTS = [
    TextComponent(
        name='Role_Assignment',
        content=(
            'You are an expert mathematician with extensive training'
            ' in mathematical reasoning and problem solving.'
        ),
    ),
    BaseComponent(
        name='Task_Statement',
        content='You are tasked with solving a math problem.',
    ),
    ExampleComponent(
        name='Example_Solution',
        framing_statement=(
            'Here is an example of a solution to a similar problem:'
            '\n\n'
            'Example problem:\n{example_problem}\n\n'
            'Example solution:\n{example_solution}\n\n'
        ),
    ),
    # TextComponent(
    #     name='Tree_of_Thoughts',
    #     content=(
    #         'Consider multiple solution paths before proceeding:\n'
    #         '1. First, identify 2-3 distinct approaches to this problem\n'
    #         '2. For each approach, evaluate its advantages and potential pitfalls\n'
    #         '3. Choose the most elegant and reliable approach\n'
    #         '4. If your chosen approach leads to a dead end, backtrack and try an alternative'
    #     ),
    # ),
    TextComponent(
        name='Chain_of_Thought',
        content=(
            'Solve the given problem step-by-step, showing all your'
            ' work clearly.'
            # 'Break down the problem into these clear steps:\n'
            # '1. Identify the key variables and constraints\n'
            # '2. Choose an appropriate mathematical approach\n'
            # '3. Execute each step with mathematical precision\n'
            # '4. Show all intermediate calculations'
        ),
    ),
    TextComponent(
        name='Self_Verification',
        content=(
            'After finding your answer, check your work to verify it before'
            ' submitting your final answer.'
        ),
    ),
    BaseComponent(
        name='Problem_Statement',
        content='Problem statement:\n\n{problem}',
    ),
    BaseComponent(
        name='Format_Instructions',
        content='Place your final answer in <answer> tags.',
    ),
]
