import json
import random


random.seed(42)


ALL_MATH_500_SUBJECTS = [
    'Prealgebra',
    'Algebra',
    'Intermediate Algebra',
    'Geometry',
    'Counting & Probability',
    'Precalculus',
    'Number Theory',
]

SUBSET_MATH_500_SUBJECTS = [
    'Prealgebra',
    'Algebra',
    'Intermediate Algebra',
    'Geometry',
    'Counting & Probability',
    # 'Precalculus',
    # 'Number Theory',
]


def write_sample_math_500_data(data_dir: str) -> None:
    """
    Randomly selects 10 problems from each subject in the subset of subjects.

    MATH 500 is currently saved in:
        data/test_sets/math_reasoning/MATH500.jsonl
    Saves the selected problems to:
        data/test_sets/math_reasoning/problems.jsonl
    Saves examples to:
        data/test_sets/math_reasoning/examples.jsonl

    Parameters
    ----------
    data_dir : str
        The directory to read data and save the sample to.
    """
    math_500_path = data_dir + '/MATH500.jsonl'
    out_file = data_dir + '/problems.jsonl'
    examples_file = data_dir + '/examples.jsonl'

    selected_problems = []
    examples = []
    for subject in SUBSET_MATH_500_SUBJECTS:
        with open(math_500_path) as f:
            problems = [
                json.loads(line)
                for line in f
                if json.loads(line)['subject'] == subject
            ]
            random.shuffle(problems)
            selected_problems.extend(problems[:10])
            examples.extend(problems[-1:])

    with open(out_file, 'w') as f:
        for problem in selected_problems:
            f.write(json.dumps(problem) + '\n')

    with open(examples_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    data_dir = 'data/math_reasoning'
    write_sample_math_500_data(data_dir)
