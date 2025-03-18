import os
import time

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from src.models.prompt_models import PromptComponent, ComponentForm

load_dotenv()


class ExperimentConfig(BaseSettings):
    llm_args: dict[str, str | int | float] = {
        'provider': 'anthropic',
        'model': 'claude-3-5-haiku-20241022',
        'api_key': os.getenv('ANTHROPIC_API_KEY'),
        'temperature': 0.0,
        'max_tokens': 4096,
    }

    components: list[PromptComponent]
    separator: str = '\n\n'

    data_dir: str

    @property
    def test_set_path(self) -> str:
        return os.path.join(self.data_dir, 'problems.jsonl')

    @property
    def examples_path(self) -> str:
        return os.path.join(self.data_dir, 'examples.jsonl')

    @property
    def prompt_templates_path(self) -> str:
        timestamp = time.strftime('%Y%m%d_%I%M%S')
        return os.path.join(
            self.data_dir, f'prompt_templates_{timestamp}.jsonl'
        )

    @property
    def formatted_prompts_path(self) -> str:
        timestamp = time.strftime('%Y%m%d_%I%M%S')
        return os.path.join(
            self.data_dir, f'formatted_prompts_{timestamp}.jsonl'
        )

    @property
    def experiment_path_suffix(self) -> str:
        model = self.llm_args['model'].replace('-', '_')
        temperature = str(self.llm_args['temperature']).replace('.', '_')
        return f'{model}_{temperature}'

    # @property
    # def responses_path(self) -> str:
    #     timestamp = time.strftime('%Y%m%d_%I%M%S')
    #     return os.path.join(
    #         self.data_dir,
    #         f'responses_{timestamp}_{self.experiment_path_suffix}.jsonl',
    #     )

    @property
    def results_path(self) -> str:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        return os.path.join(
            self.data_dir,
            f'results_{timestamp}_{self.experiment_path_suffix}.jsonl',
        )

    @property
    def paths_dict(self) -> dict[str, str]:
        return {
            'test_set_path': self.test_set_path,
            'examples_path': self.examples_path,
            'prompt_templates_path': self.prompt_templates_path,
            'formatted_prompts_path': self.formatted_prompts_path,
            'results_path': self.results_path,
        }

    @property
    def n_total_components(self) -> int:
        return len(self.components)

    @property
    def n_base_components(self) -> int:
        return len(
            [c for c in self.components if c.form == ComponentForm.BASE]
        )

    @property
    def varied_components(self) -> list[PromptComponent]:
        return [c for c in self.components if c.form != ComponentForm.BASE]

    @property
    def n_varied_components(self) -> int:
        return len(self.varied_components)

    def compile_prompt(
        self, components_to_include: list[PromptComponent]
    ) -> str:
        """
        Compile the prompt template.

        Parameters
        ----------
        components_to_include : list[PromptComponent]
            The components to include in the prompt template.

        Returns
        -------
        str
            The compiled prompt template.
        """
        compiled_components = []
        for component in components_to_include:
            if component.form == ComponentForm.EXAMPLE:
                content = (
                    component.framing_statement
                    if component.example_retriever is None
                    else component.example_retriever()
                )
                compiled_components.append(content)
            else:
                compiled_components.append(component.content)
        return self.separator.join(compiled_components)
