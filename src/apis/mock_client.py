"""
Mock model client for faking responses during testing.
"""

from src.models import ComponentForm, ExperimentConfig


class MockClient:
    """
    Mock implementation of model interface for testing without API calls.

    When 'queries', this returns the correct answer with probability:
        n_prompt_components_included / n_total_varied_prompt_components
    """

    def __init__(self, all_components: ExperimentConfig):
        """
        Initialize the mock client.

        Parameters
        ----------
        domain_components : ExperimentConfig
            The domain prompt components to use for the mock client.
        """
        self.all_components = all_components
        n_all_components = len(all_components.components)
        n_base_components = len(
            [
                component
                for component in all_components.components
                if component.form == ComponentForm.BASE
            ]
        )
        self.n_varied_components = n_all_components - n_base_components

    def query(self, prompt: str) -> str:
        """
        Query the mock client.

        Parameters
        ----------
        prompt : str
            The prompt to query.

        Returns
        -------
        str
            The response to the prompt.
        """
        # determine the number of varied components in the prompt
        n_varied_components = 0
        for c in self.all_components.components:
            if c.form != ComponentForm.BASE and c.content in prompt:
                n_varied_components += 1

        # determine the probability of a correct response
        p_correct = n_varied_components / self.n_varied_components
