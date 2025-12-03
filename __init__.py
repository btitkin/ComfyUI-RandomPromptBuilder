import os

# Path to JavaScript files directory
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

from .random_prompt_builder import RandomPromptBuilderNode
from .prompt_display import PromptDisplayNode

NODE_CLASS_MAPPINGS = {
    "RandomPromptBuilder": RandomPromptBuilderNode,
    "PromptDisplayNode": PromptDisplayNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomPromptBuilder": "Random Prompt Builder (GGUF)",
    "PromptDisplayNode": "Prompt Display (Preview Text)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']


