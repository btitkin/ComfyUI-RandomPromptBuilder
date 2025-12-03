class PromptDisplayNode:
    """
    Display node that shows the generated prompt text in a large, readable format.
    Use this to preview prompts before they go to CLIP Text Encode.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True}),  # Connect wire here
            },
            "optional": {
                "text": ("STRING", {"multiline": True, "default": "Prompt will appear here..."}),  # Display here
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "display_text"
    CATEGORY = "PromptBuilder"
    OUTPUT_NODE = True
    
    def display_text(self, text_input="", text=""):
        """Display the text and pass it through."""
        # Print to console
        print(f"\n[Prompt Display] {len(text_input)} chars:\n{text_input}\n")
        
        # Update the 'text' widget with the input value
        return {
            "ui": {"text": [text_input]}, 
            "result": (text_input,)
        }
    
    @classmethod
    def IS_CHANGED(cls, text_input, text):
        return float("nan")


# Add to existing NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "PromptDisplayNode": PromptDisplayNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptDisplayNode": "Prompt Display (Preview Text)",
}
