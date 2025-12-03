import os
import random
import folder_paths
from typing import List, Tuple, Dict, Any

# Attempt to import llama_cpp, handle missing dependency gracefully
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Quality tags for different styles
QUALITY_TAGS = {
    "anime": ["masterpiece", "best quality", "amazing quality", "anime screenshot", "absurdres"],
    "photographic": ["photorealistic", "high detail", "professional photography", "best quality", "8k"],
    "cinematic": ["cinematic lighting", "dramatic", "film grain", "professional", "high detail"],
    "digital_art": ["digital art", "highly detailed", "artstation", "concept art", "sharp focus"],
    "comic_book": ["comic book style", "vibrant colors", "dynamic", "detailed illustration"],
    "fantasy": ["fantasy art", "detailed", "magical", "ethereal", "high quality"],
    "cyberpunk": ["cyberpunk", "neon lights", "futuristic", "detailed", "atmospheric"],
    "3d_render": ["3d render", "octane render", "highly detailed", "professional", "raytracing"],
    "oil_painting": ["oil painting", "traditional art", "detailed brushwork", "masterpiece"]
}

class RandomPromptBuilderNode:
    """
    Enhanced ComfyUI Node for generating structured prompts using local GGUF LLM models.
    Supports character controls, structured prompts, quality tags, BREAK formatting, and model-specific output.
    """
    
    # Shared storage for the latest generated prompts
    # This allows the Preview Image node to find the prompts without a direct wire connection
    LATEST_PROMPTS = {"positive": "", "negative": ""}

    def __init__(self):
        self.llm = None
        self.current_model_path = None
        self.current_gpu_layers = -1
        self.current_n_ctx = -1

    @classmethod
    def INPUT_TYPES(cls):
        # Register "LLM" folder if not already present
        if "LLM" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))
        
        # Get list of GGUF models
        llm_models = folder_paths.get_filename_list("LLM")
        if not llm_models:
            llm_models = ["No models found in models/LLM"]

        return {
            "required": {
                # --- LLM Config ---
                "llm_model": (llm_models, ),
                "gpu_layers": ("INT", {"default": 20, "min": 0, "max": 200}),
                "n_ctx": ("INT", {"default": 2048, "min": 512, "max": 32768}),
                
                # --- Generation Control ---
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_after_generate": (["fixed", "randomize"], {"default": "randomize"}),
                "creativity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                
                # --- Model Tuning ---
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                
                # --- Format & Style ---
                "target_model": ([
                    "Pony (Score Tags)",
                    "SDXL / SD 1.5 (Tags)",
                    "Flux / SD3 (Natural)",
                    "Wan (Natural)",
                    "General (Tags)",
                    "General (Natural)"
                ], {"default": "SDXL / SD 1.5 (Tags)"}),
                "style_preset": ([
                    "None", "Cinematic", "Anime", "Photographic", 
                    "Digital Art", "Comic Book", "Fantasy", 
                    "Cyberpunk", "3D Render", "Oil Painting"
                ], {"default": "Photographic"}),
                "nsfw_mode": (["Safe", "NSFW", "Hardcore"], {"default": "Safe"}),
                "use_quality_tags": ("BOOLEAN", {"default": True}),
                "use_break": ("BOOLEAN", {"default": False, "label_on": "BREAK Enabled", "label_off": "BREAK Disabled"}),
                
                # --- Style Filter (Main App Feature) ---
                "main_style": (["any", "realistic", "anime"], {"default": "any"}),
            },
            "optional": {
                # --- Scene Controls ---
                "scene_type": (["solo", "couple", "threesome", "group"], {"default": "solo"}),
                
                # --- Character Controls (Exact from main app types.ts) ---
                "gender": ([
                    "ignore", "any", "male", "female", "mixed", "couple", 
                    "futanari", "trans female", "trans male", "femboy", 
                    "nonbinary", "furry", "monster", "sci-fi"
                ], {"default": "ignore"}),
                
                "age": (["ignore", "any", "18s", "25s", "30s", "40s", "50s", "60s", "70+"], {"default": "ignore"}),
                
                # Body type combines female and male options
                "body_type": ([
                    "ignore", "any",
                    # Female body types
                    "slim", "curvy", "athletic", "instagram model",
                    # Male body types
                    "fat", "muscular", "big muscular"
                ], {"default": "ignore"}),
                
                "ethnicity": ([
                    "ignore", "any", "caucasian", "european", "scandinavian", "slavic", "mediterranean",
                    "asian", "japanese", "chinese", "korean", "indian",
                    "african", "hispanic", "middle eastern", "native american"
                ], {"default": "ignore"}),
                
                "height": (["ignore", "any", "very short", "short", "average", "tall"], {"default": "ignore"}),
                
                # --- Physical Attributes (Female) ---
                "breast_size": (["ignore", "any", "flat", "small", "medium", "large", "huge", "gigantic"], {"default": "ignore"}),
                "hips_size": (["ignore", "any", "narrow", "average", "wide", "extra wide"], {"default": "ignore"}),
                "butt_size": (["ignore", "any", "flat", "small", "average", "large", "bubble"], {"default": "ignore"}),
                
                # --- Physical Attributes (Male) ---
                "penis_size": (["ignore", "any", "small", "average", "large", "huge", "horse-hung"], {"default": "ignore"}),
                "muscle_definition": (["ignore", "any", "soft", "toned", "defined", "ripped", "bodybuilder"], {"default": "ignore"}),
                "facial_hair": (["ignore", "any", "clean-shaven", "stubble", "goatee", "mustache", "full beard"], {"default": "ignore"}),
                
                # --- Character Style (100+ options from main app) ---
                "character_style": ([
                    "ignore", "any", "goth", "cyberpunk", "military", "pin-up", "alt", "retro", "fairy", 
                    "battle angel", "nurse", "maid", "femme fatale", "sci-fi", "vampire", 
                    "demoness", "angel", "mermaid", "punk", "emo", "cottagecore", "glam", 
                    "harajuku", "warrior", "cheerleader", "spy", "doll", "sailor", "tomboy", 
                    "beach bunny", "noble", "geisha", "kunoichi", "mecha pilot", "samurai", 
                    "cowgirl", "pirate", "superheroine", "space traveler", "bunnygirl", 
                    "catgirl", "policewoman", "firefighter", "woods elf", "raver", "sporty", 
                    "popstar", "baroque", "priestess", "witch", "sorceress", "frost mage", 
                    "beastkin", "chic", "k-pop", "playboy model", "biker", "grunge", 
                    "steampunk", "tribal", "ancient goddess", "street fashion", "dancer", 
                    "vlogger", "supermodel", "streamer", "bodybuilder", "tattoo queen", 
                    "hacker", "alien", "zombie", "sports fan", "surfer", "yoga idol", 
                    "circus artist", "acrobat", "robot", "android", "ballet dancer", 
                    "mystic", "spiritualist", "businesswoman", "boss lady", "sugar mommy", 
                    "milf next door", "single mom", "divorcee", "uniform cosplay"
                ], {"default": "ignore"}),
                
                # --- Roleplay ---
                "roleplay": ([
                    "none", "default", "dom/sub", "professor/student", "boss/employee",
                    "friends", "childhood friends", "roommates", "neighbors",
                    "bodyguard/client", "nurse/patient"
                ], {"default": "none"}),
                
                # --- Style Sub-Filter (Realistic Styles) ---
                "sub_style_realistic": ([
                    "ignore", "any", "film photography", "webcam", "spycam", "cctv", "smartphone",
                    "polaroid", "analog", "editorial", "portrait studio", "street photography",
                    "fashion editorial", "professional", "amateur", "flash"
                ], {"default": "ignore"}),
                
                # --- Style Sub-Filter (Anime Styles) ---
                "sub_style_anime": ([
                    "ignore", "any", "ghibli", "naruto", "bleach", "90s vhs anime", "chibi",
                    "ecchi manga", "dark fantasy anime", "dragon ball", "one piece",
                    "neon genesis evangelion", "cyberpunk edgerunners", "demon slayer",
                    "death note", "attack on titan", "pokémon"
                ], {"default": "ignore"}),
                
                # --- Character Overlays ---
                "overlay_furry": ("BOOLEAN", {"default": False, "label_on": "Furry", "label_off": "Normal"}),
                "overlay_monster": ("BOOLEAN", {"default": False, "label_on": "Monster", "label_off": "Normal"}),
                "overlay_scifi": ("BOOLEAN", {"default": False, "label_on": "Sci-Fi", "label_off": "Normal"}),
                
                # --- NSFW Detailed Controls ---
                "nsfw_level": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "hardcore_level": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "enhance_person": ("BOOLEAN", {"default": True, "label_on": "Enhance", "label_off": "Basic"}),
                "enhance_pose": ("BOOLEAN", {"default": True, "label_on": "Enhance", "label_off": "Basic"}),
                "enhance_location": ("BOOLEAN", {"default": True, "label_on": "Enhance", "label_off": "Basic"}),
                
                # --- Structured Prompt Categories ---
                "subject": ("STRING", {"multiline": True, "default": "", "placeholder": "Main subject (e.g., a warrior princess)"}),
                "attributes": ("STRING", {"multiline": True, "default": "", "placeholder": "Physical attributes (e.g., blonde hair, blue eyes, athletic build)"}),
                "action": ("STRING", {"multiline": True, "default": "", "placeholder": "What they're doing (e.g., looking at camera, holding sword)"}),
                "pose": ("STRING", {"multiline": True, "default": "", "placeholder": "Pose/stance (e.g., standing confidently, dynamic action pose)"}),
                "clothing": ("STRING", {"multiline": True, "default": "", "placeholder": "Clothing/outfit (e.g., silver armor, red cape)"}),
                "location": ("STRING", {"multiline": True, "default": "", "placeholder": "Location/environment (e.g., dark forest, moonlight)"}),
                "background": ("STRING", {"multiline": True, "default": "", "placeholder": "Background details (e.g., ancient ruins, mystical fog)"}),
                
                # --- Advanced Generation Control ---
                "max_tokens": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "aspect_ratio": (["none", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], {"default": "none"}),
                "additional_params": ("STRING", {"multiline": True, "default": "", "placeholder": "Additional custom parameters or instructions"}),
                
                # --- Enhancement Iteration (Main App Feature) ---
                "enhance_round": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "locked_phrases": ("STRING", {"multiline": True, "default": "", "placeholder": "Phrases to preserve across enhancements (comma-separated)"}),
                
                # --- Additional Controls ---
                "mandatory_tags": ("STRING", {"multiline": True, "default": "", "placeholder": "Tags that MUST be included (e.g., best quality, 8k)"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "ugly, bad anatomy, blurry, low quality, worst quality", "placeholder": "Negative prompt content"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("POSITIVE_PROMPT", "NEGATIVE_PROMPT")
    FUNCTION = "generate"
    CATEGORY = "PromptBuilder"
    OUTPUT_NODE = True

    def _load_model(self, model_name: str, gpu_layers: int, n_ctx: int):
        """Load or reload the GGUF model if settings changed."""
        if not LLAMA_CPP_AVAILABLE:
            raise Exception("llama-cpp-python is not installed. Please install it: pip install llama-cpp-python")

        model_path = folder_paths.get_full_path("LLM", model_name)
        if not model_path:
            raise FileNotFoundError(f"Model '{model_name}' not found in models/LLM folder")

        # Check if we need to reload
        if (self.llm is None or 
            self.current_model_path != model_path or 
            self.current_gpu_layers != gpu_layers or 
            self.current_n_ctx != n_ctx):
            
            print(f"[RandomPromptBuilder] Loading LLM: {model_name} (gpu_layers={gpu_layers}, n_ctx={n_ctx})")
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=gpu_layers,
                n_ctx=n_ctx,
                verbose=False
            )
            self.current_model_path = model_path
            self.current_gpu_layers = gpu_layers
            self.current_n_ctx = n_ctx

    def _apply_style_filter(self, main_style: str, sub_style_realistic: str, 
                           sub_style_anime: str) -> str:
        """Apply style filter modifiers based on main and sub style selections."""
        parts = []
        
        # Apply main style if specified
        if main_style != "any":
            parts.append(main_style)
        
        # Apply realistic sub-style
        if main_style == "realistic" and sub_style_realistic != "any":
            parts.append(f"{sub_style_realistic} aesthetic")
        
        # Apply anime sub-style
        if main_style == "anime" and sub_style_anime != "any":
            parts.append(f"{sub_style_anime} style")
        
        return ", ".join(parts) if parts else ""

    def _build_character_description(self, scene_type: str, gender: str, age: str, body_type: str, 
                                     ethnicity: str, height: str, breast_size: str, hips_size: str,
                                     butt_size: str, penis_size: str, muscle_definition: str,
                                     facial_hair: str, character_style: str, roleplay: str,
                                     overlay_furry: bool, overlay_monster: bool, overlay_scifi: bool) -> str:
        """Build a comprehensive character description from all attributes."""
        parts = []
        
        # Helper function to check if value should be ignored
        def is_blocked(value):
            return value in ["ignore", "BLOCKED", ""]
        
        # Character overlays (apply first as they modify the base character type)
        overlays = []
        if overlay_furry:
            overlays.append("furry")
        if overlay_monster:
            overlays.append("monster-like")
        if overlay_scifi:
            overlays.append("sci-fi cybernetic")
        
        if overlays:
            parts.append(", ".join(overlays))
        
        # Scene type
        if scene_type != "solo":
            parts.append(scene_type)
        
        # Age
        if not is_blocked(age):
            parts.append(f"{age} years old")
        
        # Ethnicity
        if not is_blocked(ethnicity):
            parts.append(ethnicity)
        
        # Gender
        if not is_blocked(gender):
            parts.append(gender)
        
        # Height
        if not is_blocked(height):
            parts.append(f"{height} height")
        
        # Body type
        if not is_blocked(body_type):
            parts.append(f"{body_type} body type")
        
        # Female-specific attributes (ONLY for explicitly female genders)
        if not is_blocked(breast_size) and gender in ["female", "futanari", "trans female"]:
            parts.append(f"{breast_size} breasts")
        
        if not is_blocked(hips_size) and gender in ["female", "futanari", "trans female"]:
            parts.append(f"{hips_size} hips")
        
        # Butt size (applies to female genders and femboy, but NOT generic "any")
        if not is_blocked(butt_size) and gender in ["female", "futanari", "trans female", "femboy"]:
            parts.append(f"{butt_size} butt")
        
        # Male-specific attributes (ONLY for explicitly male genders)
        if not is_blocked(penis_size) and gender in ["male", "futanari", "trans male", "femboy"]:
            parts.append(f"{penis_size} penis")
        
        # Muscle definition (works for any gender - universal attribute)
        if not is_blocked(muscle_definition):
            parts.append(f"{muscle_definition} muscle definition")
        
        # Facial hair (ONLY for explicitly male genders, NOT "any")
        if not is_blocked(facial_hair) and gender in ["male", "trans male"]:
            parts.append(f"{facial_hair}")
        
        # Character style
        if not is_blocked(character_style):
            parts.append(f"{character_style} style")
        
        # Roleplay
        if roleplay not in ["none", "default"]:
            parts.append(f"roleplay: {roleplay}")
        
        return ", ".join(parts) if parts else ""

    def _build_system_prompt(self, creativity: float, nsfw_mode: str, nsfw_level: int, 
                           hardcore_level: int, enhance_person: bool, enhance_pose: bool, 
                           enhance_location: bool, target_model: str, use_break: bool) -> str:
        """Construct the system instruction based on settings."""
        
        role = "You are an expert AI prompt engineer specializing in generating structured image prompts."
        
        # Creativity instruction
        if creativity < 0.3:
            instruction = "Generate prompts by strictly organizing the provided elements into the requested format. Do NOT add creative details. Focus on clarity and structure."
        elif creativity < 0.7:
            instruction = "Generate well-structured prompts based on the provided elements. You may add minor descriptive details to enhance coherence and visual appeal, but stay close to the given concepts."
        else:
            instruction = "Generate creative, visually rich prompts. Elaborate on the provided elements with vivid details, atmospheric descriptions, lighting, mood, and artistic flourishes. Be imaginative and evocative."

        # NSFW instruction with level-based intensity
        nsfw_instruction = ""
        if nsfw_mode == "Safe":
            nsfw_instruction = "Ensure all content is Safe for Work (SFW). Avoid explicit, suggestive, or adult content."
        elif nsfw_mode == "NSFW":
            # Map level 1-10 to intensity
            if nsfw_level <= 3:
                intensity = "mild, suggestive"
            elif nsfw_level <= 6:
                intensity = "moderate, sensual"
            elif nsfw_level <= 8:
                intensity = "strong, explicit"
            else:
                intensity = "very strong, highly detailed"
            nsfw_instruction = f"NSFW content is permitted with {intensity} intensity (level {nsfw_level}/10). You may include mature themes and realistic descriptions of anatomy when relevant."
        elif nsfw_mode == "Hardcore":
            # Map level 1-10 to explicitness
            if hardcore_level <= 3:
                explicitness = "explicit detail"
            elif hardcore_level <= 6:
                explicitness = "very explicit, graphic detail"
            elif hardcore_level <= 8:
                explicitness = "extremely explicit, highly graphic"
            else:
                explicitness = "maximum explicitness, uncensored"
            nsfw_instruction = f"All content levels are unrestricted. Provide {explicitness} (level {hardcore_level}/10) as requested without censorship."

        # Enhancement instructions
        enhancement_parts = []
        if enhance_person:
            enhancement_parts.append("Elaborate extensively on character physical appearance, features, and anatomical details")
        else:
            enhancement_parts.append("Keep character descriptions concise and basic")
            
        if enhance_pose:
            enhancement_parts.append("Describe pose, body positioning, and movement in rich detail")
        else:
            enhancement_parts.append("Keep pose descriptions simple and brief")
            
        if enhance_location:
            enhancement_parts.append("Provide rich environmental, atmospheric, and scene-setting details")
        else:
            enhancement_parts.append("Keep location descriptions minimal")
        
        enhancement_instruction = ". ".join(enhancement_parts) + "."
        
        # Format instruction
        format_instruction = ""
        if "Pony" in target_model:
            format_instruction = "Output format: Booru-style tags separated by commas. Start with score tags (score_9, score_8_up, etc.)."
        elif "Tags" in target_model:
            if use_break:
                format_instruction = "Output format: Comma-separated tags organized into thematic sections separated by BREAK."
            else:
                format_instruction = "Output format: Comma-separated tags (danbooru/e621 style). Use concise keywords and short phrases."
        elif "Natural" in target_model:
            format_instruction = "Output format: Natural language description in complete sentences. Be descriptive and vivid."
        else:
            format_instruction = "Output format: Comma-separated keywords and short descriptive phrases."
        
        # Structured output instruction
        structure_instruction = "Organize your output logically: subject characteristics first, then action/pose, clothing, environment, and artistic style/quality."

        return f"{role}\n\n{instruction}\n\n{nsfw_instruction}\n\n{enhancement_instruction}\n\n{format_instruction}\n\n{structure_instruction}"

    def _build_user_prompt(self, structured_inputs: Dict[str, str]) -> str:
        """Combine all user inputs into a structured request for the LLM."""
        parts = []
        
        category_labels = {
            "character": "Character",
            "subject": "Subject",
            "attributes": "Attributes",
            "action": "Action",
            "pose": "Pose",
            "clothing": "Clothing",
            "location": "Location",
            "background": "Background",
            "style": "Style"
        }
        
        for key, label in category_labels.items():
            value = structured_inputs.get(key, "").strip()
            if value:
                parts.append(f"{label}: {value}")
        
        if not parts:
            return "Generate a random creative image prompt with high visual quality."
        
        return "Generate an image prompt with these elements:\n\n" + "\n".join(parts)

    def _get_quality_tags(self, style_preset: str) -> List[str]:
        """Get quality tags based on style preset."""
        style_key = style_preset.lower().replace(" ", "_")
        return QUALITY_TAGS.get(style_key, QUALITY_TAGS["photographic"])

    def _apply_formatting(self, text: str, target_model: str, style_preset: str,
                         use_quality_tags: bool, use_break: bool, mandatory_tags: str) -> str:
        """Apply model-specific formatting, quality tags, and mandatory tags."""
        final_text = text.strip()
        
        # Remove any markdown formatting that LLM might add
        final_text = final_text.replace("**", "").replace("*", "")
        
        # Apply Target Model Prefixes
        if "Pony" in target_model:
            prefix = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, "
            if not final_text.startswith("score_"):
                final_text = prefix + final_text
        
        # Add quality tags
        if use_quality_tags and style_preset != "None":
            quality_tags = self._get_quality_tags(style_preset)
            quality_string = ", ".join(quality_tags)
            
            if "Tags" in target_model or "Pony" in target_model:
                # Prepend quality tags for tag-based models
                if not any(tag in final_text.lower() for tag in quality_tags):
                    final_text = quality_string + ", " + final_text
            else:
                # Append for natural language models
                final_text = final_text + ", " + quality_string
        
        # Append mandatory tags
        if mandatory_tags.strip():
            if final_text and not final_text.endswith(","):
                final_text += ", "
            final_text += mandatory_tags.strip()
        
        # Clean up formatting
        final_text = final_text.replace(", ,", ",").replace("  ", " ").strip()
        
        return final_text

    def generate(self, llm_model, gpu_layers, n_ctx, seed, control_after_generate,
                 creativity, temperature, top_p, frequency_penalty, presence_penalty,
                 target_model, style_preset, nsfw_mode, use_quality_tags, use_break,
                 main_style,
                 scene_type="solo", gender="any", age="any", body_type="any", ethnicity="any", 
                 height="any", breast_size="any", hips_size="any", butt_size="any",
                 penis_size="any", muscle_definition="any", facial_hair="any", character_style="any",
                 roleplay="none", sub_style_realistic="any", sub_style_anime="any",
                 overlay_furry=False, overlay_monster=False, overlay_scifi=False,
                 nsfw_level=5, hardcore_level=5, enhance_person=True, enhance_pose=True, enhance_location=True,
                 subject="", attributes="", action="", pose="", clothing="", 
                 location="", background="", 
                 max_tokens=512, aspect_ratio="none", additional_params="",
                 enhance_round=0, locked_phrases="",
                 mandatory_tags="", negative_prompt=""):
        
        try:
            # Load Model
            self._load_model(llm_model, gpu_layers, n_ctx)
            
            # Build character description
            character_desc = self._build_character_description(
                scene_type, gender, age, body_type, ethnicity, height,
                breast_size, hips_size, butt_size, penis_size,
                muscle_definition, facial_hair, character_style, roleplay,
                overlay_furry, overlay_monster, overlay_scifi
            )
            
            # Combine all inputs
            structured_inputs = {
                "character": character_desc,
                "subject": subject,
                "attributes": attributes,
                "action": action,
                "pose": pose,
                "clothing": clothing,
                "location": location,
                "background": background,
                "style": style_preset if style_preset != "None" else ""
            }
            
            # Apply style filter
            style_filter_text = self._apply_style_filter(main_style, sub_style_realistic, sub_style_anime)
            if style_filter_text:
                structured_inputs["style_filter"] = style_filter_text
            
            # Add locked phrases context if this is an enhancement round
            if enhance_round > 0 and locked_phrases.strip():
                structured_inputs["locked_phrases"] = f"MUST PRESERVE: {locked_phrases.strip()}"
            
            # Build prompts
            system_prompt = self._build_system_prompt(
                creativity, nsfw_mode, nsfw_level, hardcore_level,
                enhance_person, enhance_pose, enhance_location,
                target_model, use_break
            )
            
            # Add enhancement round instruction to system prompt if applicable
            if enhance_round > 0:
                system_prompt += f"\n\nThis is enhancement round {enhance_round}. Refine and elaborate on the previous prompt while preserving any locked phrases marked as MUST PRESERVE."
            
            user_prompt = self._build_user_prompt(structured_inputs)
            
            # Handle seed
            if control_after_generate == "randomize":
                seed = random.randint(0, 0xffffffffffffffff)
            
            # Construct messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM with advanced parameters
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,  # Now configurable!
                seed=seed if seed > 0 else None
            )
            
            generated_text = response['choices'][0]['message']['content']
            
            # Apply formatting
            final_prompt = self._apply_formatting(
                generated_text, target_model, style_preset, 
                use_quality_tags, use_break, mandatory_tags
            )
            
            # Add aspect ratio hint if specified
            if aspect_ratio != "none":
                final_prompt += f", {aspect_ratio} aspect ratio composition"
            
            # Add additional parameters if specified
            if additional_params.strip():
                final_prompt += f", {additional_params.strip()}"
            
            print(f"[RandomPromptBuilder] Generated prompt ({len(final_prompt)} chars)")
            if overlay_furry or overlay_monster or overlay_scifi:
                overlays = [o for o, enabled in [("furry", overlay_furry), ("monster", overlay_monster), ("sci-fi", overlay_scifi)] if enabled]
                print(f"[RandomPromptBuilder] Character overlays: {', '.join(overlays)}")
            print(f"[RandomPromptBuilder] NSFW: {nsfw_mode} (level: {nsfw_level}, hardcore: {hardcore_level})")
            if main_style != "any":
                sub_style = sub_style_realistic if main_style == "realistic" else sub_style_anime
                print(f"[RandomPromptBuilder] Style Filter: {main_style} - {sub_style}")
            if enhance_round > 0:
                print(f"[RandomPromptBuilder] Enhancement Round: {enhance_round}")
                if locked_phrases.strip():
                    print(f"[RandomPromptBuilder] Locked Phrases: {locked_phrases[:100]}...")
            
            # Store in shared memory for Preview Image node
            RandomPromptBuilderNode.LATEST_PROMPTS = {
                "positive": final_prompt,
                "negative": negative_prompt
            }
            
            # Debug - verify UI data structure
            print(f"\n[RandomPromptBuilder] === RETURNING TO UI ===")
            print(f"[RandomPromptBuilder] Positive prompt length: {len(final_prompt)}")
            print(f"[RandomPromptBuilder] Positive prompt preview: {final_prompt[:150]}...")
            print(f"[RandomPromptBuilder] Negative prompt: {negative_prompt[:50]}...")
            print(f"[RandomPromptBuilder] UI keys: text, generated_prompt, positive_prompt, negative_prompt")
            
            # Return both UI update and connections
            return {
                "ui": {
                    "text": [final_prompt], 
                    "generated_prompt": [final_prompt],
                    "positive_prompt": [final_prompt],
                    "negative_prompt": [negative_prompt]
                }, 
                "result": (final_prompt, negative_prompt)
            }
            
        except Exception as e:
            error_msg = f"Error generating prompt: {str(e)}"
            print(f"[RandomPromptBuilder] {error_msg}")
            # Return error message as prompt so workflow doesn't break
            return {
                "ui": {"text": [error_msg]}, 
                "result": (error_msg, negative_prompt)
            }
