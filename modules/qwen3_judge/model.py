
from PIL import Image
from pathlib import Path
import re

import torch

from transformers import AutoProcessor, AutoModelForMultimodalLM



class Qwen3Judge:
    def __init__(self, model_id: str, cache_dir: str, device: str, dtype: str):
        
        if isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
            
        self.device = device
        
        
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=self.dtype,
            device_map={'': self.device}
        )
        
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, image, image_prompt, explanation_max_words):

        messages = self.make_messages(image, image_prompt, explanation_max_words)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )

        inputs = inputs.to(self.device)


        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=160, 
                do_sample=False
            )


        new_ids = output_ids[:, inputs['input_ids'].shape[1]:]

        answer = self.processor.batch_decode(
            new_ids, 
            skip_special_tokens=True, 
        )[0].strip()

        return self.parse_judge_answer(answer)




    def parse_judge_answer(self, text: str):
        alignment_score = re.search(r'### ALIGNMENT SCORE:\s*([1-5])', text)
        alignment_explanation = re.search(r'### ALIGNMENT EXPLANATION:\s*(.*)', text)
        quality_score = re.search(r'### QUALITY SCORE:\s*([1-5])', text)
        quality_explanation = re.search(r'### QUALITY EXPLANATION:\s*(.*)', text)
        
        return {
            'alignment_score': int(alignment_score.group(1)) if alignment_score else None, 
            'alignment_explanation': alignment_explanation.group(1).strip() if alignment_explanation else None, 
            'quality_score': int(quality_score.group(1)) if quality_score else None, 
            'quality_explanation': quality_explanation.group(1).strip() if quality_explanation else None, 
        }


    def make_messages(self, image, image_prompt, explanation_max_words=30):
        judge_prompt = f'''
You are an assistant evaluating an image on two independent aspects:
(1) how well it aligns with the meaning of a given text prompt, and
(2) its visual quality.

The text prompt is: "{image_prompt}"

---

PART 1: PROMPT ALIGNMENT (Semantic Fidelity)
Evaluate only the meaning conveyed by the image - ignore visual artifacts.
Focus on:
- Are the correct objects present and depicted in a way that clearly demonstrates their intended roles and actions from the prompt?
- Does the scene illustrate the intended situation or use-case in a concrete and functional way, rather than through symbolic, metaphorical, or hybrid representation?
- If the described usage or interaction is missing or unclear, alignment should be penalized.
- Focus strictly on the presence, roles, and relationships of the described elements - not on rendering quality.

Score from 1 to 5:
5: Fully conveys the prompt's meaning with correct elements
4: Mostly accurate - main elements are correct, with minor conceptual or contextual issues
3: Main subjects are present but important attributes or actions are missing or wrong
2: Some relevant components are present, but key elements or intent are significantly misrepresented
1: Does not reflect the prompt at all

---

PART 2: VISUAL QUALITY (Rendering Fidelity)
Now focus only on how the image looks visually - ignore whether it matches the prompt.
Focus on:
- Are there rendering artifacts, distortions, or broken elements?
- Are complex areas like faces, hands, and shaped objects well-formed and visually coherent?
- Are complex areas like faces, hands, limbs, and object grips well-formed and anatomically correct?
- Is lighting, texture, and perspective consistent across the scene?
- Do elements appear physically coherent - i.e., do objects connect naturally (no floating tools, clipped limbs, or merged shapes)?
- Distortion, warping, or implausible blending of objects should reduce the score.
- Unusual or surreal objects are acceptable if they are clearly rendered and visually deliberate.

Score from 1 to 5:
5: Clean, realistic, and fully coherent - no visible flaws
4: Mostly clean with minor visual issues or stiffness
3: Noticeable visual flaws, but the image is still readable
2: Major visual issues - warped or broken key elements disrupt coherence
1: Severe rendering failure - image appears nonsensical or corrupted

---

Keep each explanation to at most {explanation_max_words} words.

Respond using exactly this format:
### ALIGNMENT SCORE: score
### ALIGNMENT EXPLANATION: explanation
### QUALITY SCORE: score
### QUALITY EXPLANATION: explanation
'''.strip()

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': judge_prompt},
                ],
            },
        ]
        return messages
