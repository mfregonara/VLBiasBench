import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import os
import gc

from .base_model import VLM_BaseModel

torch._dynamo.config.disable = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_path = {
    'gemma-3-4b': './models/gemma/checkpoints/gemma-3-4b-it',
    'gemma-3-12b': './models/gemma/checkpoints/gemma-3-12b-it',
}


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class Gemma3(VLM_BaseModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_path = model_path[model_name]
        
        print(f"Loading Gemma3 model from {self.model_path}...")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        self.do_sample = True
        self.temperature = self.config.get('temperature', 0.7)
        self.max_new_tokens = self.config.get('max_new_tokens', 128)
        self.top_p = self.config.get('top_p', 0.9)
        self.top_k = self.config.get('top_k', 50)
        self.num_beams = self.config.get('num_beams', 1)
        self.length_penalty = self.config.get('length_penalty', 1.0)
        
        print(f"Gemma3 model loaded successfully on device: {next(self.model.parameters()).device}")
        print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    def generate(self, instruction, images):

        assert len(images) == 1, f"Gemma3 expects single image, got {len(images)}"
        
        image_path = images[0]
        image = load_image(image_path)
        
        formatted_instruction = self.inst_pre + instruction + self.inst_suff
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": formatted_instruction}
                ]
            }
        ]
        
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"Memory before generation: Allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GB, Reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            
            with torch.no_grad():
                inputs = self.processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True,
                    return_dict=True, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                input_len = inputs["input_ids"].shape[-1]
                
                print(f"Input length: {input_len} tokens")
                print(f"GPU memory after tokenization: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict_in_generate=False,
                )
                
                generation = generation[0][input_len:]
            
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            
            del inputs
            del generation
            torch.cuda.empty_cache()
            gc.collect()
            
            return decoded.strip()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM Error: {e}")
            if 'inputs' in locals():
                del inputs
            if 'generation' in locals():
                del generation
            torch.cuda.empty_cache()
            gc.collect()
            return "Error: Out of GPU memory. Memory fragmentation issue - try restarting the process."
            
        except Exception as e:
            print(f"Error during Gemma3 generation: {e}")
            torch.cuda.empty_cache()
            return f"Error: {str(e)}"


class Gemma3_4B(Gemma3):
    def __init__(self, **kwargs):
        model_name = 'gemma-3-4b'
        super().__init__(model_name=model_name, **kwargs)


class Gemma3_12B(Gemma3):
    def __init__(self, **kwargs):
        model_name = 'gemma-3-12b'
        super().__init__(model_name=model_name, **kwargs)