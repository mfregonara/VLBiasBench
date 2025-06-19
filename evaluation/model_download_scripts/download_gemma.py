import torch
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def download_gemma3_4b_it():
    model_id = "google/gemma-3-4b-it"
    save_dir = ".././models/gemma/checkpoints/gemma-3-4b-it"

    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading model from {model_id}...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print(f"Model and processor saved to {save_dir}")

if __name__ == "__main__":
    download_gemma3_4b_it()
