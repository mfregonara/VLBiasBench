from transformers import Blip2ForConditionalGeneration, Blip2Processor
import torch
import os

def download_blip2_opt_2_7b():
    model_id = "Salesforce/blip2-opt-2.7b"
    save_dir = ".././models/LAVIS/checkpoints/blip2-opt-2.7b"

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Download model and processor
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    processor = Blip2Processor.from_pretrained(model_id)

    # Save to local dir
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print(f"Model and processor saved to {save_dir}")

if __name__ == "__main__":
    download_blip2_opt_2_7b()