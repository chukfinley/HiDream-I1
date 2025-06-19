import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from diffusers import HiDreamImagePipeline
import os
from datetime import datetime
import json
from tqdm import tqdm

class HiDreamBatchInference:
    def __init__(self, model_name="HiDream-ai/HiDream-I1-Full", device="cuda"):
        """
        Initialize the HiDream batch inference pipeline
        
        Args:
            model_name: HiDream model variant ("HiDream-ai/HiDream-I1-Full", "HiDream-ai/HiDream-I1-Dev", "HiDream-ai/HiDream-I1-Fast")
            device: Device to run inference on
        """
        print("Loading tokenizer and text encoder...")
        self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.text_encoder_4 = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16,
        )
        
        print(f"Loading HiDream pipeline: {model_name}")
        self.pipe = HiDreamImagePipeline.from_pretrained(
            model_name,
            tokenizer_4=self.tokenizer_4,
            text_encoder_4=self.text_encoder_4,
            torch_dtype=torch.bfloat16,
        )
        
        self.pipe = self.pipe.to(device)
        self.device = device
        self.model_name = model_name
        
        # Set default parameters based on model variant
        if "Dev" in model_name or "Fast" in model_name:
            self.default_guidance_scale = 0.0
            self.default_steps = 16 if "Fast" in model_name else 28
        else:
            self.default_guidance_scale = 5.0
            self.default_steps = 50
            
        print(f"Pipeline loaded successfully! Default settings: guidance_scale={self.default_guidance_scale}, steps={self.default_steps}")
    
    def generate_single(self, prompt, **kwargs):
        """Generate a single image from a prompt"""
        # Use default parameters if not specified
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        guidance_scale = kwargs.get('guidance_scale', self.default_guidance_scale)
        num_inference_steps = kwargs.get('num_inference_steps', self.default_steps)
        seed = kwargs.get('seed', None)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        
        return image
    
    def batch_generate(self, prompts, output_dir="batch_output", **kwargs):
        """
        Generate images for a batch of prompts
        
        Args:
            prompts: List of prompt strings or list of dicts with prompt and parameters
            output_dir: Directory to save generated images
            **kwargs: Default parameters for all prompts (can be overridden per prompt)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metadata file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = os.path.join(output_dir, f"batch_metadata_{timestamp}.json")
        metadata = {
            "model": self.model_name,
            "timestamp": timestamp,
            "default_params": kwargs,
            "results": []
        }
        
        print(f"Starting batch generation of {len(prompts)} images...")
        print(f"Output directory: {output_dir}")
        
        for i, prompt_data in enumerate(tqdm(prompts, desc="Generating images")):
            try:
                # Handle both string prompts and dict prompts
                if isinstance(prompt_data, str):
                    prompt = prompt_data
                    params = kwargs.copy()
                else:
                    prompt = prompt_data.get('prompt', '')
                    params = kwargs.copy()
                    params.update(prompt_data.get('params', {}))
                
                # Generate image
                image = self.generate_single(prompt, **params)
                
                # Save image
                filename = f"image_{i:04d}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                
                # Add to metadata
                result_metadata = {
                    "index": i,
                    "prompt": prompt,
                    "filename": filename,
                    "parameters": params
                }
                metadata["results"].append(result_metadata)
                
                print(f"✓ Generated image {i+1}/{len(prompts)}: {filename}")
                
            except Exception as e:
                print(f"✗ Error generating image {i+1}: {str(e)}")
                metadata["results"].append({
                    "index": i,
                    "prompt": prompt if 'prompt' in locals() else "Unknown",
                    "error": str(e)
                })
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nBatch generation complete!")
        print(f"Metadata saved to: {metadata_file}")
        
        return metadata

def main():
    """Example usage of the batch inference system"""
    
    # Initialize the batch inference system
    batch_system = HiDreamBatchInference(
        model_name="HiDream-ai/HiDream-I1-Full"  # Change to Dev or Fast if needed
    )
    
    # Example 1: Simple list of prompts
    simple_prompts = [
        'A cat holding a sign that says "HiDream.ai"',
        'A futuristic cityscape at sunset with flying cars',
        'A magical forest with glowing mushrooms and fairy lights',
        'A steampunk robot playing chess in a Victorian library',
        'An astronaut riding a horse on Mars with Earth in the background'
    ]
    
    # Example 2: Advanced prompts with individual parameters
    advanced_prompts = [
        {
            'prompt': 'A hyperrealistic portrait of a cyberpunk hacker in neon-lit Tokyo',
            'params': {'seed': 42, 'guidance_scale': 7.0}
        },
        {
            'prompt': 'A serene Japanese garden with cherry blossoms and a traditional bridge',
            'params': {'seed': 123, 'num_inference_steps': 75}
        },
        {
            'prompt': 'A dramatic black and white photograph of a lighthouse in a storm',
            'params': {'seed': 456, 'width': 768, 'height': 1024}
        }
    ]
    
    # Run batch generation with simple prompts
    print("=== Running Simple Batch Generation ===")
    batch_system.batch_generate(
        prompts=simple_prompts,
        output_dir="simple_batch_output",
        seed=0  # Default seed for all images
    )
    
    # Run batch generation with advanced prompts
    print("\n=== Running Advanced Batch Generation ===")
    batch_system.batch_generate(
        prompts=advanced_prompts,
        output_dir="advanced_batch_output"
    )

if __name__ == "__main__":
    main()