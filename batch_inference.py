import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
import os
from datetime import datetime
import json
from tqdm import tqdm
import gc

# Import HiDreamImagePipeline from the latest diffusers
from diffusers import HiDreamImagePipeline
print("✓ Imported HiDreamImagePipeline from diffusers")

class HiDreamBatchInference:
    def __init__(self, model_name="HiDream-ai/HiDream-I1-Fast", device="cuda"):
        """
        Initialize the HiDream batch inference pipeline with memory optimization
        
        Args:
            model_name: HiDream model variant ("HiDream-ai/HiDream-I1-Full", "HiDream-ai/HiDream-I1-Dev", "HiDream-ai/HiDream-I1-Fast")
            device: Device to run inference on
        """
        self.device = device
        self.model_name = model_name
        
        # Set memory optimization environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("Loading tokenizer and text encoder with memory optimization...")
        
        # Load tokenizer first (lightweight)
        self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        
        # Load text encoder with the correct parameters
        print("Loading LLaMA text encoder...")
        self.text_encoder_4 = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatically distribute across available GPU memory
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        )
        
        # Clear cache after text encoder loading
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Loading HiDream pipeline: {model_name}")
        # Load pipeline using HiDreamImagePipeline
        self.pipe = HiDreamImagePipeline.from_pretrained(
            model_name,
            tokenizer_4=self.tokenizer_4,
            text_encoder_4=self.text_encoder_4,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        
        # Move pipeline to device with proper error handling
        try:
            self.pipe = self.pipe.to(device)
        except torch.cuda.OutOfMemoryError:
            print("Warning: Could not move entire pipeline to GPU. Using CPU offloading.")
            # Enable CPU offloading for memory-intensive operations
            self.pipe.enable_model_cpu_offload()
        
        # Enable memory efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ Enabled xFormers memory efficient attention")
        except:
            try:
                self.pipe.enable_attention_slicing()
                print("✓ Enabled attention slicing for memory efficiency")
            except:
                print("! Could not enable memory optimizations")
        
        # Set default parameters based on model variant
        if "Dev" in model_name or "Fast" in model_name:
            self.default_guidance_scale = 0.0
            self.default_steps = 16 if "Fast" in model_name else 28
        else:
            self.default_guidance_scale = 5.0
            self.default_steps = 50
            
        # Final memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
        print(f"Pipeline loaded successfully! Default settings: guidance_scale={self.default_guidance_scale}, steps={self.default_steps}")
    
    def generate_single(self, prompt, **kwargs):
        """Generate a single image from a prompt with memory management"""
        # Clear cache before generation
        torch.cuda.empty_cache()
        
        # Use default parameters if not specified
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        guidance_scale = kwargs.get('guidance_scale', self.default_guidance_scale)
        num_inference_steps = kwargs.get('num_inference_steps', self.default_steps)
        seed = kwargs.get('seed', None)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        
        # Generate with memory optimization
        with torch.cuda.device(self.device):
            image = self.pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
        
        # Clear cache after generation
        torch.cuda.empty_cache()
        
        return image
    
    def load_prompts_from_file(self, file_path):
        """
        Load prompts from a text file
        
        Args:
            file_path: Path to text file containing prompts
            
        Returns:
            List of prompts or prompt dictionaries
            
        File formats supported:
        1. Simple text file (one prompt per line)
        2. JSON file with prompt objects
        3. CSV file with prompt column
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.txt':
            # Simple text file - one prompt per line
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
            print(f"✓ Loaded {len(prompts)} prompts from text file")
            return prompts
            
        elif file_ext == '.json':
            # JSON file with prompt objects
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict) and 'prompts' in data:
                prompts = data['prompts']
            else:
                raise ValueError("JSON file must contain a list of prompts or have a 'prompts' key")
            
            print(f"✓ Loaded {len(prompts)} prompts from JSON file")
            return prompts
            
        elif file_ext == '.csv':
            # CSV file with prompt column
            import csv
            prompts = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Look for common prompt column names
                    prompt_col = None
                    for col in ['prompt', 'prompts', 'text', 'description']:
                        if col in row:
                            prompt_col = col
                            break
                    
                    if prompt_col:
                        prompt_data = {'prompt': row[prompt_col].strip()}
                        
                        # Add other parameters if they exist
                        params = {}
                        for key, value in row.items():
                            if key != prompt_col and value.strip():
                                try:
                                    # Try to convert to appropriate type
                                    if key in ['seed', 'num_inference_steps', 'height', 'width']:
                                        params[key] = int(value)
                                    elif key in ['guidance_scale']:
                                        params[key] = float(value)
                                    else:
                                        params[key] = value
                                except ValueError:
                                    params[key] = value
                        
                        if params:
                            prompt_data['params'] = params
                        
                        prompts.append(prompt_data)
            
            if not prompts:
                raise ValueError("No prompts found in CSV file. Make sure you have a 'prompt' column.")
            
            print(f"✓ Loaded {len(prompts)} prompts from CSV file")
            return prompts
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: .txt, .json, .csv")

    def batch_generate_from_file(self, file_path, output_dir=None, **kwargs):
        """
        Generate images from prompts in a file
        
        Args:
            file_path: Path to file containing prompts
            output_dir: Directory to save images (auto-generated if None)
            **kwargs: Default parameters for all prompts
        """
        # Load prompts from file
        prompts = self.load_prompts_from_file(file_path)
        
        # Auto-generate output directory if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = f"batch_output_{file_name}_{timestamp}"
        
    def batch_generate(self, prompts, output_dir="batch_output", **kwargs):
        """
        Generate images for a batch of prompts with memory management
        
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
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
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
                
                # Generate image with memory monitoring
                memory_before = torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0
                
                image = self.generate_single(prompt, **params)
                
                memory_after = torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0
                
                # Save image
                filename = f"image_{i:04d}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                
                # Add to metadata
                result_metadata = {
                    "index": i,
                    "prompt": prompt,
                    "filename": filename,
                    "parameters": params,
                    "memory_usage_gb": {
                        "before": memory_before,
                        "after": memory_after,
                        "peak": memory_after
                    }
                }
                metadata["results"].append(result_metadata)
                
                print(f"✓ Generated image {i+1}/{len(prompts)}: {filename} (GPU: {memory_after:.1f}GB)")
                
                # Aggressive cleanup between generations
                del image
                gc.collect()
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"✗ CUDA OOM error on image {i+1}. Attempting recovery...")
                # Aggressive cleanup
                torch.cuda.empty_cache()
                gc.collect()
                
                # Try with reduced resolution if OOM
                if 'params' in locals() and isinstance(params, dict):
                    params['height'] = min(params.get('height', 1024), 768)
                    params['width'] = min(params.get('width', 1024), 768)
                    try:
                        image = self.generate_single(prompt, **params)
                        filename = f"image_{i:04d}_{timestamp}_lowres.png"
                        filepath = os.path.join(output_dir, filename)
                        image.save(filepath)
                        print(f"✓ Generated image {i+1}/{len(prompts)} at reduced resolution: {filename}")
                        
                        metadata["results"].append({
                            "index": i,
                            "prompt": prompt,
                            "filename": filename,
                            "parameters": params,
                            "note": "Generated at reduced resolution due to memory constraints"
                        })
                        del image
                    except Exception as retry_error:
                        print(f"✗ Failed even with reduced resolution: {str(retry_error)}")
                        metadata["results"].append({
                            "index": i,
                            "prompt": prompt if 'prompt' in locals() else "Unknown",
                            "error": f"CUDA OOM - {str(e)}"
                        })
                else:
                    metadata["results"].append({
                        "index": i,
                        "prompt": prompt if 'prompt' in locals() else "Unknown",
                        "error": f"CUDA OOM - {str(e)}"
                    })
                
                gc.collect()
                torch.cuda.empty_cache()
                
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
    
    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'text_encoder_4'):
            del self.text_encoder_4
        if hasattr(self, 'tokenizer_4'):
            del self.tokenizer_4
        
        gc.collect()
        torch.cuda.empty_cache()
        print("✓ Memory cleaned up")

def main():
    """Example usage of the batch inference system"""
    
    batch_system = None
    try:
        # Initialize the batch inference system
        batch_system = HiDreamBatchInference(
            model_name="HiDream-ai/HiDream-I1-Fast"  # Using Fast model for quicker generation
        )
        
        # Example 1: Generate from text file
        print("=== Example: Generate from text file ===")
        print("Create a file called 'prompts.txt' with one prompt per line, for example:")
        print("A cat holding a sign that says \"HiDream.ai\"")
        print("A futuristic cityscape at sunset with flying cars")
        print("A magical forest with glowing mushrooms")
        print()
        
        # Check if prompts.txt exists
        if os.path.exists("prompts.txt"):
            print("Found prompts.txt file, processing...")
            batch_system.batch_generate_from_file("prompts.txt")
        else:
            # Create example prompts.txt
            example_prompts = [
                'A cat holding a sign that says "HiDream.ai"',
                'A futuristic cityscape at sunset with flying cars',
                'A magical forest with glowing mushrooms and fairy lights'
            ]
            
            with open("prompts.txt", "w") as f:
                for prompt in example_prompts:
                    f.write(prompt + "\n")
            
            print("Created example prompts.txt file")
            batch_system.batch_generate_from_file("prompts.txt")
        
        # Example 2: Generate from CSV file with parameters
        print("\n=== Example: Generate from CSV file with parameters ===")
        csv_data = [
            ["prompt", "seed", "guidance_scale", "height", "width"],
            ["A hyperrealistic portrait of a cyberpunk hacker", "42", "", "1024", "1024"],
            ["A serene Japanese garden with cherry blossoms", "123", "", "768", "1024"],
            ["A dramatic lighthouse in a storm", "456", "", "1024", "768"]
        ]
        
        if not os.path.exists("prompts.csv"):
            import csv
            with open("prompts.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            print("Created example prompts.csv file")
        
        batch_system.batch_generate_from_file("prompts.csv")
        
        # Example 3: Generate from JSON file
        print("\n=== Example: Generate from JSON file ===")
        json_data = {
            "prompts": [
                {
                    "prompt": "An astronaut riding a horse on Mars with Earth in the background",
                    "params": {"seed": 789, "height": 1024, "width": 1024}
                },
                {
                    "prompt": "A steampunk robot playing chess in a Victorian library",
                    "params": {"seed": 101112, "num_inference_steps": 20}
                }
            ]
        }
        
        if not os.path.exists("prompts.json"):
            with open("prompts.json", "w") as f:
                json.dump(json_data, f, indent=2)
            print("Created example prompts.json file")
        
        batch_system.batch_generate_from_file("prompts.json")
        
        print("\n=== All examples completed! ===")
        print("File formats supported:")
        print("1. TXT: One prompt per line")
        print("2. CSV: Columns for prompt, seed, guidance_scale, height, width, etc.")
        print("3. JSON: List of prompt objects with parameters")
        
    except Exception as e:
        print(f"Error during batch generation: {str(e)}")
    finally:
        # Always clean up
        if batch_system:
            batch_system.cleanup()

def cli_main():
    """Command line interface for the batch inference system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HiDream Batch Inference")
    parser.add_argument("input_file", help="Path to input file (txt, csv, or json)")
    parser.add_argument("-o", "--output", help="Output directory", default=None)
    parser.add_argument("-m", "--model", help="Model to use", 
                       choices=["HiDream-ai/HiDream-I1-Fast", "HiDream-ai/HiDream-I1-Dev", "HiDream-ai/HiDream-I1-Full"],
                       default="HiDream-ai/HiDream-I1-Fast")
    parser.add_argument("--seed", type=int, help="Default seed", default=None)
    parser.add_argument("--height", type=int, help="Default height", default=1024)
    parser.add_argument("--width", type=int, help="Default width", default=1024)
    parser.add_argument("--steps", type=int, help="Default inference steps", default=None)
    parser.add_argument("--guidance", type=float, help="Default guidance scale", default=None)
    
    args = parser.parse_args()
    
    # Build kwargs from command line arguments
    kwargs = {}
    if args.seed is not None:
        kwargs['seed'] = args.seed
    if args.height != 1024:
        kwargs['height'] = args.height
    if args.width != 1024:
        kwargs['width'] = args.width
    if args.steps is not None:
        kwargs['num_inference_steps'] = args.steps
    if args.guidance is not None:
        kwargs['guidance_scale'] = args.guidance
    
    batch_system = None
    try:
        print(f"Loading model: {args.model}")
        batch_system = HiDreamBatchInference(model_name=args.model)
        
        print(f"Processing file: {args.input_file}")
        batch_system.batch_generate_from_file(args.input_file, args.output, **kwargs)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if batch_system:
            batch_system.cleanup()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Command line mode
        cli_main()
    else:
        # Example mode
        main()