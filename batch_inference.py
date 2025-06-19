#!/usr/bin/env python3
"""
Batch inference script for HiDream-I1 that processes prompts from a text file.
Usage: python batch_inference.py --prompts_file prompts.txt --model_type full
"""

import argparse
import os
import time
from pathlib import Path
import torch
from datetime import datetime
import json

# Import the original inference components
from inference import load_model, generate_image


def load_prompts_from_file(file_path):
    """Load prompts from a text file, one prompt per line."""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    prompts.append({
                        'prompt': line,
                        'line_number': line_num
                    })
        print(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def sanitize_filename(prompt, max_length=100):
    """Convert prompt to a safe filename."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        prompt = prompt.replace(char, '_')
    
    # Truncate if too long
    if len(prompt) > max_length:
        prompt = prompt[:max_length] + "..."
    
    return prompt


def batch_generate_images(prompts, model_components, args):
    """Generate images for all prompts in batch."""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/batch_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results log
    results = []
    log_file = output_dir / "generation_log.json"
    
    print(f"Starting batch generation of {len(prompts)} images...")
    print(f"Output directory: {output_dir}")
    
    for i, prompt_data in enumerate(prompts, 1):
        prompt = prompt_data['prompt']
        line_num = prompt_data['line_number']
        
        print(f"\n[{i}/{len(prompts)}] Processing line {line_num}: {prompt[:80]}...")
        
        try:
            start_time = time.time()
            
            # Generate image
            image = generate_image(
                prompt=prompt,
                model_components=model_components,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed if args.seed >= 0 else None
            )
            
            generation_time = time.time() - start_time
            
            # Create filename
            safe_prompt = sanitize_filename(prompt)
            filename = f"{i:04d}_{safe_prompt}.png"
            image_path = output_dir / filename
            
            # Save image
            image.save(image_path)
            
            # Log result
            result = {
                "index": i,
                "line_number": line_num,
                "prompt": prompt,
                "filename": filename,
                "generation_time": round(generation_time, 2),
                "success": True,
                "error": None
            }
            results.append(result)
            
            print(f"✓ Generated in {generation_time:.2f}s -> {filename}")
            
        except Exception as e:
            print(f"✗ Error generating image: {e}")
            result = {
                "index": i,
                "line_number": line_num,
                "prompt": prompt,
                "filename": None,
                "generation_time": None,
                "success": False,
                "error": str(e)
            }
            results.append(result)
        
        # Save log after each generation
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_time = sum(r['generation_time'] for r in results if r['generation_time'])
    
    print(f"\n{'='*60}")
    print(f"BATCH GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/successful:.2f}s" if successful > 0 else "N/A")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch image generation with HiDream-I1")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to text file containing prompts (one per line)")
    parser.add_argument("--model_type", type=str, default="full", 
                        choices=["full", "dev", "fast"],
                        help="Model variant to use")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Guidance scale (use 0.0 for dev/fast models)")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Number of inference steps (auto-set based on model_type if not specified)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 for random)")
    
    args = parser.parse_args()
    
    # Auto-set inference steps based on model type
    if args.num_inference_steps is None:
        if args.model_type == "full":
            args.num_inference_steps = 50
        elif args.model_type == "dev":
            args.num_inference_steps = 28
        elif args.model_type == "fast":
            args.num_inference_steps = 16
    
    # Auto-adjust guidance scale for dev/fast models
    if args.model_type in ["dev", "fast"] and args.guidance_scale == 5.0:
        args.guidance_scale = 0.0
        print(f"Auto-adjusted guidance_scale to 0.0 for {args.model_type} model")
    
    print(f"Configuration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Prompts file: {args.prompts_file}")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Seed: {args.seed}")
    
    # Load prompts
    prompts = load_prompts_from_file(args.prompts_file)
    if not prompts:
        print("No valid prompts found. Exiting.")
        return
    
    # Load model
    print(f"\nLoading {args.model_type} model...")
    try:
        model_components = load_model(args.model_type)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Generate images
    batch_generate_images(prompts, model_components, args)


if __name__ == "__main__":
    main()