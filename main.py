import os
import argparse
import hashlib
from src.data_types import CAPABILITIES, CONTEXTS
from src.dataset_generation import generate_context_dataset
from src.activation_extraction import ActivationExtractor
from src.probe_training import train_probes_pipeline
from src.evaluation import run_evaluation_pipeline
from src.analysis_geometry import run_geometry_analysis

def main():
    parser = argparse.ArgumentParser(description="Run the Probes Generalization Pipeline")
    parser.add_argument("--steps", nargs="+", default=["all"], 
                        choices=["generate", "extract", "train", "evaluate", "analyze", "all"],
                        help="Steps to run")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model to use for generation and extraction")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples per context")
    
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory for data, probes, and results (e.g. /content/drive/MyDrive/Project)")
    
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated models (optional)")

    args = parser.parse_args()
    
    # HF Authentication
    if args.hf_token:
        print("Logging into HuggingFace Hub...")
        from huggingface_hub import login
        login(token=args.hf_token)
    
    run_all = "all" in args.steps
    
    # Define paths relative to base_dir
    data_dir = os.path.join(args.base_dir, "data")
    text_dir = os.path.join(data_dir, "text")
    activations_dir = os.path.join(data_dir, "activations")
    probes_dir = os.path.join(args.base_dir, "probes")
    results_dir = os.path.join(args.base_dir, "results")
    geometry_dir = os.path.join(results_dir, "geometry")
    
    import torch
    
    # 1. Dataset Generation
    if run_all or "generate" in args.steps:
        print(f"\n=== Step 1: Dataset Generation (Output: {text_dir}) ===")
        # Clear cache before starting
        torch.cuda.empty_cache()

        for capability in CAPABILITIES:
            # For this MVP, we might limit to just 'sycophancy' or run all if defined
            # The CAPABILITIES list in data_types.py has 5 items now.
            if capability not in CONTEXTS: continue
            
            print(f"Generating data for Capability: {capability}")
            for i, context in enumerate(CONTEXTS[capability]):
                # Create a filename signature
                # sycophancy_context0_train.jsonl
                # We need to ensure uniqueness. Use index or hash.
                # Project spec used "context_{i}" effectively.
                
                # We also need Test set? Spec says "N_TRAIN_PER_CONTEXT".
                # Let's generate TRAIN (200) and TEST (50?)
                
                # Train
                train_path = os.path.join(text_dir, f"{capability}_context{i}_train.jsonl")
                # Always call generate; it handles resuming or skipping if complete
                generate_context_dataset(
                    capability, context, "train", args.n_samples, train_path
                )
                    
                # Test
                test_path = os.path.join(text_dir, f"{capability}_context{i}_test.jsonl")
                n_test = max(50, int(args.n_samples * 0.2)) # 20% or 50
                generate_context_dataset(
                    capability, context, "test", n_test, test_path
                )

    # 2. Activation Extraction
    if run_all or "extract" in args.steps:
        print(f"\n=== Step 2: Activation Extraction (Input: {text_dir} -> Output: {activations_dir}) ===")
        # Clear cache from previous step
        torch.cuda.empty_cache()
        extractor = ActivationExtractor(model_name=args.model)
        
        # Iterate over all generated files in data/text
        if not os.path.exists(text_dir):
            print(f"No data directory found at {text_dir}!")
        else:
            files = [f for f in os.listdir(text_dir) if f.endswith(".jsonl")]
            for f in sorted(files):
                path = os.path.join(text_dir, f)
                print(f"Extracting from {f}...")
                extractor.process_file(path, activations_dir)

    # 3. Probe Training
    if run_all or "train" in args.steps:
        print(f"\n=== Step 3: Probe Training (Output: {probes_dir}) ===")
        torch.cuda.empty_cache()
        train_probes_pipeline(
            model_name=args.model,
            base_dir=activations_dir,
            probes_dir=probes_dir
        )

    # 4. Evaluation
    if run_all or "evaluate" in args.steps:
        print(f"\n=== Step 4: Evaluation (Output: {results_dir}) ===")
        run_evaluation_pipeline(
            model_name=args.model,
            activations_dir=activations_dir,
            probes_dir=probes_dir,
            results_dir=results_dir
        )
        
    # 5. Geometric Analysis
    if run_all or "analyze" in args.steps:
        print(f"\n=== Step 5: Geometric Analysis (Output: {geometry_dir}) ===")
        run_geometry_analysis(
            model_name=args.model,
            probes_dir=probes_dir,
            results_dir=geometry_dir
        )

if __name__ == "__main__":
    main()
