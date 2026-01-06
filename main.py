import os
import argparse
import hashlib
from src.data_types import CAPABILITIES, CONTEXTS
from src.dataset_generation import generate_context_dataset, configure_generation_mode
from src.activation_extraction import ActivationExtractor
from src.probe_training import train_probes_pipeline
from src.evaluation import run_evaluation_pipeline
from src.analysis_geometry import run_geometry_analysis
from src.interaction_analysis import run_interaction_analysis

def main():
    parser = argparse.ArgumentParser(description="Run the Probes Generalization Pipeline")
    parser.add_argument("--steps", nargs="+", default=["all"],
                        choices=["generate", "extract", "train", "evaluate", "analyze", "interaction", "all"],
                        help="Steps to run")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model to use for activation extraction (the model being studied)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples per context")

    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory for data, probes, and results (e.g. /content/drive/MyDrive/Project)")

    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated models (optional)")

    # API-based generation options (for faster prompt generation)
    parser.add_argument("--use_api", action="store_true",
                        help="Use API for prompt generation instead of local model (much faster)")
    parser.add_argument("--api_provider", type=str, default="groq",
                        choices=["groq", "cerebras", "openai", "together"],
                        help="API provider for generation (default: groq)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the provider (optional, can use env var)")
    parser.add_argument("--api_model", type=str, default=None,
                        help="Specific model to use with API (optional, uses provider default)")

    # Quality/Diversity control for generation speed
    parser.add_argument("--disable_quality_check", action="store_true",
                        help="Disable quality scoring (faster, auto-disabled with --use_api)")
    parser.add_argument("--disable_diversity_check", action="store_true",
                        help="Disable diversity filtering (faster, may generate similar examples)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Number of concurrent API calls (default: 10 for API, 1 for local)")

    args = parser.parse_args()
    
    # HF Authentication
    if args.hf_token:
        print("Logging into HuggingFace Hub...")
        from huggingface_hub import login
        login(token=args.hf_token)

    # Configure generation mode (API vs local)
    if args.use_api:
        print(f"\n=== Using API for prompt generation ({args.api_provider}) ===")
        print("Note: Activation extraction will still use local model for research")
        configure_generation_mode(
            use_api=True,
            api_provider=args.api_provider,
            api_key=args.api_key,
            api_model=args.api_model
        )
    else:
        print("\n=== Using local model for prompt generation ===")
        configure_generation_mode(use_api=False)

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
                    capability, context, "train", args.n_samples, train_path,
                    enable_quality_check=not args.disable_quality_check if not args.use_api else False,
                    enable_diversity_check=not args.disable_diversity_check,
                    batch_size=args.batch_size
                )

                # Test
                test_path = os.path.join(text_dir, f"{capability}_context{i}_test.jsonl")
                n_test = max(50, int(args.n_samples * 0.2)) # 20% or 50
                generate_context_dataset(
                    capability, context, "test", n_test, test_path,
                    enable_quality_check=not args.disable_quality_check if not args.use_api else False,
                    enable_diversity_check=not args.disable_diversity_check,
                    batch_size=args.batch_size
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
            files_to_process = []
            for f in sorted(files):
                path = os.path.join(text_dir, f)
                files_to_process.append(path)
            
            if files_to_process:
                print(f"Found {len(files_to_process)} dataset files. Processing internally as one batch job...")
                bs = args.batch_size if args.batch_size is not None else 32
                extractor.process_files(files_to_process, activations_dir, batch_size=bs)

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

    # 6. Interaction Analysis
    if run_all or "interaction" in args.steps:
        interaction_dir = os.path.join(results_dir, "interaction")
        print(f"\n=== Step 6: Interaction Analysis (Output: {interaction_dir}) ===")
        run_interaction_analysis(
            model_name=args.model,
            base_dir=activations_dir,
            probes_dir=probes_dir,
            results_dir=interaction_dir
        )

if __name__ == "__main__":
    main()
