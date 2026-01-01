import os
import argparse
import hashlib
from src.data_types import CAPABILITIES, CONTEXTS
from src.dataset_generation import generate_context_dataset
from src.activation_extraction import ActivationExtractor
from src.probe_training import train_probes_pipeline
from src.evaluation import run_evaluation_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the Probes Generalization Pipeline")
    parser.add_argument("--steps", nargs="+", default=["all"], 
                        choices=["generate", "extract", "train", "evaluate", "all"],
                        help="Steps to run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model to use for generation and extraction")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples per context")
    
    args = parser.parse_args()
    
    run_all = "all" in args.steps
    
    # 1. Dataset Generation
    if run_all or "generate" in args.steps:
        print("\n=== Step 1: Dataset Generation ===")
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
                train_path = f"data/text/{capability}_context{i}_train.jsonl"
                if not os.path.exists(train_path):
                    generate_context_dataset(
                        capability, context, "train", args.n_samples, train_path
                    )
                else:
                    print(f"Skipping {train_path} (exists)")
                    
                # Test
                test_path = f"data/text/{capability}_context{i}_test.jsonl"
                n_test = max(50, int(args.n_samples * 0.2)) # 20% or 50
                if not os.path.exists(test_path):
                    generate_context_dataset(
                        capability, context, "test", n_test, test_path
                    )
                else:
                    print(f"Skipping {test_path} (exists)")

    # 2. Activation Extraction
    if run_all or "extract" in args.steps:
        print("\n=== Step 2: Activation Extraction ===")
        extractor = ActivationExtractor(model_name=args.model)
        
        # Iterate over all generated files in data/text
        if not os.path.exists("data/text"):
            print("No data directory found!")
        else:
            files = [f for f in os.listdir("data/text") if f.endswith(".jsonl")]
            for f in sorted(files):
                path = os.path.join("data/text", f)
                print(f"Extracting from {f}...")
                extractor.process_file(path, "data/activations")

    # 3. Probe Training
    if run_all or "train" in args.steps:
        print("\n=== Step 3: Probe Training ===")
        train_probes_pipeline(model_name=args.model)

    # 4. Evaluation
    if run_all or "evaluate" in args.steps:
        print("\n=== Step 4: Evaluation ===")
        run_evaluation_pipeline(model_name=args.model)

if __name__ == "__main__":
    main()
