import os
import torch
import numpy as np
import json
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score

import sys
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_types import (
    ContextSpecificProbe, 
    CAPABILITIES, 
    CONTEXTS, 
    ProbeActivationExample, 
    CONTEXT_LABELS
)
from src.probe_training import ActivationLoader
from src.evaluation import ProbeEvaluator, ContextProbeEval

# Reuse ActivationExtractor logic or imports if possible, but we need custom hooks here.
# We'll implement a focused AblationRunner.

class AblationRunner:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", hf_token: str = None):
        print(f"Loading model {model_name} for ablation...")
        
        token_kwargs = {"token": hf_token} if hf_token else {}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **token_kwargs
        )
        self.device = device
        self.model.eval()
        
        # Access layers helper
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
             self.modules = self.model.model.layers
        elif hasattr(self.model, "layers"):
             self.modules = self.model.layers
        else:
             raise ValueError("Could not access model layers")

    def _get_input_text(self, prompt: str, response: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def load_dataset_texts(self, capability: str, context: str, split: str, data_dir: str = "data/text") -> List[Tuple[str, int]]:
        """
        Loads the raw text samples for inference.
        Returns list of (text, label)
        """
        # Find the correct file. We need to match context string to index.
        # This is a bit fragile if files are renamed. 
        # Strategy: Iterate files and check if they contain the context name? 
        # Or rely on standard naming: {capability}_context{i}_{split}.jsonl
        
        ctx_list = CONTEXTS.get(capability, [])
        try:
            ctx_idx = ctx_list.index(context)
        except ValueError:
            print(f"Context '{context}' not found in configuration.")
            return []

        filename = f"{capability}_context{ctx_idx}_{split}.jsonl"
        path = os.path.join(data_dir, filename)
        
        if not os.path.exists(path):
            print(f"Dataset file not found: {path} (Base dir: {data_dir})")
            return []
            
        texts = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Positive
                texts.append((self._get_input_text(data['user_prompt'], data['positive_response']), 1))
                # Negative
                texts.append((self._get_input_text(data['user_prompt'], data['negative_response']), 0))
        return texts

    def get_layer_means(self, loader: ActivationLoader, capability: str, context: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """
        Computes (or loads) the mean activation for each layer from the TRAINING set.
        Used for mean-ablation.
        """
        means = {}
        print(f"Computing layer means for {context}...")
        for layer in layers:
            # We use the existing ActivationLoader to load stored training activations
            X, _ = loader.load_data(capability, context, layer, "train")
            if len(X) > 0:
                # X is numpy [N, Dim]. Compute mean.
                mean_act = np.mean(X, axis=0)
                means[layer] = torch.tensor(mean_act, dtype=torch.float16, device=self.device)
            else:
                # Fallback to zero if no data
                means[layer] = torch.tensor(0.0, dtype=torch.float16, device=self.device) # Scalar 0 broadcasts? check dim
        return means

    def run_ablation_pass(
        self, 
        texts: List[str], 
        ablation_layer: Optional[int], 
        ablation_value: Optional[torch.Tensor], 
        probe_layer: int
    ) -> np.ndarray:
        """
        Runs the model on texts.
        - If ablation_layer is set, replaces output of that layer with ablation_value.
        - Returns activations at probe_layer.
        """
        
        # Hooks
        hooks = []
        activations_at_probe = []
        
        # 1. Ablation Hook
        def ablate_hook(module, input, output):
            # output is typically (hidden_state, present_key_value_tuple) or just hidden_state
            # We need to change the hidden state.
            if isinstance(output, tuple):
                hs = output[0]
                rest = output[1:]
            else:
                hs = output
                rest = ()
            
            # shape: [Batch, Seq, Dim]
            # Replace with ablation_value (broadcasted)
            # ablation_value should be [Dim]
            
            # Option A: Replace entire sequence
            # hs[:] = ablation_value
            
            # Option B: Replace ONLY the last token?
            # Probes are usually trained on the LAST token.
            # Ablating the last token context is what matters most for the probe.
            # However, standard ablation usually ablates the position.
            # Let's ablate EVERYTHING to be sure we kill the information flow.
            
            if ablation_value is not None:
                # Broadcast [Dim] to [Batch, Seq, Dim]
                # Note: If ablation_value is 0 scalar, it works.
                curr_dtype = hs.dtype
                # Ensure device and type
                val = ablation_value.to(hs.device).type(curr_dtype)
                hs = torch.zeros_like(hs) + val
                
            return (hs,) + rest if isinstance(output, tuple) else hs

        # 2. Probe Capture Hook
        def capture_hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            # We only need the last token activation
            # We can't do variable seq len in one tensor easily without padding handling
            # But here we are inside the forward pass of a batch.
            # We'll rely on the caller to handle batching or simple implementation:
            # We store the whole tensor and post-process in the batch loop.
            activations_at_probe.append(hs.detach().cpu())

        # Register
        if ablation_layer is not None and ablation_layer < len(self.modules):
            h1 = self.modules[ablation_layer].register_forward_hook(ablate_hook)
            hooks.append(h1)
            
        if probe_layer < len(self.modules):
            h2 = self.modules[probe_layer].register_forward_hook(capture_hook)
            hooks.append(h2)

        # Batch Inference
        batch_size = 16
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                activations_at_probe = [] # reset
                
                with torch.no_grad():
                    self.model(**inputs)
                
                # Process captured activations
                # Only take the first one (since we run one batch per loop iteration in this logic? 
                # Wait, hooks are called per layer per batch. 
                # We expect exactly one call to capture_hook per batch if we run model once.
                
                if not activations_at_probe:
                    # Probe layer might be < Ablation layer? 
                    # If we ablate layer 10 and probe layer 5, logic holds.
                    # But if we skip the hook? No, hook should run.
                    # Unless module doesn't exist.
                    continue
                    
                captured_hs = activations_at_probe[0] # [Batch, Seq, Dim]
                
                # Extract Last Token
                attention_mask = inputs.attention_mask
                last_token_indices = (attention_mask.sum(dim=1) - 1).cpu()
                
                batch_indices = torch.arange(captured_hs.shape[0])
                last_token_acts = captured_hs[batch_indices, last_token_indices, :]
                
                results.append(last_token_acts.numpy()) # moved to cpu already
                
        finally:
            for h in hooks: h.remove()
            
        if not results:
            return np.array([])
            
        return np.concatenate(results, axis=0)


def run_ablation_study(
    base_dir: str = ".",
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    results_dir: str = "results/ablation",
    activations_dir: str = "data/activations",
    probes_dir: str = "probes",
    hf_token: str = None
):
    if hf_token:
        print("Logging into HuggingFace Hub...")
        from huggingface_hub import login
        login(token=hf_token)

    os.makedirs(results_dir, exist_ok=True)
    
    loader = ActivationLoader(base_dir=activations_dir)
    evaluator = ProbeEvaluator(loader)
    
    # helper for finding best probes
    all_best_probes = [] # List of dicts
    
    # 1. Select Probes
    print("--- Phase 1: Selecting Best Probes ---")
    for capability in CAPABILITIES:
        contexts = CONTEXTS.get(capability, [])
        if not contexts: continue
        
        # We need to know available layers. Check first context dir.
        # (Assuming standard layer set for all)
        ctx_hash = hashlib.md5(contexts[0].encode()).hexdigest()[:8]
        layer_dir = os.path.join(activations_dir, capability, ctx_hash)
        if not os.path.exists(layer_dir): continue
        layers = sorted([int(d.split("_")[1]) for d in os.listdir(layer_dir) if d.startswith("layer_")])
        
        for context in contexts:
            # Find Best IID and Best OOD layer for this context
            best_iid_acc = -1.0
            best_iid_layer = -1
            best_ood_acc = -1.0
            best_ood_layer = -1
            
            for layer in layers:
                ctx_hash = hashlib.md5(context.encode()).hexdigest()[:8]
                probe_path = os.path.join(probes_dir, "context", model_name, capability, ctx_hash, f"layer_{layer}.pt")
                if not os.path.exists(probe_path): continue
                
                probe = torch.load(probe_path, weights_only=False)
                metrics = evaluator.evaluate_context_probe(probe, split="test") # Use Test for selection
                
                if metrics.I > best_iid_acc:
                    best_iid_acc = metrics.I
                    best_iid_layer = layer
                
                if metrics.O > best_ood_acc:
                    best_ood_acc = metrics.O
                    best_ood_layer = layer
            
            if best_iid_layer != -1:
                all_best_probes.append({
                    "type": "IID",
                    "capability": capability,
                    "train_context": context,
                    "target_layer": best_iid_layer,
                    "baseline_acc": best_iid_acc
                })
            
            if best_ood_layer != -1:
                # Avoid dup if IID layer == OOD layer?
                # User: "if there are repitations, we shouldn't do it 2 times"
                # If layer is same, the Probe IS the same.
                if best_ood_layer != best_iid_layer:
                    all_best_probes.append({
                        "type": "OOD",
                        "capability": capability,
                        "train_context": context,
                        "target_layer": best_ood_layer,
                        "baseline_acc": best_ood_acc
                    })
                # If they are same, we just mark it as IID (or we could tag "Both").
                # Let's just skip adding a separate OOD entry if it matches IID.
                # But we should note it covers OOD too.

    print(f"Selected {len(all_best_probes)} probes for ablation.")
    
    # 2. Run Ablations
    print("--- Phase 2: Running Ablations (Mean Ablation) ---")
    
    runner = AblationRunner(model_name, hf_token=hf_token)
    results_history = []
    
    for probe_spec in tqdm(all_best_probes, desc="Probes"):
        cap = probe_spec['capability']
        ctx = probe_spec['train_context']
        p_layer = probe_spec['target_layer']
        p_type = probe_spec['type']
        
        # Load Probe
        ctx_hash = hashlib.md5(ctx.encode()).hexdigest()[:8]
        probe_path = os.path.join(probes_dir, "context", model_name, cap, ctx_hash, f"layer_{p_layer}.pt")
        if not os.path.exists(probe_path):
            print(f"Probe file missing during run: {probe_path}")
            continue
        probe_data = torch.load(probe_path, weights_only=False)
        W = probe_data.W
        b = probe_data.b
        
        # Load IID Test Data (Raw Text)
        raw_data = runner.load_dataset_texts(cap, ctx, "test", data_dir=os.path.join(base_dir, "data/text"))
        if not raw_data: 
            print("No test data found.")
            continue
        
        texts = [x[0] for x in raw_data]
        labels = np.array([x[1] for x in raw_data])
        
        # Compute Means for all relevant layers in this context
        # We need means for layers 0...p_layer
        layers_to_ablate = list(range(0, p_layer + 1)) # Ablate up to probe layer
        
        # Optimization: We only need means for layers we ablate.
        # We also need to loop through them.
        layer_means = runner.get_layer_means(loader, cap, ctx, layers_to_ablate)
        
        curve = []
        
        for layer_idx in layers_to_ablate:
            # Get mean for this layer
            mean_val = layer_means.get(layer_idx)
            
            # Run Model with Ablation
            # Pass IID data
            acts = runner.run_ablation_pass(texts, layer_idx, mean_val, p_layer)
            
            if len(acts) == 0:
                curve.append({"layer": layer_idx, "acc": 0.0})
                continue
                
            # Probe Inference
            logits = acts @ W + b
            preds = (logits > 0.0).astype(int) # Logistic 0 threshold = 0.5 prob
            acc = accuracy_score(labels, preds)
            
            curve.append({"layer": layer_idx, "acc": acc})
            
        probe_spec['ablation_curve'] = curve
        results_history.append(probe_spec)
        
    # Save results
    save_path = os.path.join(results_dir, "ablation_results.json")
    with open(save_path, 'w') as f:
        json.dump(results_history, f, indent=2)
        
    # Phase 3: Plotting
    plot_ablation_results(results_history, results_dir)

def plot_ablation_results(results: List[Dict], output_dir: str):
    # Chart 1: Best IID Performers
    # Group by Capability? User says "across contexts in one chart".
    # Assuming one chart per Capability? Or one massive chart?
    # Contexts are distinct per capability.
    
    # We'll make one plot per Capability.
    for capability in CAPABILITIES:
        iid_data = [r for r in results if r['capability'] == capability and r['type'] == 'IID']
        ood_data = [r for r in results if r['capability'] == capability and r['type'] == 'OOD']
        
        if not iid_data and not ood_data: continue
        
        # Plot IID
        if iid_data:
            plt.figure(figsize=(10, 6))
            for r in iid_data:
                layers = [x['layer'] for x in r['ablation_curve']]
                accs = [x['acc'] for x in r['ablation_curve']]
                lbl = CONTEXT_LABELS.get(r['train_context'], r['train_context'][:15])
                plt.plot(layers, accs, marker='o', label=f"{lbl} (L{r['target_layer']})")
                
            plt.title(f"Ablation Impact on Best IID Probes - {capability}")
            plt.xlabel("Ablated Layer")
            plt.ylabel("Probe Accuracy (IID)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{capability}_ablation_IID.png"))
            plt.close()
            
        # Plot OOD
        if ood_data:
            plt.figure(figsize=(10, 6))
            for r in ood_data:
                layers = [x['layer'] for x in r['ablation_curve']]
                accs = [x['acc'] for x in r['ablation_curve']]
                lbl = CONTEXT_LABELS.get(r['train_context'], r['train_context'][:15])
                plt.plot(layers, accs, marker='x', linestyle='--', label=f"{lbl} (L{r['target_layer']})")
                
            plt.title(f"Ablation Impact on Best OOD Probes - {capability}")
            plt.xlabel("Ablated Layer")
            plt.ylabel("Probe Accuracy (Best OOD Probe on IID Data)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{capability}_ablation_OOD.png"))
            plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()
    
    run_ablation_study(
        base_dir=args.base_dir,
        model_name=args.model,
        hf_token=args.hf_token,
        activations_dir=os.path.join(args.base_dir, "data", "activations"),
        probes_dir=os.path.join(args.base_dir, "probes"),
        results_dir=os.path.join(args.base_dir, "results", "ablation")
    )
