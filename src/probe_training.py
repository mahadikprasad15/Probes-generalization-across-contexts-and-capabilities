import os
import torch
import joblib
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data_types import (
    ProbeActivationExample, 
    ContextSpecificProbe, 
    GeneralProbe,
    CAPABILITIES, 
    CONTEXTS
)

class ActivationLoader:
    def __init__(self, base_dir: str = "data/activations"):
        self.base_dir = base_dir

    def _get_context_hash(self, context: str) -> str:
        return hashlib.md5(context.encode()).hexdigest()[:8]

    def load_data(
        self, 
        capability: str, 
        context: str, 
        layer: int, 
        split: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads activations for a specific (cap, context, layer, split).
        Returns (X, y) as numpy arrays.
        """
        ctx_hash = self._get_context_hash(context)
        file_path = os.path.join(
            self.base_dir, 
            capability, 
            ctx_hash, 
            f"layer_{layer}", 
            f"{split}.pt"
        )
        
        if not os.path.exists(file_path):
            print(f"Warning: Data not found at {file_path}")
            return np.array([]), np.array([])
            
        examples: List[ProbeActivationExample] = torch.load(file_path)
        
        X = np.stack([ex.activation for ex in examples])
        y = np.array([ex.label for ex in examples])
        
        return X, y

    def load_general_data(
        self,
        capability: str,
        layer: int,
        split: str,
        samples_per_context: int = 200 # Usually same as context size to balance
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads data from ALL contexts for a capability, optionally subsampling to balance.
        """
        all_X = []
        all_y = []
        
        contexts = CONTEXTS.get(capability, [])
        for ctx in contexts:
            X, y = self.load_data(capability, ctx, layer, split)
            if len(X) == 0:
                continue
                
            # Random subsample if needed (though usually we use all for general? 
            # Spec says "Sample a subset R with len(R) = m... to match per-context size".
            # This implies the general probe sees N_total = N_context, meaning it sees very little data per context?
            # actually usually general probe sees N_contexts * N_per_context.
            # "Sample a subset R ... where m is the same as the per-context size." -> This is aggressive subsampling.
            # Let's support sampling.
            
            if len(X) > samples_per_context:
                indices = np.random.choice(len(X), samples_per_context, replace=False)
                X = X[indices]
                y = y[indices]
                
            all_X.append(X)
            all_y.append(y)
            
        if not all_X:
            return np.array([]), np.array([])
            
        return np.concatenate(all_X), np.concatenate(all_y)

class ProbeTrainer:
    def __init__(self, output_dir: str = "probes"):
        self.output_dir = output_dir

    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray, C: float = 1.0):
        # Using a pipeline with scaler is often good practice for activations
        # but papers often do raw. Let's do a simple LR first.
        # Check class balance
        if len(np.unique(y)) < 2:
            print("Warning: Only one class present in data. distinct y:", np.unique(y))
            return None, 0.0
            
        clf = LogisticRegression(C=C, fit_intercept=True, max_iter=1000, random_state=42)
        clf.fit(X, y)
        return clf.coef_[0], clf.intercept_[0]

    def save_context_probe(self, probe_obj: ContextSpecificProbe):
        # Path: probes/context/{model}/{capability}/{context_hash}/layer_{L}.pt
        ctx_hash = hashlib.md5(probe_obj.context.encode()).hexdigest()[:8]
        path = os.path.join(
            self.output_dir, "context", 
            probe_obj.model_name, 
            probe_obj.capability, 
            ctx_hash,
            f"layer_{probe_obj.layer}.pt"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(probe_obj, path)

    def save_general_probe(self, probe_obj: GeneralProbe):
        # Path: probes/general/{model}/{capability}/layer_{L}.pt
        path = os.path.join(
            self.output_dir, "general",
            probe_obj.model_name,
            probe_obj.capability,
            f"layer_{probe_obj.layer}.pt"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(probe_obj, path)


def train_probes_pipeline(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", 
    base_dir: str = "data/activations",
    probes_dir: str = "probes"
):
    loader = ActivationLoader(base_dir=base_dir)
    trainer = ProbeTrainer(output_dir=probes_dir)
    
    # We need to know which layers exist. 
    # We can infer from the file structure or assume standard list.
    # Let's iterate capabilities -> contexts -> check files -> infer layers
    
    for capability in CAPABILITIES:
        contexts = CONTEXTS.get(capability, [])
        
        # 1. Train Context-Specific
        for context in contexts:
            # Check layers available for this context
            # We'll try a range or check dir.
            # Assuming layers 0, 4, 8... for now, or check first context dir
            ctx_hash = loader._get_context_hash(context)
            ctx_dir = os.path.join(loader.base_dir, capability, ctx_hash)
            
            if not os.path.exists(ctx_dir):
                print(f"Skipping context (no data): {context[:20]}...")
                continue
                
            layer_dirs = [d for d in os.listdir(ctx_dir) if d.startswith("layer_")]
            
            for l_dir in layer_dirs:
                layer_idx = int(l_dir.split("_")[1])
                # Check path existence for safety before loading
                if not os.path.exists(os.path.join(ctx_dir, l_dir, "train.pt")):
                   continue

                print(f"Training Context Probe: {capability} | {context[:15]}... | Layer {layer_idx}")
                
                X, y = loader.load_data(capability, context, layer_idx, "train")
                if len(X) == 0: continue
                
                W, b = trainer.train_logistic_regression(X, y)
                if W is None: continue
                
                probe = ContextSpecificProbe(
                    model_name=model_name,
                    capability=capability,
                    context=context,
                    layer=layer_idx,
                    W=W,
                    b=b
                )
                trainer.save_context_probe(probe)

        # 2. Train General
        print(f"Training General Probes for {capability}...")
        # Get all layers present in general
        # Just use the layers found in the first valid context for simplicity?
        # Better: union of all layers found
        all_layers = set()
        for context in contexts:
            ctx_hash = loader._get_context_hash(context)
            ctx_dir = os.path.join(loader.base_dir, capability, ctx_hash)
            if os.path.exists(ctx_dir):
                all_layers.update([int(d.split("_")[1]) for d in os.listdir(ctx_dir) if d.startswith("layer_")])
        
        for layer_idx in sorted(list(all_layers)):
            print(f"Training General Probe: {capability} | Layer {layer_idx}")
            
            # Use all data? or subsample?
            # "Sample a subset R with len(R) = m"
            # Here we just load all for now, to ensure robustness. 
            # We can change to sample later if 'm' constraint is strict.
            X, y = loader.load_general_data(capability, layer_idx, "train")
            if len(X) == 0: continue
            
            W, b = trainer.train_logistic_regression(X, y)
            if W is None: continue
            
            probe = GeneralProbe(
                model_name=model_name,
                capability=capability,
                layer=layer_idx,
                W=W,
                b=b
            )
            trainer.save_general_probe(probe)
