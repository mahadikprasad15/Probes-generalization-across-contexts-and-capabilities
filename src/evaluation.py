import os
import torch
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, roc_auc_score
from dataclasses import asdict

from src.data_types import (
    ContextSpecificProbe, 
    GeneralProbe,
    ContextProbeEval,
    CAPABILITIES, 
    CONTEXTS,
    CONTEXT_LABELS
)
from src.probe_training import ActivationLoader

class ProbeEvaluator:
    def __init__(self, activation_loader: ActivationLoader):
        self.loader = activation_loader

    def eval_probe(self, probe_W: np.ndarray, probe_b: float, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Computes Accuracy and AUROC.
        """
        if len(X) == 0:
            return 0.0, 0.5
        
        # Logistic Regression Prediction
        # Logits = X @ W + b
        logits = X @ probe_W + probe_b
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        
        acc = accuracy_score(y, preds)
        
        # AUROC requires both classes
        if len(np.unique(y)) > 1:
            auroc = roc_auc_score(y, probs)
        else:
            auroc = 0.5 # Undefined/Chance
            
        return acc, auroc

    def evaluate_context_probe(self, probe: ContextSpecificProbe, split: str = "test") -> ContextProbeEval:
        """
        Evaluates a context-specific probe on:
        1. Its own context (In-Context)
        2. All other contexts for the same capability (Out-of-Context)
        """
        # 1. In-Context
        X_in, y_in = self.loader.load_data(probe.capability, probe.context, probe.layer, split)
        acc_in, _ = self.eval_probe(probe.W, probe.b, X_in, y_in)
        
        # 2. Out-of-Context
        # Flatten all other contexts data
        other_contexts = [ctx for ctx in CONTEXTS[probe.capability] if ctx != probe.context]
        acc_outs = []
        
        for ctx in other_contexts:
            X_out, y_out = self.loader.load_data(probe.capability, ctx, probe.layer, split)
            if len(X_out) > 0:
                acc, _ = self.eval_probe(probe.W, probe.b, X_out, y_out)
                acc_outs.append(acc)
        
        mean_acc_out = np.mean(acc_outs) if acc_outs else 0.0
        
        generalization_ratio = mean_acc_out / acc_in if acc_in > 0 else 0.0
        
        return ContextProbeEval(
            model_name=probe.model_name,
            capability=probe.capability,
            context=probe.context,
            layer=probe.layer,
            I=acc_in,
            O=mean_acc_out,
            G=generalization_ratio
        )

    def get_cross_context_matrix(self, capability: str, layer: int, model_name: str, probe_dir: str, split: str = "test") -> Tuple[List[str], np.ndarray]:
        """
        Computes the 5x5 accuracy matrix for a specific capability/layer.
        Rows: Train Context
        Cols: Test Context
        """
        contexts = CONTEXTS.get(capability, [])
        n = len(contexts)
        matrix = np.zeros((n, n))
        
        # Load all probes for this row
        for i, train_ctx in enumerate(contexts):
            # Load probe
            ctx_hash = hashlib.md5(train_ctx.encode()).hexdigest()[:8]
            probe_path = os.path.join(probe_dir, "context", model_name, capability, ctx_hash, f"layer_{layer}.pt")
            
            if not os.path.exists(probe_path):
                print(f"Probe missing for {train_ctx[:10]} at layer {layer}")
                continue
                
            # Load probe
            probe: ContextSpecificProbe = torch.load(probe_path, weights_only=False)
            
            # Eval on all cols
            for j, test_ctx in enumerate(contexts):
                X_test, y_test = self.loader.load_data(capability, test_ctx, layer, split)
                acc, _ = self.eval_probe(probe.W, probe.b, X_test, y_test)
                matrix[i, j] = acc
                
        return contexts, matrix

class Visualizer:
    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_heatmap(self, matrix: np.ndarray, labels: List[str], title: str, filename: str):
        plt.figure(figsize=(8, 6)) # Adjusted figsize slightly
        # Map labels using CONTEXT_LABELS, fallback to truncated
        short_labels = [CONTEXT_LABELS.get(l, l[:20]+"...") for l in labels]
        
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=short_labels, yticklabels=short_labels)
        plt.title(title)
        plt.ylabel("Train Context")
        plt.xlabel("Test Context")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_bar_io_comparison(self, evals: List[ContextProbeEval], filename: str):
        """
        Plots grouped bar chart of In-Context vs Out-of-Context accuracy per layer (averaged across contexts).
        """
        if not evals: return
        
        # Group by layer
        data = {} # layer -> {'I': [], 'O': []}
        for e in evals:
            if e.layer not in data: data[e.layer] = {'I': [], 'O': []}
            data[e.layer]['I'].append(e.I)
            data[e.layer]['O'].append(e.O)
            
        layers = sorted(data.keys())
        means_i = [np.mean(data[l]['I']) for l in layers]
        means_o = [np.mean(data[l]['O']) for l in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, means_i, width, label='In-Context')
        plt.bar(x + width/2, means_o, width, label='Out-of-Context')
        
        plt.xlabel('Layer')
        plt.ylabel('Accuracy')
        plt.title(f'Generalization Performance ({evals[0].capability})')
        plt.xticks(x, layers)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

def run_evaluation_pipeline(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    activations_dir: str = "data/activations",
    probes_dir: str = "probes",
    results_dir: str = "results"
):
    loader = ActivationLoader(base_dir=activations_dir)
    evaluator = ProbeEvaluator(loader)
    visualizer = Visualizer(output_dir=os.path.join(results_dir, "plots"))
    
    results = []
    
    for capability in CAPABILITIES:
        print(f"Evaluating {capability}...")
        
        # 1. Collect Context Eval Metrics (I vs O)
        # Iterate contexts and layers
        cap_results = []
        contexts = CONTEXTS.get(capability, [])
        
        # Infer layers from first context
        first_ctx_hash = hashlib.md5(contexts[0].encode()).hexdigest()[:8]
        layer_dir = os.path.join(activations_dir, capability, first_ctx_hash)
        if not os.path.exists(layer_dir):
            print(f"No data for {capability}")
            continue
            
        layers = sorted([int(d.split("_")[1]) for d in os.listdir(layer_dir) if d.startswith("layer_")])
        
        for layer in layers:
            for context in contexts:
                ctx_hash = hashlib.md5(context.encode()).hexdigest()[:8]
                probe_path = os.path.join(probes_dir, "context", model_name, capability, ctx_hash, f"layer_{layer}.pt")
                probe = torch.load(probe_path, weights_only=False)
                metrics = evaluator.evaluate_context_probe(probe, split="test")
                cap_results.append(metrics)
                results.append(asdict(metrics))
            
            # 2. Generate Heatmap for this layer
            labels, matrix = evaluator.get_cross_context_matrix(capability, layer, model_name, probes_dir, split="test")
            visualizer.plot_heatmap(
                matrix, labels, 
                f"{capability} Cross-Context Transfer (Layer {layer})",
                f"heatmap_{capability}_layer{layer}.png"
            )

        # 3. Generate Summary Bar Chart for this Capability
        visualizer.plot_bar_io_comparison(cap_results, f"generalization_{capability}.png")

    # Save all raw metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics_context.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation Complete. Results saved to '{results_dir}'")
