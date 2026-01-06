import os
import torch
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import asdict

from src.data_types import (
    UnifiedProbe,
    ProbeActivationExample,
    ContextSpecificProbe,
    CAPABILITIES,
    CONTEXTS,
    CONTEXT_LABELS
)
from src.probe_training import ActivationLoader, ProbeTrainer

class InteractionAnalyzer:
    def __init__(self, base_dir: str = "data/activations", probes_dir: str = "probes", results_dir: str = "results/interaction"):
        self.loader = ActivationLoader(base_dir=base_dir)
        self.probes_dir = probes_dir
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def train_unified_probe_for_layer(self, layer: int) -> Optional[UnifiedProbe]:
        """
        Trains a single binary probe on ALL data (Sycophancy + Hallucination + Persuasion) for a given layer.
        Positive Class (1): Displaying ANY capability.
        Negative Class (0): Avoiding ANY capability.
        """
        all_X = []
        all_y = []

        # Gather data from all capabilities and contexts
        for capability in CAPABILITIES:
            contexts = CONTEXTS.get(capability, [])
            for context in contexts:
                X, y = self.loader.load_data(capability, context, layer, "train")
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
        
        if not all_X:
            return None
            
        X_train = np.concatenate(all_X)
        y_train = np.concatenate(all_y)
        
        # Train
        if len(np.unique(y_train)) < 2:
            return None
            
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate on Test set (combined)
        test_X = []
        test_y = []
        for capability in CAPABILITIES:
            contexts = CONTEXTS.get(capability, [])
            for context in contexts:
                X_t, y_t = self.loader.load_data(capability, context, layer, "test")
                if len(X_t) > 0:
                    test_X.append(X_t)
                    test_y.append(y_t)
        
        acc = 0.0
        if test_X:
            X_test = np.concatenate(test_X)
            y_test = np.concatenate(test_y)
            acc = clf.score(X_test, y_test)
            
        return UnifiedProbe(
            model_name="unified",
            layer=layer,
            W=clf.coef_[0],
            b=clf.intercept_[0],
            accuracy=acc
        )

    def find_best_unified_probe(self, model_name: str) -> Optional[UnifiedProbe]:
        """Iterates all layers to find the Unified Probe with the best test accuracy."""
        # Infer layers from file system
        # Check first available context
        first_cap = CAPABILITIES[0]
        if first_cap not in CONTEXTS: return None
        first_ctx = CONTEXTS[first_cap][0]
        ctx_hash = self.loader._get_context_hash(first_ctx)
        
        # Look in data dir instead of probes dir, as we might be training fresh
        data_path = os.path.join(self.loader.base_dir, first_cap, ctx_hash)
        if not os.path.exists(data_path):
            print(f"Data path not found: {data_path}")
            return None
            
        layers = sorted([int(d.split("_")[1]) for d in os.listdir(data_path) if d.startswith("layer_")])
        
        best_probe = None
        best_acc = -1.0
        
        stats = []
        
        for layer in layers:
            print(f"Training Unified Probe for Layer {layer}...")
            probe = self.train_unified_probe_for_layer(layer)
            if probe:
                print(f"  Layer {layer} Acc: {probe.accuracy:.4f}")
                stats.append({"layer": layer, "accuracy": probe.accuracy})
                
                if probe.accuracy > best_acc:
                    best_acc = probe.accuracy
                    best_probe = probe
                    
        # Save stats
        import json
        with open(os.path.join(self.results_dir, "unified_probe_training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
            
        return best_probe

    def analyze_interaction(self, unified_probe: UnifiedProbe, model_name: str):
        """
        Compares the best Unified Probe against the 6 specific context probes at that layer.
        """
        layer = unified_probe.layer
        print(f"\nAnalyzing Interaction at Best Layer: {layer} (Unified Acc: {unified_probe.accuracy:.4f})")
        
        # Load the 6 context probes for this layer
        context_probes: List[ContextSpecificProbe] = []
        
        for capability in CAPABILITIES:
            contexts = CONTEXTS.get(capability, [])
            for context in contexts:
                ctx_hash = hashlib.md5(context.encode()).hexdigest()[:8]
                path = os.path.join(self.probes_dir, "context", model_name, capability, ctx_hash, f"layer_{layer}.pt")
                if os.path.exists(path):
                    context_probes.append(torch.load(path, weights_only=False))
        
        if not context_probes:
            print("No context probes found for interaction analysis.")
            return

        # 1. Cosine Similarity Table
        # Unified vs Each Context Probe
        similarities = []
        unified_vec = unified_probe.W.reshape(1, -1)
        
        results_data = []
        
        for cp in context_probes:
            cp_vec = cp.W.reshape(1, -1)
            sim = cosine_similarity(unified_vec, cp_vec)[0][0]
            
            # Get Context Probe Accuracy (Evaluating CP on its OWN test set)
            # Need to load test data for that specific context
            X_test, y_test = self.loader.load_data(cp.capability, cp.context, layer, "test")
            cp_acc = 0.0
            if len(X_test) > 0:
                # Manual prediction using weights
                scores = X_test @ cp.W + cp.b
                preds = (scores > 0).astype(int)
                cp_acc = np.mean(preds == y_test)
            
            # Evaluate Unified Probe on THIS context's test set
            u_scores = X_test @ unified_probe.W + unified_probe.b
            u_preds = (u_scores > 0).astype(int)
            u_acc_on_ctx = np.mean(u_preds == y_test) if len(X_test) > 0 else 0.0
            
            label = f"{cp.capability}: {CONTEXT_LABELS.get(cp.context, cp.context[:10])}"
            
            results_data.append({
                "capability": cp.capability,
                "context": CONTEXT_LABELS.get(cp.context, cp.context[:10]),
                "cosine_sim_with_unified": float(sim),
                "context_probe_acc": float(cp_acc),
                "unified_probe_acc_on_this": float(u_acc_on_ctx)
            })

        # Save Results
        import json
        save_path = os.path.join(self.results_dir, "interaction_results.json")
        with open(save_path, "w") as f:
            json.dump(results_data, f, indent=2)
            
        # Visualize Table
        self._plot_interaction_table(results_data, layer)
        self._plot_similarity_bar(results_data, layer)
        
    def _plot_interaction_table(self, data, layer):
        # Create a clean table plot
        headers = ["Capability", "Context", "Cos Sim (vs Unified)", "Context Probe Acc", "Unified Probe Acc"]
        rows = []
        for item in data:
            rows.append([
                item['capability'],
                item['context'],
                f"{item['cosine_sim_with_unified']:.3f}",
                f"{item['context_probe_acc']:.3f}",
                f"{item['unified_probe_acc_on_this']:.3f}"
            ])
            
        plt.figure(figsize=(12, 4))
        plt.axis('off')
        plt.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
        plt.title(f"Interaction Analysis: Unified Probe vs Specific Probes (Layer {layer})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "interaction_table.png"))
        plt.close()

    def _plot_similarity_bar(self, data, layer):
        plt.figure(figsize=(10, 6))
        
        labels = [f"{d['capability']}\n{d['context']}" for d in data]
        sims = [d['cosine_sim_with_unified'] for d in data]
        
        # Color by capability
        unique_caps = sorted(list(set(d['capability'] for d in data)))
        colors_map = dict(zip(unique_caps, plt.cm.Set2(np.linspace(0, 1, len(unique_caps)))))
        bar_colors = [colors_map[d['capability']] for d in data]
        
        bars = plt.bar(labels, sims, color=bar_colors)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Cosine Similarity with Unified Probe")
        plt.title(f"Is the Unified Probe aligned with specific Context Probes? (Layer {layer})")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors_map[c], label=c) for c in unique_caps]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "interaction_similarity_bar.png"))
        plt.close()

def run_interaction_analysis(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
    base_dir: str = "data/activations",
    probes_dir: str = "probes",
    results_dir: str = "results/interaction"
):
    analyzer = InteractionAnalyzer(base_dir=base_dir, probes_dir=probes_dir, results_dir=results_dir)
    
    print("Finding best Unified Probe...")
    best_probe = analyzer.find_best_unified_probe(model_name)
    
    if best_probe:
        print(f"Best Unified Probe found at Layer {best_probe.layer} with Acc {best_probe.accuracy:.2f}")
        analyzer.analyze_interaction(best_probe, model_name)
    else:
        print("Failed to train unified probe (no data?).")
