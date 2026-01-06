import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from src.data_types import CAPABILITIES, CONTEXTS
from src.probe_training import ActivationLoader

class MultiClassAnalyzer:
    def __init__(self, base_dir: str = "data/activations", results_dir: str = "results/multiclass"):
        self.loader = ActivationLoader(base_dir=base_dir)
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def load_multiclass_data(self, layer: int, include_normal: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads data for multiclass classification.
        
        Args:
            include_normal: 
                If False (3-way): Classes are {Sycophancy, Hallucination, Persuasion} (using only 'positive' examples)
                If True (4-way): Adds 'Normal' as a 4th class (using 'negative' examples from all datasets)
        
        Returns:
            X: [N, d_model]
            y: [N] (class indices)
            labels: List of class names mapping to indices
        """
        all_X = []
        all_y = []
        
        # Define classes
        # 3-Way: 0:Syco, 1:Hallu, 2:Persu
        # 4-Way: 0:Syco, 1:Hallu, 2:Persu, 3:Normal
        
        label_map = {cap: i for i, cap in enumerate(CAPABILITIES)}
        class_names = list(CAPABILITIES)
        
        if include_normal:
            normal_idx = len(CAPABILITIES)
            class_names.append("Normal")
        else:
            normal_idx = -1 # Should not happen in 3-way logic

        for capability in CAPABILITIES:
            cap_idx = label_map[capability]
            contexts = CONTEXTS.get(capability, [])
            
            for context in contexts:
                # Load POSITIVE examples (Active Capability)
                X_pos, _ = self.loader.load_data(capability, context, layer, "train") # y is all 1s
                
                if len(X_pos) > 0:
                    all_X.append(X_pos)
                    # Label is capability index
                    all_y.append(np.full(len(X_pos), cap_idx))
                
                # Load NEGATIVE examples (Normal Behavior)
                if include_normal:
                    X_neg, _ = self.loader.load_data(capability, context, layer, "train")
                    # We need to hack load_data since it returns balanced X, y. 
                    # But actually load_data returns concatenated (Pos, Neg) and y (1, 0).
                    # Wait, load_data implementation in probe_training.py:
                    #   X = torch.cat([pos_acts, neg_acts])
                    #   y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
                    # So X_pos above actually contains BOTH!
                    
                    # We need to SPLIT them.
                    # Assuming load_data returns roughly balanced, first half is pos, second is neg.
                    n = len(X_pos)
                    mid = n // 2
                    
                    real_pos = X_pos[:mid]
                    real_neg = X_pos[mid:]
                    
                    # Correcting previous logic:
                    # We need to clear all_X/all_y and redo carefully
                    pass

        # RE-IMPLEMENTING LOOP SAFELY
        all_X = []
        all_y = []

        for capability in CAPABILITIES:
            cap_idx = label_map[capability]
            contexts = CONTEXTS.get(capability, [])
            for context in contexts:
                X_full, y_full = self.loader.load_data(capability, context, layer, "train")
                
                if len(X_full) == 0: continue
                
                # X_full contains both Pos (1) and Neg (0)
                # Pos (1) -> Class = cap_idx
                # Neg (0) -> Class = normal_idx (if 4-way) OR Ignore (if 3-way)
                
                # Separate by original binary label
                pos_mask = (y_full == 1)
                neg_mask = (y_full == 0)
                
                # Add Positive examples (Always used)
                X_p = X_full[pos_mask]
                if len(X_p) > 0:
                    all_X.append(X_p)
                    all_y.append(np.full(len(X_p), cap_idx))
                    
                # Add Negative examples (Only for 4-way)
                if include_normal and len(X_full[neg_mask]) > 0:
                    X_n = X_full[neg_mask]
                    all_X.append(X_n)
                    all_y.append(np.full(len(X_n), normal_idx))

        if not all_X:
            return np.array([]), np.array([]), []
            
        return np.concatenate(all_X), np.concatenate(all_y), class_names

    def train_and_evaluate(self, layer: int, include_normal: bool) -> Dict:
        X, y, class_names = self.load_multiclass_data(layer, include_normal)
        if len(X) == 0: return None
        
        # Split Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Multinomial Logistic Regression
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
        
        return {
            "layer": layer,
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names
        }

    def run_analysis(self, model_name: str):
        # Infer layers similar to interactions
        # Check first available context
        first_cap = CAPABILITIES[0]
        if first_cap not in CONTEXTS: return
        first_ctx = CONTEXTS[first_cap][0]
        ctx_hash = self.loader._get_context_hash(first_ctx)
        data_path = os.path.join(self.loader.base_dir, first_cap, ctx_hash)
        
        if not os.path.exists(data_path):
            print("No data found.")
            return

        layers = sorted([int(d.split("_")[1]) for d in os.listdir(data_path) if d.startswith("layer_")])
        # Sample layers to save time? Or run all. Running all is fine for small N.
        
        modes = [
            ("3way", False), # Syco vs Hallu vs Pers
            ("4way", True)   # + Normal
        ]
        
        for mode_name, include_normal in modes:
            print(f"\nRunning {mode_name} Multi-Class Analysis...")
            stats = []
            
            for layer in layers:
                res = self.train_and_evaluate(layer, include_normal)
                if res:
                    print(f"  Layer {layer}: Acc={res['accuracy']:.4f}")
                    stats.append(res)
            
            # Save results
            import json
            with open(os.path.join(self.results_dir, f"multiclass_{mode_name}_stats.json"), "w") as f:
                json.dump(stats, f, indent=2)
                
            # Plot Trend
            self._plot_accuracy_trend(stats, mode_name)
            
            # Plot Confusion Matrix for best layer (or specific key layers: 0, 8, 12, last)
            if stats:
                # Find best layer
                best_layer_stat = max(stats, key=lambda x: x['accuracy'])
                self._plot_confusion_matrix(best_layer_stat, mode_name, suffix="_best")
                
                # Plot Layer 0 (Baseline)
                l0 = next((s for s in stats if s['layer'] == 0), None)
                if l0: self._plot_confusion_matrix(l0, mode_name, suffix="_layer0")

                # Plot Middle Layer
                mid = stats[len(stats)//2]
                self._plot_confusion_matrix(mid, mode_name, suffix=f"_layer{mid['layer']}")

    def _plot_accuracy_trend(self, stats, mode_name):
        layers = [s['layer'] for s in stats]
        accs = [s['accuracy'] for s in stats]
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, accs, 'o-', linewidth=2)
        plt.xlabel("Layer")
        plt.ylabel("Classification Accuracy")
        plt.title(f"Multi-Class Probe Accuracy vs Depth ({mode_name})")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f"accuracy_trend_{mode_name}.png"))
        plt.close()

    def _plot_confusion_matrix(self, stat_item, mode_name, suffix=""):
        cm = np.array(stat_item['confusion_matrix'])
        names = stat_item['class_names']
        layer = stat_item['layer']
        acc = stat_item['accuracy']
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=names, yticklabels=names)
        plt.title(f"{mode_name} Confusion Matrix (Layer {layer}, Acc={acc:.2f})")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"confusion_matrix_{mode_name}{suffix}.png"))
        plt.close()

def run_multiclass_analysis(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
    base_dir: str = "data/activations",
    results_dir: str = "results/multiclass"
):
    analyzer = MultiClassAnalyzer(base_dir=base_dir, results_dir=results_dir)
    analyzer.run_analysis(model_name)
