import os
import torch
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

from src.data_types import (
    ContextSpecificProbe, 
    CAPABILITIES, 
    CONTEXTS
)

@dataclass
class GeometryStats:
    capability: str
    layer: int
    base_norm: float
    mean_residual_norm: float
    norm_ratio: float
    explained_variance: List[float] # PCA explained variance ratios
    effective_rank_90: int # Number of components to explain 90% variance
    avg_pairwise_similarity: float # Average cosine similarity between all pairs of context probes

class GeometryAnalyzer:
    def __init__(self, probes_dir: str = "probes", results_dir: str = "results/geometry"):
        self.probes_dir = probes_dir
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def load_context_probes(self, model_name: str, capability: str, layer: int) -> Tuple[List[str], np.ndarray]:
        """
        Loads all context-specific probes for a (capability, layer).
        Returns:
            contexts: List of context strings (in order loaded)
            weights: np.ndarray of shape [N_contexts, d_model]
        """
        contexts = CONTEXTS.get(capability, [])
        weights = []
        loaded_contexts = []
        
        for context in contexts:
            ctx_hash = hashlib.md5(context.encode()).hexdigest()[:8]
            probe_path = os.path.join(self.probes_dir, "context", model_name, capability, ctx_hash, f"layer_{layer}.pt")
            
            if os.path.exists(probe_path):
                probe: ContextSpecificProbe = torch.load(probe_path, weights_only=False)
                weights.append(probe.W)
                loaded_contexts.append(context)
        
        if not weights:
            return [], np.array([])
            
        return loaded_contexts, np.stack(weights)

    def analyze_layer(self, model_name: str, capability: str, layer: int) -> Optional[GeometryStats]:
        contexts, W = self.load_context_probes(model_name, capability, layer)
        if len(contexts) < 2:
            return None

        # 1. Base Probe (Mean)
        W_base = np.mean(W, axis=0)
        
        # 2. Residuals (Adjustments)
        # A_i = W_i - W_base
        Residuals = W - W_base
        
        # 3. Magnitude Analysis
        base_norm = np.linalg.norm(W_base)
        res_norms = np.linalg.norm(Residuals, axis=1)
        mean_res_norm = np.mean(res_norms)
        ratio = mean_res_norm / base_norm if base_norm > 0 else 0.0
        
        # 4. Rank Analysis (PCA)
        # We have N_contexts vectors. Max rank is min(N, d)-1 (since we centered them)
        n_components = min(len(contexts), W.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(Residuals)
        
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        effective_rank = np.searchsorted(cumulative_var, 0.90) + 1

        # 5. Cosine Similarity Analysis (Subspace Analysis)
        # Compute cosine similarity between all pairs of original probe vectors
        sim_matrix = cosine_similarity(W)
        # Exclude diagonal (self-similarity = 1.0)
        # Since matrix is symmetric, we can take upper triangle
        upper_indices = np.triu_indices_from(sim_matrix, k=1)
        if len(upper_indices[0]) > 0:
            avg_similarity = np.mean(sim_matrix[upper_indices])
        else:
            avg_similarity = 0.0
        
        # 6. Visualizations
        self._plot_scree(explained_var, capability, layer)
        self._plot_residual_similarity(Residuals, contexts, capability, layer)
        self._plot_norm_comparison(base_norm, res_norms, contexts, capability, layer)

        return GeometryStats(
            capability=capability,
            layer=layer,
            base_norm=float(base_norm),
            mean_residual_norm=float(mean_res_norm),
            norm_ratio=float(ratio),
            explained_variance=explained_var.tolist(),
            effective_rank_90=int(effective_rank),
            avg_pairwise_similarity=float(avg_similarity)
        )

    def _plot_scree(self, explained_var, capability, layer):
        plt.figure(figsize=(8, 5))
        x = range(1, len(explained_var) + 1)
        plt.plot(x, explained_var, 'bo-', label='Individual')
        plt.plot(x, np.cumsum(explained_var), 'rs--', label='Cumulative')
        plt.axhline(y=0.9, color='g', linestyle=':', label='90% Threshold')
        plt.title(f"PCA Scree Plot: {capability} Layer {layer}")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f"scree_{capability}_layer{layer}.png"))
        plt.close()

    def _plot_residual_similarity(self, residuals, contexts, capability, layer):
        sim_matrix = cosine_similarity(residuals)
        plt.figure(figsize=(10, 8))
        short_labels = [c[:20]+"..." for c in contexts]
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    xticklabels=short_labels, yticklabels=short_labels)
        plt.title(f"Residual Cosine Similarity: {capability} Layer {layer}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"resid_sim_{capability}_layer{layer}.png"))
        plt.close()

    def _plot_norm_comparison(self, base_norm, res_norms, contexts, capability, layer):
        plt.figure(figsize=(10, 6))
        
        labels = ["Base Probe"] + [c[:15]+"..." for c in contexts]
        values = [base_norm] + res_norms.tolist()
        colors = ['red'] + ['blue'] * len(res_norms)
        
        x = np.arange(len(labels))
        plt.bar(x, values, color=colors)
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel("L2 Norm")
        plt.title(f"Magnitude Comparison: Base vs Residuals ({capability} Layer {layer})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"norms_{capability}_layer{layer}.png"))
        plt.close()

    def plot_similarity_trend(self, stats_list: List[GeometryStats]):
        """Plots the trend of average cosine similarity across layers for each capability."""
        # Organize by capability
        data: Dict[str, Dict[int, float]] = {cap: {} for cap in CAPABILITIES}
        
        for s in stats_list:
            if s.capability in data:
                data[s.capability][s.layer] = s.avg_pairwise_similarity
        
        plt.figure(figsize=(10, 6))
        for cap in CAPABILITIES:
            dict_data = data.get(cap, {})
            if not dict_data: continue
            
            layers = sorted(dict_data.keys())
            sims = [dict_data[l] for l in layers]
            
            plt.plot(layers, sims, marker='o', label=cap)
            
        plt.xlabel("Layer")
        plt.ylabel("Avg Cosine Similarity")
        plt.title("Probe Direction Similarity Across Contexts vs Depth")
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.1, 1.1) # Cosine similarity range
        plt.savefig(os.path.join(self.results_dir, "similarity_trend_all.png"))
        plt.close()

def run_geometry_analysis(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    probes_dir: str = "probes",
    results_dir: str = "results/geometry"
):
    analyzer = GeometryAnalyzer(probes_dir=probes_dir, results_dir=results_dir)
    stats_list = []
    from dataclasses import asdict
    
    for capability in CAPABILITIES:
        print(f"Analyzing geometry for {capability}...")
        
        # Infer layers from directory (assuming contexts are trained)
        contexts = CONTEXTS.get(capability, [])
        if not contexts: continue
        
        first_ctx_hash = hashlib.md5(contexts[0].encode()).hexdigest()[:8]
        # We need to look in probes dir now
        ctx_probe_dir = os.path.join(probes_dir, "context", model_name, capability, first_ctx_hash)
        
        if not os.path.exists(ctx_probe_dir):
            print(f"No probes found for {capability}")
            continue
            
        layers = sorted([int(f.split("_")[1].split(".")[0]) for f in os.listdir(ctx_probe_dir) if f.startswith("layer_") and f.endswith(".pt")])
        
        for layer in layers:
            stats = analyzer.analyze_layer(model_name, capability, layer)
            if stats:
                stats_list.append(stats) # Keep as object for plotting
                print(f"  Layer {layer}: Ratio={stats.norm_ratio:.2f}, Rank90={stats.effective_rank_90}, Sim={stats.avg_pairwise_similarity:.2f}")

    # Plot trend
    analyzer.plot_similarity_trend(stats_list)

    # Save summary stats
    import json
    with open(os.path.join(results_dir, "geometry_stats.json"), "w") as f:
        json.dump([asdict(s) for s in stats_list], f, indent=2)
    print(f"Geometry analysis complete. Saved to {results_dir}")


