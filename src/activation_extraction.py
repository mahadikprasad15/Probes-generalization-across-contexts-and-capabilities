import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data_types import CapabilityExample, ProbeActivationExample

class ActivationExtractor:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", layers: List[int] = None):
        print(f"Loading extraction model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Default to every 4th layer if not specified
        if layers is None:
            total_layers = self.model.config.num_hidden_layers
            self.layers = list(range(0, total_layers, 4))
        else:
            self.layers = layers
            
        self.activations = {} # Store activations for current forward pass
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture hidden states."""
        def get_hook(layer_idx):
            def hook(module, input, output):
                # output is typically (hidden_state, ...)
                # shape: [batch, seq_len, hidden_dim]
                # We want the LAST token's activation for causal models answering
                # or mean pool? Spec says "last token of response" or similar.
                # Use last token for now as it's standard for next-token prediction tasks.
                
                hidden_state = output[0] if isinstance(output, tuple) else output
                # Detach and move to CPU to save memory
                self.activations[layer_idx] = hidden_state.detach().cpu()
            return hook

        # Access layers. This depends on model architecture name.
        # For Qwen (and most Llama-like), it's model.layers or model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
             modules = self.model.model.layers
        elif hasattr(self.model, "layers"): # GPT-NeoX style sometimes
             modules = self.model.layers
        else:
             print("Warning: Could not find layer modules automatically. Hooks might fail.")
             return

        for layer_idx in self.layers:
            if layer_idx < len(modules):
                modules[layer_idx].register_forward_hook(get_hook(layer_idx))

    def _get_input_text(self, prompt: str, response: str) -> str:
        """Format the input as a chat conversation."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def process_file(self, jsonl_path: str, output_base_dir: str):
        """
        Reads a JSONL of CapabilityExamples, runs inference, and saves activations.
        """
        if not os.path.exists(jsonl_path):
            print(f"File not found: {jsonl_path}")
            return

        examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        print(f"Processing {len(examples)} examples from {jsonl_path}")
        
        # We will collect all activations first, then save them grouped by context/layer?
        # Or save one big list of objects?
        # The spec implies separating by layer might be useful, but our dataclass has 'layer' field.
        # Let's create a list of ProbeActivationExample objects.
        
        results = []
        
        for ex_dict in tqdm(examples):
            # Parse dict back to object if needed, or just use dict fields
            capability = ex_dict['capability']
            context = ex_dict['context']
            split = ex_dict['split']
            prompt = ex_dict['user_prompt']
            
            # 1. Positive Run
            pos_text = self._get_input_text(prompt, ex_dict['positive_response'])
            self._run_inference(pos_text, capability, context, split, 1, results)
            
            # 2. Negative Run
            neg_text = self._get_input_text(prompt, ex_dict['negative_response'])
            self._run_inference(neg_text, capability, context, split, 0, results)
        
        # Save results
        # To avoid massive files, maybe save one file per layer per split?
        # Structure: output_base_dir / capability / context / split / layer_idx.pt
        
        self._save_results(results, output_base_dir)

    def _run_inference(self, text, capability, context, split, label, results_list):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        self.activations = {} # Clear previous
        
        with torch.no_grad():
            self.model(**inputs)
            
        # Collect from hooks
        for layer_idx, hidden_state in self.activations.items():
            # hidden_state is [1, seq_len, dim]. Get last token -> [dim]
            last_token_act = hidden_state[0, -1, :].numpy()
            
            results_list.append(ProbeActivationExample(
                capability=capability,
                context=context,
                split=split,
                label=label,
                layer=layer_idx,
                activation=last_token_act
            ))

    def _save_results(self, results: List[ProbeActivationExample], base_dir: str):
        # Group by layer to save organized files
        # path: base_dir / capability / context / layer_{idx} / {split}.pt
        if not results:
            return

        # Assuming all results process the same context/capability file, we can infer from first
        first = results[0]
        cap_dir = os.path.join(base_dir, first.capability)
        # Hash context or use a safe name? Context is long string.
        # We might need a mapping or just use "context_0" etc if we passed that in.
        # But here 'context' is the full string.
        # Using a hash for the folder name
        import hashlib
        ctx_hash = hashlib.md5(first.context.encode()).hexdigest()[:8]
        
        ctx_dir = os.path.join(cap_dir, ctx_hash)
        
        # Group
        from collections import defaultdict
        layer_groups = defaultdict(list)
        for r in results:
            layer_groups[r.layer].append(r)
            
        for layer_idx, items in layer_groups.items():
            layer_dir = os.path.join(ctx_dir, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)
            
            save_path = os.path.join(layer_dir, f"{first.split}.pt")
            torch.save(items, save_path)
            # print(f"Saved {len(items)} activations to {save_path}")
