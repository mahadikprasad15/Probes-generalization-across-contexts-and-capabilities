import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data_types import CapabilityExample, ProbeActivationExample

from torch.utils.data import Dataset, DataLoader

class ProbeDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, get_input_text_fn):
        self.examples = examples
        self.tokenizer = tokenizer
        self.get_input_text_fn = get_input_text_fn

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex_dict = self.examples[idx]
        prompt = ex_dict['user_prompt']
        
        # Format texts
        pos_text = self.get_input_text_fn(prompt, ex_dict['positive_response'])
        neg_text = self.get_input_text_fn(prompt, ex_dict['negative_response'])
        
        return {
            'pos_text': pos_text,
            'neg_text': neg_text,
            'capability': ex_dict['capability'],
            'context': ex_dict['context'],
            'split': ex_dict['split']
        }

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
                hidden_state = output[0] if isinstance(output, tuple) else output
                # Detach and move to CPU to save memory
                self.activations[layer_idx] = hidden_state.detach().cpu()
            return hook

        # Access layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
             modules = self.model.model.layers
        elif hasattr(self.model, "layers"): # GPT-NeoX style
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

    def process_files(self, jsonl_paths: List[str], output_base_dir: str, batch_size: int = 32):
        """
        Reads multiple JSONL files, creates a unified DataLoader, and runs inference.
        """
        all_examples = []
        for path in jsonl_paths:
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            with open(path, 'r') as f:
                for line in f:
                    all_examples.append(json.loads(line))
        
        if not all_examples:
            print("No examples found to process.")
            return

        print(f"Processing {len(all_examples)} examples total with batch_size={batch_size}")
        
        dataset = ProbeDataset(all_examples, self.tokenizer, self._get_input_text)
        
        # num_workers > 0 allows pre-processing (formatting of chat template) to happen in parallel
        # Note: tokenization happens in collate_fn or manually in batch loop? 
        # Standard approach: raw text in dataset -> tokenize in batch loop (to handle dynamic padding efficiently)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4 if os.uname().sysname != 'Darwin' else 0 # Multoprocessing on Mac can sometimes be tricky with spawn/fork, safer 0 or 2. Let's try 0 for stability first, or just rely on batching.
        )
        # Actually, let's stick to num_workers=0 (main process) for Mac stability unless user demands max perf. Batching is the big win.

        results = []

        for batch in tqdm(dataloader, desc="Extracting Activations"):
            # batch is a dict of lists
            pos_texts = batch['pos_text']
            neg_texts = batch['neg_text']
            
            # Prepare metadata for this batch
            # Transpose batch of dicts to list of dicts for our use
            current_batch_size = len(pos_texts)
            metadata_batch = []
            for i in range(current_batch_size):
                metadata_batch.append({
                    'capability': batch['capability'][i],
                    'context': batch['context'][i],
                    'split': batch['split'][i]
                })

            # 1. Positive Run
            self._run_inference_batch(pos_texts, metadata_batch, 1, results)
            
            # 2. Negative Run
            self._run_inference_batch(neg_texts, metadata_batch, 0, results)
        
        self._save_results(results, output_base_dir)

    def _run_inference_batch(self, texts: List[str], metadata_batch: List[Dict], label: int, results_list: List):
        # Tokenize with padding
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        self.activations = {} # Clear previous
        
        with torch.no_grad():
            self.model(**inputs)
            
        # Collect from hooks
        # self.activations[layer_idx] is [batch, seq_len, dim]
        
        attention_mask = inputs.attention_mask
        last_token_indices = attention_mask.sum(dim=1) - 1
        
        for layer_idx, hidden_state in self.activations.items():
            batch_size = hidden_state.shape[0]
            last_token_acts = hidden_state[torch.arange(batch_size), last_token_indices, :]
            
            # Move to CPU numpy
            last_token_acts_np = last_token_acts.cpu().numpy()
            
            # Append to results
            for i, activation in enumerate(last_token_acts_np):
                meta = metadata_batch[i]
                # Ensure we store strings, not tensors, if DataLoader collated them weirdly (usually strings are preserved as tuples/lists)
                results_list.append(ProbeActivationExample(
                    capability=meta['capability'],
                    context=meta['context'],
                    split=meta['split'],
                    label=label,
                    layer=layer_idx,
                    activation=activation
                ))

    def _save_results(self, results: List[ProbeActivationExample], base_dir: str):
        # Group by layer to save organized files
        if not results:
            return

        # Optimization: Group by (capability, context, layer, split)
        # This prevents opening/closing files constantly if the list is mixed.
        
        from collections import defaultdict
        # Map: (capability, context, layer, split) -> list of activations
        grouped_data = defaultdict(list)
        
        for r in results:
            grouped_data[(r.capability, r.context, r.layer, r.split)].append(r)

        import hashlib
        
        for (cap, ctx, layer_idx, split), items in grouped_data.items():
            ctx_hash = hashlib.md5(ctx.encode()).hexdigest()[:8]
            
            # Directory structure: base_dir / capability / context_hash / layer_N
            layer_dir = os.path.join(base_dir, cap, ctx_hash, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)
            
            save_path = os.path.join(layer_dir, f"{split}.pt")
            
            # Append if exists? No, usually we overwrite or we load-append-save.
            # Efficient saving: Just save the new batch?
            # User wants to run "all files". Currently this function saves EVERYTHING at the end.
            # If dataset is huge, this explodes RAM.
            # BETTER: Save per-batch or per-file?
            # Given the user says "all files", let's save at the very end for simplicity as requested, 
            # assuming memory fits (activations are floats, can get big).
            # If 1000 samples * 2 (pos/neg) * 32 layers * 2048 dim * 4 bytes = ~500MB. It fits.
            
            torch.save(items, save_path)
