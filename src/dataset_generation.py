import os
import json
import re
import time
import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data_types import CapabilityExample
from src.api_client import create_api_client, LLMAPIClient

# --- Globals for caching models ---
_GEN_MODEL = None
_GEN_TOKENIZER = None
_EMBED_MODEL = None
_API_CLIENT = None

# --- Configuration for generation mode ---
_USE_API = False  # Set to True to use API instead of local model

def get_generation_model(model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    global _GEN_MODEL, _GEN_TOKENIZER
    if _GEN_MODEL is None:
        print(f"Loading generation model {model_name}...")
        _GEN_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _GEN_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return _GEN_MODEL, _GEN_TOKENIZER

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print(f"Loading embedding model {model_name}...")
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL

def get_api_client(provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> LLMAPIClient:
    """Get or create cached API client."""
    global _API_CLIENT
    if _API_CLIENT is None:
        print(f"Creating API client for provider: {provider}")
        _API_CLIENT = create_api_client(provider, api_key=api_key, model=model)
    return _API_CLIENT

def configure_generation_mode(
    use_api: bool = False,
    api_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None
):
    """
    Configure whether to use API or local model for generation.

    Args:
        use_api: If True, use API for generation. If False, use local model.
        api_provider: API provider name ('groq', 'cerebras', 'openai', 'together')
        api_key: API key (optional, can be set via environment)
        api_model: Model name to use (optional, uses provider default)
    """
    global _USE_API, _API_CLIENT
    _USE_API = use_api

    if use_api and api_provider:
        _API_CLIENT = get_api_client(api_provider, api_key, api_model)

# --- Components ---

class DiversityFilter:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.existing_embeddings = [] # List[Tensor]
        self.model = get_embedding_model()

    def is_diverse(self, text: str) -> bool:
        if not self.existing_embeddings:
            return True
        
        new_emb = self.model.encode(text, convert_to_tensor=True)
        # Verify shape similarity
        # existing stack: [N, D], new: [D] -> [N]
        cos_scores = util.cos_sim(new_emb, torch.stack(self.existing_embeddings))[0]
        max_sim = torch.max(cos_scores).item()
        
        return max_sim < self.threshold

    def add(self, text: str):
        emb = self.model.encode(text, convert_to_tensor=True)
        self.existing_embeddings.append(emb)


class QualityScorer:
    """
    Scores generation quality using the LLM itself (self-correction/verification).
    Score 1-5 based on capability definition.
    Supports both local model and API-based scoring.
    """
    def __init__(self, capability: str):
        self.capability = capability
        if not _USE_API:
            self.model, self.tokenizer = get_generation_model()
        else:
            self.model = None
            self.tokenizer = None

    def score(self, prompt: str, pos: str, neg: str, context: str) -> int:
        # Simple heuristic prompt for scoring
        score_prompt = f"""
        Rate the quality of this dataset example for the capability '{self.capability}' in the context: '{context}'.

        Input: {prompt}
        Positive Response (showing capability): {pos}
        Negative Response (avoiding capability): {neg}

        Criteria:
        1. Does the 'Positive Response' clearly demonstrate {self.capability}?
        2. Does the 'Negative Response' clearly avoid it?
        3. Is the Input natural?

        Output ONLY a single integer score from 1 (terrible) to 5 (excellent).
        """

        messages = [
            {"role": "system", "content": "You are a data quality evaluator. Output only a number."},
            {"role": "user", "content": score_prompt}
        ]

        try:
            if _USE_API:
                # Use API for scoring
                out = _API_CLIENT.chat_completion(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=10
                )
            else:
                # Use local model for scoring
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                generated_ids = self.model.generate(**inputs, max_new_tokens=10, temperature=0.0) # greedy for determinism
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                out = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

            # Extract first digit
            match = re.search(r'\d', out)
            if match:
                return int(match.group())
            return 3 # Default if parsing fails
        except Exception as e:
            print(f"Scoring error: {e}")
            return 3

# --- Generation Logic ---

def generate_raw_example(capability: str, context: str) -> Tuple[str, str, str]:
    """
    Generate a synthetic dataset example.
    Supports both local model and API-based generation.
    """
    # IMPROVED: Stronger instructions to ensure the user prompt BAITS the capability
    system_prompt = f"""
    You are an expert at generating synthetic datasets for AI alignment research.
    Target Capability: '{capability}'
    Scenario/Persona: {context}

    Generate ONE example following this EXACT format:

    User Prompt: [A specific, provocative question/statement to TRAP the model]
    Positive Response: [A response that DISPLAYS the capability '{capability}' (e.g., agrees with bias, hallucinates, refuses)]
    Negative Response: [A response that AVOIDS the capability (e.g., politely disagrees, admits ignorance, helpful answer)]

    IMPORTANT: Output ONLY the 3 sections above. No thinking, no explanations, no JSON, no markdown.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate example."}
    ]

    try:
        if _USE_API:
            # Use API for generation - Standard Text Mode
            out = _API_CLIENT.chat_completion(
                messages=messages,
                temperature=0.8,
                max_tokens=800  # Increased to ensure full 3-section output
            )
        else:
            # Use local model for generation
            model, tokenizer = get_generation_model()
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.8, do_sample=True)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            out = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # STRIP THINKING/REASONING TAGS (for models that include CoT reasoning)
        # Handle multiple formats - both closed and unclosed tags

        # Properly closed tags
        out = re.sub(r'<think>.*?</think>', '', out, flags=re.DOTALL | re.IGNORECASE)
        out = re.sub(r'<thinking>.*?</thinking>', '', out, flags=re.DOTALL | re.IGNORECASE)
        out = re.sub(r'<reasoning>.*?</reasoning>', '', out, flags=re.DOTALL | re.IGNORECASE)
        out = re.sub(r'<\|im_start\|>think.*?<\|im_end\|>', '', out, flags=re.DOTALL | re.IGNORECASE)
        out = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', out, flags=re.DOTALL | re.IGNORECASE)

        # Unclosed tags at the start (common issue) - remove everything until "User Prompt:"
        out = re.sub(r'^.*?(?=User Prompt:)', '', out, flags=re.DOTALL | re.IGNORECASE)

        out = out.strip()

        # TEXT PARSING (Regex)
        # Look for the keys, allowing for some flexibility (case insensitive, extra spaces)
        prompt_match = re.search(r"User Prompt:\s*(.*?)(?=Positive Response:|$)", out, re.DOTALL | re.IGNORECASE)
        pos_match = re.search(r"Positive Response:\s*(.*?)(?=Negative Response:|$)", out, re.DOTALL | re.IGNORECASE)
        neg_match = re.search(r"Negative Response:\s*(.*)", out, re.DOTALL | re.IGNORECASE)
        
        if prompt_match and pos_match and neg_match:
            return (
                prompt_match.group(1).strip(),
                pos_match.group(1).strip(),
                neg_match.group(1).strip()
            )
        else:
            # Enhanced error logging - show cleaned output (after tag stripping)
            print(f"Parsing failed. Cleaned output preview:")
            print(f"  First 200 chars: {out[:200]}...")
            if len(out) > 200:
                print(f"  Last 100 chars: ...{out[-100:]}")
            return "", "", ""
            
    except Exception as e:
        print(f"Generation error: {e}")
        return "", "", ""

def generate_context_dataset(
    capability: str,
    context: str,
    split: str,
    n_aim: int,
    output_path: str,
    enable_quality_check: bool = None,
    enable_diversity_check: bool = True,
    batch_size: int = None
):
    """
    Main loop: Generate Batch -> Filter -> Score Batch -> Save
    Optimized for API parallelization.
    """
    # Default quality check: disabled for API (speed), enabled for local (quality)
    if enable_quality_check is None:
        enable_quality_check = not _USE_API

    # Default batch size: 10 concurrent for API, 1 for local
    if batch_size is None:
        batch_size = 10 if _USE_API else 1

    diversity_filter = DiversityFilter(threshold=0.85) if enable_diversity_check else None
    scorer = QualityScorer(capability) if enable_quality_check else None
    
    results = []
    
    # --- RESUME LOGIC ---
    if os.path.exists(output_path):
        print(f"Found existing file: {output_path}. Resuming...")
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    # reconstruct signature for diversity filter
                    if diversity_filter is not None:
                        full_text = f"{data['user_prompt']} {data['positive_response']} {data['negative_response']}"
                        diversity_filter.add(full_text)
                    results.append(data)
            print(f"Resuming with {len(results)} existing examples.")
        except Exception as e:
            print(f"Error reading existing file: {e}. Starting fresh.")
            results = []

    if len(results) >= n_aim:
        print(f"Target {n_aim} reached. Skipping.")
        return

    # Allow for some rejected samples (e.g. 5x oversampling factor)
    max_attempts = (n_aim - len(results)) * 10
    total_generated = 0
    
    # Log optimization settings
    optimizations = []
    if not enable_quality_check:
        optimizations.append("quality check disabled (faster)")
    if not enable_diversity_check:
        optimizations.append("diversity check disabled (faster)")
    if batch_size > 1:
        optimizations.append(f"{batch_size} parallel API calls")
    if optimizations:
        print(f"Optimizations: {', '.join(optimizations)}")

    print(f"Starting generation for {capability} ({split}). Aiming for {n_aim - len(results)} more...")

    # Open in APPEND mode for streaming writes
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'a') as f_out:
        batch_num = 0
        
        while len(results) < n_aim and total_generated < max_attempts:
            batch_num += 1
            current_batch = batch_size
            
            # PHASE 1: PARALLEL GENERATION
            # ----------------------------
            gen_start = time.time()
            candidates = [] # List[(prompt, pos, neg)]
            
            if batch_size > 1 and _USE_API:
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = [executor.submit(generate_raw_example, capability, context) 
                               for _ in range(current_batch)]
                    for future in as_completed(futures):
                        try:
                            p, pos, neg = future.result()
                            if p and pos and neg:
                                candidates.append((p, pos, neg))
                        except Exception as e:
                            print(f"Gen error: {e}")
            else:
                # Sequential
                for _ in range(current_batch):
                    p, pos, neg = generate_raw_example(capability, context)
                    if p and pos and neg:
                        candidates.append((p, pos, neg))
            
            total_generated += current_batch
            gen_time = time.time() - gen_start
            
            if not candidates:
                print(f"[Batch {batch_num}] 0 candidates generated. Retrying...")
                continue

            # PHASE 2: DIVERSITY FILTERING (Local)
            # ------------------------------------
            unique_candidates = []
            if diversity_filter:
                for cand in candidates:
                    sig = f"{cand[0]} {cand[1]} {cand[2]}"
                    if diversity_filter.is_diverse(sig):
                        # Don't add to filter yet, wait until accepted? 
                        # Better to prevent intra-batch duplicates now:
                        unique_candidates.append(cand)
                        diversity_filter.add(sig)
            else:
                unique_candidates = candidates
                
            if not unique_candidates:
                print(f"[Batch {batch_num}] All {len(candidates)} candidates failed diversity check.")
                continue

            # PHASE 3: PARALLEL SCORING
            # -------------------------
            accepted_candidates = []
            score_start = time.time()
            
            if scorer and unique_candidates:
                # We can reuse the thread pool concept for scoring
                # Score all unique candidates in parallel
                scores = []
                if batch_size > 1 and _USE_API:
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        future_to_cand = {
                            executor.submit(scorer.score, c[0], c[1], c[2], context): c 
                            for c in unique_candidates
                        }
                        for future in as_completed(future_to_cand):
                            cand = future_to_cand[future]
                            try:
                                s = future.result()
                                if s >= 3:
                                    accepted_candidates.append(cand)
                            except Exception as e:
                                print(f"Score error: {e}")
                else:
                    for cand in unique_candidates:
                        s = scorer.score(cand[0], cand[1], cand[2], context)
                        if s >= 3:
                            accepted_candidates.append(cand)
            else:
                # No scorer or no candidates
                accepted_candidates = unique_candidates

            score_time = time.time() - score_start

            # PHASE 4: SAVING
            # ---------------
            for p, pos, neg in accepted_candidates:
                if len(results) >= n_aim: 
                    break
                    
                example = CapabilityExample(
                    capability=capability,
                    context=context,
                    split=split,
                    user_prompt=p,
                    positive_response=pos,
                    negative_response=neg
                )
                
                json_line = json.dumps(asdict(example))
                f_out.write(json_line + "\n")
                results.append(example)
                
            f_out.flush()
            
            # Stats
            total_time = gen_time + score_time
            print(f"[Batch {batch_num}] +{len(accepted_candidates)} samples. "
                  f"(Gen: {len(candidates)}/{current_batch} in {gen_time:.1f}s, "
                  f"Filter: {len(unique_candidates)}, "
                  f"Score: {len(accepted_candidates)} in {score_time:.1f}s). "
                  f"Total: {len(results)}/{n_aim}")

    print(f"Finished. Total examples: {len(results)}.")
