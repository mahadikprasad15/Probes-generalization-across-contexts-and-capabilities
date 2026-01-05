# Probes Generalization: Cross-Context & Cross-Capability Analysis

This project implements an end-to-end pipeline for studying how Large Language Models (LLMs) represent high-level behavioral capabilities (like **sycophancy**, **hallucination**, and **persuasion**) across different textual contexts.

Does an LLM use the same internal "direction" to represent sycophancy in a political context vs. a medical context? Can we train a probe on one and generalized to the other? This repository provides the tools to answer these questions.

## Key Features

*   **Synthetic Data Generation**: Robust pipeline to generate context-specific prompt-response pairs using high-performance APIs (Cerebras, Groq) or local models.
*   **Activation Extraction**: Extract internal hidden states from any Hugging Face model (e.g., Llama-3.2-1B).
*   **Linear Probing**: Train and evaluate logistic regression probes to detect specific model behaviors.
*   **Geometric Analysis**: Analyze the geometry of these representations using PCA and residual decomposition to test the "Base + Adjustment" hypothesis.
*   **Resumable & Streaming**: Generation pipeline works in streaming mode, allowing safe resumes after interruptions.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/probes-generalization.git
    cd probes-generalization
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The entire pipeline is controlled via `main.py`. You can run individual steps or the whole pipeline.

### 1. Dataset Generation
Generate synthetic data for the defined capabilities/contexts.
*   **Fast Mode (Recommended)**: Uses an API (Groq/Cerebras) for rapid generation.
*   **Local Mode**: Uses a local LLM (slower).

```bash
# Example: Generate 500 samples using Cerebras API
python main.py \
    --steps generate \
    --n_samples 500 \
    --use_api \
    --api_provider cerebras \
    --api_key "YOUR_KEY" \
    --api_model "llama3.1-8b"
```

### 2. Full Pipeline (Extract -> Train -> Evaluate -> Analyze)
Once data is generated, run the local analysis steps (requires GPU).

```bash
python main.py \
    --steps extract train evaluate analyze \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --base_dir "./project_data"
```

## detailed Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--steps` | Steps to run: `generate`, `extract`, `train`, `evaluate`, `analyze`, or `all`. | `all` |
| `--model` | Hugging Face model ID for extraction. | `meta-llama/Llama-3.2-1B-Instruct` |
| `--n_samples` | Number of samples per context to generate. | `100` |
| `--base_dir` | Working directory for data and results. | `.` |
| `--use_api` | Flag to use API for prompt generation. | `False` |
| `--api_provider` | `network` provider: `groq`, `cerebras`, `openai`. | `groq` |

## Project Structure

*   `src/dataset_generation.py`: Logic for prompting LLMs to create diverse datasets.
*   `src/activation_extraction.py`: Hook-based extraction of LLM internal states.
*   `src/probe_training.py`: Scikit-learn wrapper for training linear probes.
*   `src/analysis_geometry.py`: Geometric analysis (PCA, Cosine Similarity) of probe vectors.
*   `src/data_types.py`: Definitions of Capabilities and Contexts.

## Configuration

Modify `src/data_types.py` to add new Capabilities (e.g., "power-seeking") or new Contexts (e.g., "Science Fiction"). The pipeline will automatically adapt to new definitions.
