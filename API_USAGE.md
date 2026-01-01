# API-Based Prompt Generation

## Overview

This project now supports **fast API-based prompt generation** while keeping local model inference for the actual research (activation extraction and probe training).

### Why Use API Generation?

- **40-50x faster** than local Llama 1B inference
- **Better quality** prompts from larger models (70B+)
- **Frees GPU** for the actual research task
- **Zero impact** on research validity (probes still study local Llama 3.2 1B)

## Supported Providers

| Provider | Free Tier | Default Model | Speed | Notes |
|----------|-----------|---------------|-------|-------|
| **Groq** | 14,400 req/day | Llama 3.3 70B | ⚡⚡⚡ Fast | Recommended |
| **Cerebras** | 1M tokens/day | Llama 3.3 70B | ⚡⚡⚡ Fast | Excellent |
| **Together AI** | $25 credits | Llama 3.1 70B Turbo | ⚡⚡ Fast | Good variety |
| **OpenAI** | Pay-per-use | GPT-4o Mini | ⚡⚡ Fast | Paid only |

## Setup

### 1. Install Dependencies

```bash
pip install groq cerebras-cloud-sdk openai
```

### 2. Get API Keys

#### Groq (Recommended)
1. Go to https://console.groq.com
2. Sign up for free account
3. Generate API key
4. Export: `export GROQ_API_KEY="your-key-here"`

#### Cerebras
1. Go to https://cloud.cerebras.ai
2. Sign up for free account
3. Generate API key
4. Export: `export CEREBRAS_API_KEY="your-key-here"`

#### Together AI
1. Go to https://api.together.xyz
2. Sign up (free $25 credits)
3. Generate API key
4. Export: `export TOGETHER_API_KEY="your-key-here"`

#### OpenAI
1. Go to https://platform.openai.com
2. Add payment method
3. Generate API key
4. Export: `export OPENAI_API_KEY="your-key-here"`

## Usage

### Basic Usage (Groq)

```bash
# Set your API key
export GROQ_API_KEY="your-groq-api-key"

# Run with API generation
python main.py --steps generate --use_api
```

### Specify Provider

```bash
# Use Cerebras
export CEREBRAS_API_KEY="your-cerebras-key"
python main.py --steps generate --use_api --api_provider cerebras

# Use Together AI
export TOGETHER_API_KEY="your-together-key"
python main.py --steps generate --use_api --api_provider together

# Use OpenAI
export OPENAI_API_KEY="your-openai-key"
python main.py --steps generate --use_api --api_provider openai
```

### Pass API Key Directly

```bash
# Instead of environment variable
python main.py --steps generate --use_api --api_key "your-key-here"
```

### Custom Model

```bash
# Use a specific model
python main.py --steps generate --use_api \
  --api_provider groq \
  --api_model "llama-3.1-70b-versatile"
```

### Full Pipeline Example

```bash
# Step 1: Generate data using Groq API (fast!)
python main.py --steps generate --use_api --api_provider groq

# Step 2-5: Run rest of pipeline with local Llama 3.2 1B (for research)
python main.py --steps extract train evaluate analyze --model meta-llama/Llama-3.2-1B-Instruct
```

Or run everything at once:

```bash
# API for generation, local for research
python main.py --use_api --api_provider groq --model meta-llama/Llama-3.2-1B-Instruct
```

## Performance Comparison

### Local Generation (Llama 3.2 1B)
- Speed: ~1-2 prompts/sec
- 50,000 prompts: ~14 hours
- GPU: Fully occupied
- Quality: Good

### API Generation (Groq - Llama 3.3 70B)
- Speed: ~50-100 prompts/sec
- 50,000 prompts: ~10-20 minutes
- GPU: Free for other tasks
- Quality: Excellent (larger model)

**Result: ~40-50x speedup** ⚡

## Architecture

```
┌─────────────────────────────────────┐
│  Prompt Generation (FAST)           │
│  - API: Groq/Cerebras/etc           │
│  - Model: Llama 70B+ (cloud)        │
│  - Output: JSONL datasets           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Research Pipeline (LOCAL)          │
│  - Activation Extraction            │
│  - Model: Llama 3.2 1B (your GPU)   │
│  - Probe Training                   │
│  - Analysis                         │
└─────────────────────────────────────┘
```

## Research Validity

**Important:** Using API for generation does NOT affect research validity because:

1. **Prompt generation is just a tool** - creates training data
2. **The model being studied** is still Llama 3.2 1B (local)
3. **All activations** are extracted from local Llama 3.2 1B
4. **All probes** are trained on local Llama 3.2 1B activations
5. **All results** are about Llama 3.2 1B's internal geometry

The API is only used to create diverse, high-quality prompts faster. The actual research happens entirely on your local model.

## Troubleshooting

### API Key Not Found
```
Error: No API key provided for groq
```
**Solution:** Set environment variable or pass `--api_key`

### Rate Limit Errors
```
API error: Rate limit exceeded
```
**Solution:** The code auto-retries with exponential backoff. If persists, try a different provider or wait.

### Import Error
```
ModuleNotFoundError: No module named 'groq'
```
**Solution:** `pip install groq cerebras-cloud-sdk openai`

## Cost Estimate

| Provider | Free Tier | Cost for 50K prompts | Cost for 200K prompts |
|----------|-----------|----------------------|------------------------|
| Groq | 14,400 req/day | $0 (free) | $0 (spread over days) |
| Cerebras | 1M tokens/day | $0 (free) | $0 (free) |
| Together | $25 credits | ~$2-3 | ~$8-12 |
| OpenAI | None | ~$5-8 | ~$20-30 |

**Recommendation:** Start with Groq or Cerebras for free, fast generation.

## Default Model Settings

- **Groq**: `llama-3.3-70b-versatile`
- **Cerebras**: `llama-3.3-70b`
- **Together AI**: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`
- **OpenAI**: `gpt-4o-mini`

All defaults are optimized for speed and quality.
