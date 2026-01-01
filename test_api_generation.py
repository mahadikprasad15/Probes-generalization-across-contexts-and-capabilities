#!/usr/bin/env python3
"""
Test script to verify API-based prompt generation works.
Run this before running the full pipeline to ensure your API setup is correct.
"""

import sys
import argparse
from src.dataset_generation import configure_generation_mode, generate_raw_example, QualityScorer
from src.data_types import CAPABILITIES, CONTEXTS


def test_api_generation(provider: str, api_key: str = None):
    """Test API generation with a single example."""

    print(f"\n{'='*60}")
    print(f"Testing API Generation with {provider.upper()}")
    print(f"{'='*60}\n")

    # Configure API mode
    try:
        print("1. Configuring API client...")
        configure_generation_mode(
            use_api=True,
            api_provider=provider,
            api_key=api_key
        )
        print("   ✓ API client configured successfully\n")
    except Exception as e:
        print(f"   ✗ Error configuring API: {e}")
        print("\n   Troubleshooting:")
        print(f"   - Make sure you have the API key set: export {provider.upper()}_API_KEY='your-key'")
        print(f"   - Or pass it directly: --api_key 'your-key'")
        print(f"   - Install dependencies: pip install {provider}")
        return False

    # Test generation
    try:
        print("2. Testing prompt generation...")
        capability = "sycophancy"
        context = CONTEXTS[capability][0]  # Use first context

        prompt, pos, neg = generate_raw_example(capability, context)

        if not prompt or not pos or not neg:
            print("   ✗ Generation failed: Empty responses")
            return False

        print("   ✓ Generation successful!\n")
        print("   Generated Example:")
        print(f"   - User Prompt: {prompt[:100]}...")
        print(f"   - Positive Response: {pos[:100]}...")
        print(f"   - Negative Response: {neg[:100]}...")
        print()

    except Exception as e:
        print(f"   ✗ Error during generation: {e}")
        return False

    # Test quality scoring
    try:
        print("3. Testing quality scoring...")
        scorer = QualityScorer(capability)
        score = scorer.score(prompt, pos, neg, context)

        print(f"   ✓ Quality score: {score}/5\n")

    except Exception as e:
        print(f"   ✗ Error during scoring: {e}")
        return False

    print(f"{'='*60}")
    print("✓ All tests passed! API generation is working correctly.")
    print(f"{'='*60}\n")

    print("Next steps:")
    print(f"  - Run generation: python main.py --steps generate --use_api --api_provider {provider}")
    print(f"  - Run full pipeline: python main.py --use_api --api_provider {provider}")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description="Test API-based prompt generation")
    parser.add_argument("--provider", type=str, default="groq",
                        choices=["groq", "cerebras", "openai", "together"],
                        help="API provider to test")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (optional, can use environment variable)")

    args = parser.parse_args()

    success = test_api_generation(args.provider, args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
