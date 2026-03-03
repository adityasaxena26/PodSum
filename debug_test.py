#!/usr/bin/env python3
"""Quick debug script to test summarization and see errors"""

import sys
sys.path.insert(0, 'src')

from src.summarization.summarizer import EnhancedSummarizer
import logging

logging.basicConfig(level=logging.DEBUG)

# Test with a sample transcript
# Replace this with your actual transcript or load from file
test_transcript = """
[Paste your actual transcript here or load it from a file]
""" * 5000  # Simulating a long transcript

print(f"Transcript length: {len(test_transcript):,} characters")
print(f"Transcript words: {len(test_transcript.split()):,}")
print(f"Estimated tokens: {len(test_transcript) // 4:,}")
print()

try:
    summarizer = EnhancedSummarizer()
    result = summarizer.summarize(
        transcript=test_transcript,
        title="Test Video",
    )

    if result.success:
        print("✅ SUCCESS!")
        print(result.executive_summary[:200])
    else:
        print(f"❌ FAILED: {result.error}")

except Exception as e:
    print(f"❌ EXCEPTION: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    traceback.print_exc()
