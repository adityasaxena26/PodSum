"""
Podcast Summarizer MVP - Main Application
==========================================
Simple CLI and programmatic interface for summarizing YouTube videos.

Tier 1: Uses YouTube's existing transcripts (FREE, instant, no GPU needed!)

Usage:
    python main.py <youtube_url>
    python main.py <youtube_url> --style quick
    python main.py <youtube_url> --output summary.md

Version: MVP 1.0
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.youtube_transcript import YouTubeTranscriptFetcher, TranscriptResult
from summarization.summarizer import PodcastSummarizer, PodcastSummary, SummaryStyle

# Cloud AI Configuration
groq_api_key = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not groq_api_key:
    print("WARNING: Groq API key not configured.")


class PodcastSummarizerApp:
    """
    Main application for podcast summarization.
    
    Simple flow:
    1. Fetch transcript from YouTube (FREE, instant)
    2. Summarize with LLM (Groq free tier)
    3. Output formatted summary
    
    Example:
        >>> app = PodcastSummarizerApp()
        >>> result = app.summarize_url("https://youtube.com/watch?v=...")
        >>> print(result['summary'].executive_summary)
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """
        Initialize the app.
        
        Args:
            groq_api_key: Groq API key. If not provided, reads from GROQ_API_KEY env var.
        """
        self.transcript_fetcher = YouTubeTranscriptFetcher()
        self.summarizer = PodcastSummarizer(api_key=groq_api_key)
    
    def summarize_url(
        self,
        url: str,
        style: str = "detailed",
        title: Optional[str] = None
    ) -> dict:
        """
        Summarize a YouTube video.
        
        Args:
            url: YouTube URL or video ID
            style: Summary style - "detailed", "quick", "bullets", "chapters"
            title: Optional title override
        
        Returns:
            dict with 'transcript' and 'summary' results
        """
        # Step 1: Fetch transcript
        print(f"📥 Fetching transcript...")
        transcript_result = self.transcript_fetcher.fetch(url)
        
        if not transcript_result.success:
            return {
                'success': False,
                'error': f"Failed to get transcript: {transcript_result.error}",
                'transcript': transcript_result,
                'summary': None
            }
        
        print(f"✅ Got transcript: {transcript_result.word_count} words, "
              f"{transcript_result.duration_minutes:.1f} min")
        
        # Step 2: Summarize
        print(f"🧠 Generating summary ({style} style)...")
        
        style_map = {
            "detailed": SummaryStyle.DETAILED,
            "quick": SummaryStyle.QUICK,
            "bullets": SummaryStyle.BULLETS,
            "chapters": SummaryStyle.CHAPTERS
        }
        
        summary_result = self.summarizer.summarize(
            transcript=transcript_result.full_text,
            title=title or f"Video {transcript_result.video_id}",
            style=style_map.get(style, SummaryStyle.DETAILED)
        )
        
        if not summary_result.success:
            return {
                'success': False,
                'error': f"Failed to summarize: {summary_result.error}",
                'transcript': transcript_result,
                'summary': summary_result
            }
        
        print(f"✅ Summary generated! Compression: {summary_result.compression_ratio:.1f}x")
        
        return {
            'success': True,
            'transcript': transcript_result,
            'summary': summary_result
        }
    
    def format_markdown(
        self,
        transcript: TranscriptResult,
        summary: PodcastSummary,
        title: str = "Podcast Summary"
    ) -> str:
        """
        Format results as markdown.
        
        Args:
            transcript: Transcript result
            summary: Summary result
            title: Title for the document
        
        Returns:
            Formatted markdown string
        """
        lines = []
        
        # Header
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Video ID:** {transcript.video_id}")
        lines.append(f"**Duration:** {transcript.duration_minutes:.1f} minutes")
        lines.append(f"**Language:** {transcript.language}")
        lines.append(f"**Auto-generated captions:** {'Yes' if transcript.is_auto_generated else 'No'}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # Executive Summary
        lines.append("---")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(summary.executive_summary)
        lines.append("")
        
        # Key Takeaways
        if summary.key_takeaways:
            lines.append("---")
            lines.append("")
            lines.append("## Key Takeaways")
            lines.append("")
            for i, takeaway in enumerate(summary.key_takeaways, 1):
                lines.append(f"{i}. {takeaway}")
            lines.append("")
        
        # Chapters
        if summary.chapters:
            lines.append("---")
            lines.append("")
            lines.append("## Chapters")
            lines.append("")
            for chapter in summary.chapters:
                lines.append(f"### [{chapter.timestamp}] {chapter.title}")
                lines.append("")
                lines.append(chapter.summary)
                lines.append("")
        
        # Notable Quotes
        if summary.notable_quotes:
            lines.append("---")
            lines.append("")
            lines.append("## Notable Quotes")
            lines.append("")
            for quote in summary.notable_quotes:
                lines.append(f"> \"{quote.text}\"")
                if quote.context:
                    lines.append(f">")
                    lines.append(f"> *{quote.context}*")
                lines.append("")
        
        # Topics
        if summary.topics:
            lines.append("---")
            lines.append("")
            lines.append("## Topics")
            lines.append("")
            lines.append(" • ".join([f"`{t}`" for t in summary.topics]))
            lines.append("")
        
        # Action Items
        if summary.action_items:
            lines.append("---")
            lines.append("")
            lines.append("## Action Items")
            lines.append("")
            for item in summary.action_items:
                lines.append(f"- [ ] {item}")
            lines.append("")
        
        # Stats
        lines.append("---")
        lines.append("")
        lines.append("## Stats")
        lines.append("")
        lines.append(f"- Original: {summary.word_count_original} words")
        lines.append(f"- Summary: {summary.word_count_summary} words")
        lines.append(f"- Compression: {summary.compression_ratio:.1f}x")
        lines.append("")
        
        return "\n".join(lines)


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Summarize YouTube videos using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
  python main.py dQw4w9WgXcQ --style quick
  python main.py https://youtu.be/VIDEO_ID --output summary.md
  python main.py VIDEO_ID --style chapters --title "My Podcast"

Styles:
  detailed  - Full summary with chapters, quotes, takeaways (default)
  quick     - Just executive summary and key takeaways
  bullets   - Bullet points only
  chapters  - Focus on chapter breakdown
        """
    )
    
    parser.add_argument(
        "url",
        help="YouTube URL or video ID"
    )
    parser.add_argument(
        "--style", "-s",
        choices=["detailed", "quick", "bullets", "chapters"],
        default="detailed",
        help="Summary style (default: detailed)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: print to console)"
    )
    parser.add_argument(
        "--title", "-t",
        help="Custom title for the summary"
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Only fetch transcript, don't summarize"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎙️  PODCAST SUMMARIZER MVP")
    print("=" * 50)
    print("")
    
    # Transcript only mode
    if args.transcript_only:
        fetcher = YouTubeTranscriptFetcher()
        result = fetcher.fetch(args.url)
        
        if result.success:
            print(f"✅ Transcript fetched!")
            print(f"   Words: {result.word_count}")
            print(f"   Duration: {result.duration_minutes:.1f} min")
            print("")
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(result.full_text)
                print(f"📄 Saved to: {args.output}")
            else:
                print("--- TRANSCRIPT ---")
                print(result.full_text)
        else:
            print(f"❌ Failed: {result.error}")
            sys.exit(1)
        return
    
    # Full summarization
    app = PodcastSummarizerApp()
    result = app.summarize_url(
        url=args.url,
        style=args.style,
        title=args.title
    )
    
    if not result['success']:
        print(f"\n❌ Failed: {result['error']}")
        sys.exit(1)
    
    # Format output
    markdown = app.format_markdown(
        transcript=result['transcript'],
        summary=result['summary'],
        title=args.title or f"Video {result['transcript'].video_id}"
    )
    
    if args.output:
        # Save to file
        with open(args.output, 'w') as f:
            f.write(markdown)
        print(f"\n📄 Summary saved to: {args.output}")
    else:
        # Print to console
        print("\n" + "=" * 50)
        print(markdown)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
