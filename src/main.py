"""
Podcast Summarizer v2.0 - Main Application
==========================================
Multi-platform video/podcast summarizer with intelligent fallback.

Supports:
- YouTube, Vimeo, Twitter, TikTok, Spotify, and 1000+ more sites
- Local audio/video files
- Automatic content type detection
- Multiple output formats

Usage:
    python main.py <URL>
    python main.py <URL> --format quick
    python main.py --file audio.mp3
    python main.py <URL> --output summary.md

Version: 2.0.0
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.multi_platform import (
    MultiPlatformFetcher, 
    TranscriptResult,
    TranscriptSource
)
from src.summarization.summarizer import (
    EnhancedSummarizer,
    Summary,
    SummaryFormat,
    ContentType
)

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

class PodcastSummarizerV2:
    """
    Main application class for Podcast Summarizer v2.

    Unified interface for:
    - URL-based content (YouTube, Spotify, etc.)
    - Local files (audio/video)

    Example:
        >>> app = PodcastSummarizerV2()
        >>> result = app.summarize_url("https://youtube.com/watch?v=...")
        >>> print(result['summary'].executive_summary)
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        provider: str = "auto",
        whisper_model: str = "small"
    ):
        """
        Initialize the application.

        Args:
            groq_api_key: Groq API key (or set GROQ_API_KEY env var)
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
            provider: "gemini", "groq", or "auto"
            whisper_model: Whisper model for audio fallback
        """
        self.fetcher = MultiPlatformFetcher(whisper_model=whisper_model)
        self.summarizer = EnhancedSummarizer(
            api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            provider=provider,
        )
    
    def summarize_url(
        self,
        url: str,
        format: str = "detailed",
        title: Optional[str] = None,
        force_audio: bool = False,
        progress_callback=None
    ) -> dict:
        """
        Summarize content from URL.
        
        Args:
            url: URL to video/podcast
            format: Summary format ("detailed", "quick", "bullets", "chapters")
            title: Optional title override
            force_audio: Skip transcript APIs, use audio
            progress_callback: Optional callback for progress updates
        
        Returns:
            dict with 'success', 'transcript', 'summary', 'error'
        """
        # Step 1: Fetch transcript
        if progress_callback:
            progress_callback("Fetching transcript...", 0.1)
        
        transcript_result = self.fetcher.fetch(
            url,
            force_audio=force_audio,
            progress_callback=progress_callback
        )
        
        if not transcript_result.success:
            return {
                'success': False,
                'error': f"Transcript fetch failed: {transcript_result.error}",
                'transcript': transcript_result,
                'summary': None
            }
        
        logger.info(
            f"Got transcript via {transcript_result.source.value}: "
            f"{transcript_result.word_count} words"
        )
        
        # Step 2: Summarize
        if progress_callback:
            progress_callback("Generating summary...", 0.6)
        
        format_map = {
            "detailed": SummaryFormat.DETAILED,
            "quick": SummaryFormat.QUICK,
            "bullets": SummaryFormat.BULLETS,
            "chapters": SummaryFormat.CHAPTERS,
        }
        
        summary_result = self.summarizer.summarize(
            transcript=transcript_result.text,
            title=title or transcript_result.title or "Untitled",
            format=format_map.get(format, SummaryFormat.DETAILED),
            duration_minutes=transcript_result.duration_minutes
        )
        
        if not summary_result.success:
            return {
                'success': False,
                'error': f"Summary failed: {summary_result.error}",
                'transcript': transcript_result,
                'summary': summary_result
            }
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        return {
            'success': True,
            'transcript': transcript_result,
            'summary': summary_result
        }
    
    def summarize_file(
        self,
        file_path: str,
        format: str = "detailed",
        title: Optional[str] = None,
        progress_callback=None
    ) -> dict:
        """
        Summarize a local audio/video file.
        
        Args:
            file_path: Path to file
            format: Summary format
            title: Optional title
            progress_callback: Progress callback
        
        Returns:
            dict with 'success', 'transcript', 'summary', 'error'
        """
        if progress_callback:
            progress_callback("Transcribing file...", 0.1)
        
        transcript_result = self.fetcher.transcribe_file(
            file_path,
            progress_callback=progress_callback
        )
        
        if not transcript_result.success:
            return {
                'success': False,
                'error': f"Transcription failed: {transcript_result.error}",
                'transcript': transcript_result,
                'summary': None
            }
        
        if progress_callback:
            progress_callback("Generating summary...", 0.7)
        
        format_map = {
            "detailed": SummaryFormat.DETAILED,
            "quick": SummaryFormat.QUICK,
            "bullets": SummaryFormat.BULLETS,
            "chapters": SummaryFormat.CHAPTERS,
        }
        
        summary_result = self.summarizer.summarize(
            transcript=transcript_result.text,
            title=title or Path(file_path).stem,
            format=format_map.get(format, SummaryFormat.DETAILED),
            duration_minutes=transcript_result.duration_minutes
        )
        
        if not summary_result.success:
            return {
                'success': False,
                'error': f"Summary failed: {summary_result.error}",
                'transcript': transcript_result,
                'summary': summary_result
            }
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        return {
            'success': True,
            'transcript': transcript_result,
            'summary': summary_result
        }
    
    def format_markdown(
        self,
        transcript: TranscriptResult,
        summary: Summary
    ) -> str:
        """Format results as markdown"""
        
        lines = []
        
        # Header
        lines.append(f"# {summary.title or 'Summary'}")
        lines.append("")
        
        # Metadata
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Source:** {transcript.source.value}")
        lines.append(f"- **Platform:** {transcript.platform}")
        lines.append(f"- **Duration:** {transcript.duration_minutes:.1f} minutes")
        lines.append(f"- **Content Type:** {summary.content_type.value}")
        lines.append(f"- **Words:** {summary.word_count_original:,}")
        lines.append(f"- **Compression:** {summary.compression_ratio:.1f}x")
        lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # Executive Summary
        if summary.executive_summary:
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
            for i, t in enumerate(summary.key_takeaways, 1):
                lines.append(f"{i}. {t}")
            lines.append("")
        
        # Chapters
        if summary.chapters:
            lines.append("---")
            lines.append("")
            lines.append("## Chapters")
            lines.append("")
            for ch in summary.chapters:
                lines.append(f"### [{ch.timestamp}] {ch.title}")
                lines.append("")
                lines.append(ch.summary)
                lines.append("")
        
        # Notable Quotes
        if summary.notable_quotes:
            lines.append("---")
            lines.append("")
            lines.append("## Notable Quotes")
            lines.append("")
            for q in summary.notable_quotes:
                lines.append(f"> \"{q.text}\"")
                if q.speaker:
                    lines.append(f"> — {q.speaker}")
                if q.context:
                    lines.append(f">")
                    lines.append(f"> *{q.context}*")
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
        
        return "\n".join(lines)
    
    def format_json(
        self,
        transcript: TranscriptResult,
        summary: Summary
    ) -> dict:
        """Format results as JSON-serializable dict"""
        return {
            'metadata': {
                'title': summary.title,
                'source': transcript.source.value,
                'platform': transcript.platform,
                'url': transcript.url,
                'duration_minutes': transcript.duration_minutes,
                'content_type': summary.content_type.value,
                'word_count': summary.word_count_original,
                'compression_ratio': summary.compression_ratio,
                'language': transcript.language,
            },
            'summary': {
                'executive_summary': summary.executive_summary,
                'key_takeaways': summary.key_takeaways,
                'chapters': [
                    {'title': c.title, 'timestamp': c.timestamp, 'summary': c.summary}
                    for c in summary.chapters
                ],
                'notable_quotes': [
                    {'text': q.text, 'speaker': q.speaker, 'context': q.context}
                    for q in summary.notable_quotes
                ],
                'topics': summary.topics,
                'action_items': summary.action_items,
            },
            'transcript': {
                'text': transcript.text,
                'word_count': transcript.word_count,
            }
        }


def main():
    """CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="Summarize videos and podcasts from any platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YouTube video
  python main.py https://www.youtube.com/watch?v=VIDEO_ID
  
  # Quick summary
  python main.py https://youtube.com/watch?v=ID --format quick
  
  # Local file
  python main.py --file podcast.mp3
  
  # Save output
  python main.py URL --output summary.md
  
  # Force audio transcription (skip captions)
  python main.py URL --force-audio

Supported Platforms:
  YouTube, Vimeo, Twitter/X, TikTok, Spotify, SoundCloud,
  Twitch, Facebook, Instagram, and 1000+ more via yt-dlp

Formats:
  detailed  - Full summary with chapters, quotes (default)
  quick     - Executive summary + takeaways only
  bullets   - Key points as bullet list
  chapters  - Chapter/timeline breakdown

Environment Variables:
  GROQ_API_KEY  - Your Groq API key (required)
        """
    )
    
    parser.add_argument(
        "url",
        nargs="?",
        help="URL to video/podcast"
    )
    parser.add_argument(
        "--file", "-f",
        help="Local audio/video file path"
    )
    parser.add_argument(
        "--format",
        choices=["detailed", "quick", "bullets", "chapters"],
        default="detailed",
        help="Summary format (default: detailed)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (.md or .json)"
    )
    parser.add_argument(
        "--title", "-t",
        help="Custom title"
    )
    parser.add_argument(
        "--force-audio",
        action="store_true",
        help="Force audio download (skip transcript APIs)"
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Only fetch transcript, don't summarize"
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model for audio (default: small)"
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "groq"],
        default="auto",
        help="LLM provider (default: auto - prefers Gemini if key available)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.url and not args.file:
        parser.print_help()
        print("\nError: Provide a URL or --file")
        sys.exit(1)

    # Check API key — need at least one
    if not args.transcript_only and not groq_api_key and not gemini_api_key:
        print("⚠️  No API key set!")
        print("")
        print("Option 1 (recommended):")
        print("  1. Go to https://aistudio.google.com/apikey")
        print("  2. Create a free Gemini API key")
        print("  3. Run: export GEMINI_API_KEY='your-key'")
        print("")
        print("Option 2:")
        print("  1. Go to https://console.groq.com/keys")
        print("  2. Create a free Groq API key")
        print("  3. Run: export GROQ_API_KEY='your-key'")
        print("")
        print("Or use --transcript-only to just get transcript.")
        sys.exit(1)
    
    # Banner
    if not args.quiet:
        print("=" * 60)
        print("🎙️  PODCAST SUMMARIZER v2.0")
        print("   Multi-platform • Audio fallback • Smart summaries")
        print("=" * 60)
        print()
    
    # Progress callback
    def progress(msg, pct):
        if not args.quiet:
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            print(f"\r[{bar}] {pct*100:5.1f}% - {msg:<30}", end="", flush=True)
            if pct >= 1.0:
                print()
    
    # Initialize app
    app = PodcastSummarizerV2(
        whisper_model=args.whisper_model,
        provider=args.provider,
    )
    
    # Transcript only mode
    if args.transcript_only:
        if args.file:
            result = app.fetcher.transcribe_file(args.file, progress_callback=progress)
        else:
            result = app.fetcher.fetch(args.url, progress_callback=progress)
        
        print()
        if result.success:
            print(f"✅ Transcript fetched ({result.source.value})")
            print(f"   Words: {result.word_count:,}")
            print(f"   Duration: {result.duration_minutes:.1f} min")
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(result.text)
                print(f"   Saved to: {args.output}")
            else:
                print("\n" + "=" * 40)
                print(result.text[:2000])
                if len(result.text) > 2000:
                    print(f"\n... [{len(result.text) - 2000:,} more characters]")
        else:
            print(f"❌ Failed: {result.error}")
            sys.exit(1)
        return
    
    # Full summarization
    if args.file:
        result = app.summarize_file(
            args.file,
            format=args.format,
            title=args.title,
            progress_callback=progress
        )
    else:
        result = app.summarize_url(
            args.url,
            format=args.format,
            title=args.title,
            force_audio=args.force_audio,
            progress_callback=progress
        )
    
    print()
    
    if not result['success']:
        print(f"❌ Failed: {result['error']}")
        sys.exit(1)
    
    transcript = result['transcript']
    summary = result['summary']
    
    # Output
    if args.output:
        if args.output.endswith('.json'):
            import json
            data = app.format_json(transcript, summary)
            with open(args.output, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            markdown = app.format_markdown(transcript, summary)
            with open(args.output, 'w') as f:
                f.write(markdown)
        
        if not args.quiet:
            print(f"✅ Saved to: {args.output}")
    else:
        # Print to console
        markdown = app.format_markdown(transcript, summary)
        print()
        print(markdown)
    
    # Stats
    if not args.quiet:
        print()
        print("=" * 40)
        print(f"📊 Source: {transcript.source.value}")
        print(f"📊 Content type: {summary.content_type.value}")
        print(f"📊 Compression: {summary.compression_ratio:.1f}x")
        print("✅ Done!")


if __name__ == "__main__":
    main()
