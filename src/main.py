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
import hashlib
import time
import concurrent.futures
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
    ContentType,
    Chapter,
    Quote
)

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Centralized model config — change here or via env vars
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

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
        gemini_api_key: Optional[str] = None,
        whisper_model: str = "small"
    ):
        """
        Initialize the application.

        Args:
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
            whisper_model: Whisper model for audio fallback
        """
        self.fetcher = MultiPlatformFetcher(whisper_model=whisper_model)
        self.summarizer = EnhancedSummarizer(
            gemini_api_key=gemini_api_key,
        )
        # Simple result cache: {cache_key: result_dict}
        # Avoids repeat API calls for same URL + format
        self._cache = {}
        self._cache_max = 32
    
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
        summary_format = SummaryFormat.from_string(format)

        # Check cache (same URL + format = same result)
        cache_key = hashlib.md5(f"{url}|{format}|{force_audio}".encode()).hexdigest()
        if cache_key in self._cache:
            logger.info(f"Cache hit for {url}")
            if progress_callback:
                progress_callback("Complete! (cached)", 1.0)
            return self._cache[cache_key]

        # Fast path: Combined Gemini transcribe + summarize in ONE API call.
        # When the URL is YouTube and Gemini is available, this avoids two
        # separate expensive Gemini calls (transcribe → summarize).
        _cached_transcript = None  # May be populated by fast path for reuse
        _gemini_error = None       # Capture fast-path error for diagnostics
        if (
            not force_audio
            and self._is_youtube_url(url)
            and os.environ.get('GEMINI_API_KEY')
        ):
            if progress_callback:
                progress_callback("Transcribing & summarizing via Gemini...", 0.1)
            result = self._combined_gemini(url, summary_format, title, progress_callback)
            if result and result['success']:
                self._cache_put(cache_key, result)
                return result
            # Preserve any transcript fetched during the fast path so we
            # don't re-fetch it in the 2-step fallback below.
            if result and result.get('transcript') and result['transcript'].success:
                _cached_transcript = result['transcript']
            if result and result.get('error'):
                _gemini_error = result['error']
            logger.info("Combined Gemini path failed, falling back to 2-step pipeline")

        # Standard 2-step path: fetch transcript, then summarize separately.
        # Step 1: Fetch transcript (skip if already fetched by fast path)
        if _cached_transcript:
            transcript_result = _cached_transcript
            logger.info(f"Reusing transcript from fast path: {transcript_result.word_count} words")
        else:
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

        summary_result = self.summarizer.summarize(
            transcript=transcript_result.text,
            title=title or transcript_result.title or "Untitled",
            format=summary_format,
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

        result = {
            'success': True,
            'transcript': transcript_result,
            'summary': summary_result
        }
        self._cache_put(cache_key, result)
        return result

    # =================================================================
    # Combined Gemini fast path (single API call)
    # =================================================================

    def _cache_put(self, key: str, value: dict):
        """Store result in cache, evicting oldest if full."""
        if len(self._cache) >= self._cache_max:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        """Check if the URL is a YouTube video."""
        from urllib.parse import urlparse
        domain = urlparse(url.lower()).netloc
        return any(d in domain for d in ['youtube.com', 'youtu.be'])

    # Minimum expected words-per-minute for a valid transcript.
    # Normal speech is 130-170 wpm. Below this threshold, captions
    # are considered too sparse and we fall back to Gemini video mode.
    _MIN_WORDS_PER_MINUTE = 50

    def _combined_gemini(
        self,
        url: str,
        summary_format: SummaryFormat,
        title: Optional[str],
        progress_callback=None,
    ) -> Optional[dict]:
        """
        Smart fast path for YouTube summarization.

        Path A — transcript API returns a GOOD transcript (>50 wpm):
          youtube-transcript-api (<1s) → Gemini text summary (~3-6s)
        Path A-Video — good transcript but very long (>90min):
          Gemini native video understanding (~20-30s)
        Path B — no transcript OR sparse captions:
          single Gemini video call (~20-30s) → summary + notes

        Speed optimizations:
          - Parallel transcript + metadata fetch
          - response_mime_type="application/json" (native JSON, no parsing retries)
          - Concise prompts (fewer input tokens = faster)
          - Word-count targets for consistent summary depth
        """
        import json, re

        try:
            from google.genai import types
        except ImportError:
            return None

        if not os.environ.get('GEMINI_API_KEY'):
            return None

        video_id = self.fetcher._extract_youtube_id(url)
        if not video_id:
            return None

        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        t0 = time.time()

        # ── Step 1: Fetch transcript + metadata IN PARALLEL ─────────
        if progress_callback:
            progress_callback("Fetching transcript & metadata...", 0.05)

        yt_transcript = None
        metadata = {}
        yt_api_key = os.environ.get('YOUTUBE_API_KEY')

        def _fetch_transcript():
            return self.fetcher._fetch_youtube_transcript(url)

        def _fetch_metadata():
            if yt_api_key:
                return self.fetcher._yt_data_api_metadata(video_id, yt_api_key)
            return {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ft_transcript = executor.submit(_fetch_transcript)
            ft_metadata = executor.submit(_fetch_metadata)
            try:
                yt_transcript = ft_transcript.result(timeout=15)
                if not (yt_transcript and yt_transcript.success):
                    yt_transcript = None
            except Exception as e:
                logger.info(f"Captions unavailable: {e}")
            try:
                metadata = ft_metadata.result(timeout=10) or {}
            except Exception:
                metadata = {}

        t_fetch = time.time() - t0
        if yt_transcript:
            logger.info(f"Captions: {yt_transcript.word_count:,} words (fetched in {t_fetch:.1f}s)")

        # Use title from: explicit param > metadata > transcript result > fallback
        yt_title = yt_transcript.title if yt_transcript and yt_transcript.title else ''
        video_title = title or metadata.get('title', '') or yt_title or 'Untitled'
        duration = metadata.get('duration', 0)
        # Also check transcript duration if metadata has none
        if not duration and yt_transcript and yt_transcript.duration_seconds > 0:
            duration = yt_transcript.duration_seconds
        duration_min = duration / 60 if duration else 0

        # ── Step 2: Validate transcript quality ─────────────────────
        transcript_is_good = False
        if yt_transcript:
            if duration_min > 0:
                wpm = yt_transcript.word_count / duration_min
                if wpm >= self._MIN_WORDS_PER_MINUTE:
                    transcript_is_good = True
                    logger.info(f"Transcript quality: {wpm:.0f} wpm ✓")
                else:
                    logger.warning(
                        f"Sparse transcript: {yt_transcript.word_count} words / "
                        f"{duration_min:.1f}min = {wpm:.0f} wpm (need ≥{self._MIN_WORDS_PER_MINUTE})"
                    )
            else:
                estimated_dur = yt_transcript.word_count / 150.0
                if yt_transcript.word_count >= 200:
                    transcript_is_good = True
                    duration_min = estimated_dur
                    logger.info(f"No metadata — estimated {duration_min:.1f}min from word count")
                else:
                    logger.warning(f"Too short ({yt_transcript.word_count} words), no duration")

        # ── Step 3: Detect content type ─────────────────────────────
        if transcript_is_good:
            content_type = self.summarizer._detect_content_type(
                yt_transcript.text, video_title
            )
        else:
            content_type = EnhancedSummarizer.detect_content_type_from_title(video_title)
        logger.info(f"Content type: {content_type.value}")

        schema = EnhancedSummarizer.get_format_schema(summary_format, content_type)
        client = self.summarizer._get_gemini_client()
        MODEL = GEMINI_MODEL

        # Target summary word count based on duration for quality consistency
        target_words = max(400, int(duration_min * 15)) if duration_min > 0 else 500

        try:
            t_llm = time.time()

            # ── Path A: Good transcript → text-only (~3-6s) ────────
            if transcript_is_good:
                use_video_for_long = (
                    yt_transcript.word_count > 15000 and duration_min > 90
                )

                if use_video_for_long:
                    # Path A-Video: very long content → Gemini video mode
                    logger.info(
                        f"Path A-Video: {yt_transcript.word_count:,} words / "
                        f"{duration_min:.0f}min → video mode"
                    )
                    if progress_callback:
                        progress_callback("AI analyzing video...", 0.2)

                    video_part = types.Part.from_uri(
                        file_uri=canonical_url, mime_type='video/*',
                    )
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=[
                            video_part,
                            f"""Summarize this {content_type.value}. Return JSON only — no markdown.

TITLE: {video_title} | DURATION: {duration_min:.0f}min

{schema}

TARGET: ~{target_words} words total. Cover the ENTIRE {duration_min:.0f} minutes — not just the beginning. Use specific names, numbers, claims. Quotes must be exact words spoken.""",
                        ],
                        config=types.GenerateContentConfig(
                            # NOTE: response_mime_type is NOT used with video input —
                            # structured output is incompatible with multimodal content.
                            max_output_tokens=8192,
                            temperature=0.1,
                        ),
                    )
                    summary_raw = resp.text.strip()
                    transcript_text = yt_transcript.text
                    transcript_source = yt_transcript.source
                    word_count = yt_transcript.word_count

                else:
                    # Path A: text transcript → fast text summary
                    logger.info(f"Path A: {yt_transcript.word_count:,} words → {MODEL}")
                    if progress_callback:
                        progress_callback("Summarizing with AI...", 0.3)

                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=f"""Summarize this {content_type.value}. TITLE: {video_title}{f" | DURATION: {duration_min:.0f}min" if duration_min else ""}

---TRANSCRIPT---
{yt_transcript.text}
---END---

{schema}

TARGET: ~{target_words} words total. Be specific — use real names, numbers, claims from the transcript. Quotes must be exact words spoken.""",
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            max_output_tokens=8192,
                            temperature=0.1,
                        ),
                    )
                    summary_raw = resp.text.strip()
                    transcript_text = yt_transcript.text
                    transcript_source = yt_transcript.source
                    word_count = yt_transcript.word_count

            # ── Path B: No/sparse transcript → Gemini video ────────
            else:
                if progress_callback:
                    progress_callback("AI watching the video...", 0.15)
                logger.info(f"Path B: video → {MODEL}")

                video_part = types.Part.from_uri(
                    file_uri=canonical_url, mime_type='video/*',
                )
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        video_part,
                        f"""Summarize this video. Return JSON only — no markdown.

TITLE: {video_title}{f" | DURATION: {duration_min:.0f}min" if duration_min else ""}

{schema}

TARGET: ~{target_words} words total. Cover the ENTIRE video. Use specific names, numbers, claims. Quotes must be exact words spoken.""",
                    ],
                    config=types.GenerateContentConfig(
                        # NOTE: response_mime_type is NOT used with video input —
                        # structured output is incompatible with multimodal content.
                        max_output_tokens=8192,
                        temperature=0.1,
                    ),
                )
                summary_raw = resp.text.strip()
                transcript_text = None
                transcript_source = TranscriptSource.YOUTUBE_CAPTIONS
                word_count = 0

            t_llm_done = time.time() - t_llm
            logger.info(f"LLM response in {t_llm_done:.1f}s")

            if progress_callback:
                progress_callback("Parsing response...", 0.85)

            # ── Parse JSON ──────────────────────────────────────────
            # Text path uses response_mime_type="application/json" (clean JSON).
            # Video paths can't use it, so may have markdown fences.
            raw = summary_raw
            if raw.startswith('```'):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)
            summary_data = json.loads(raw)

            # Path B: build transcript from chapters/summary (no verbatim available)
            if transcript_text is None:
                parts = []
                exec_sum = summary_data.get('executive_summary', '')
                if exec_sum:
                    parts.append(f"OVERVIEW\n{exec_sum}")
                for ch in summary_data.get('chapters', []):
                    if isinstance(ch, dict):
                        ts = ch.get('timestamp', '')
                        title_ch = ch.get('title', '')
                        summ = ch.get('summary', '')
                        parts.append(f"[{ts}] {title_ch}\n{summ}")
                for q in summary_data.get('notable_quotes', []):
                    if isinstance(q, dict):
                        speaker = q.get('speaker', '')
                        text_q = q.get('text', '')
                        parts.append(f'"{text_q}" — {speaker}')
                transcript_text = "\n\n".join(parts) if parts else exec_sum
                word_count = len(transcript_text.split())

            # ── Build result objects ────────────────────────────────
            transcript_result = TranscriptResult(
                success=True,
                source=transcript_source,
                text=transcript_text,
                segments=[],
                title=video_title,
                duration_seconds=duration,
                language="en",
                is_auto_generated=False,
                platform="youtube",
                url=canonical_url,
            )

            # Count ALL summary output words (not just executive_summary)
            # for an accurate compression ratio
            summary_word_count = len(summary_data.get('executive_summary', '').split())
            for t in summary_data.get('key_takeaways', []):
                summary_word_count += len(str(t).split())
            for ch in summary_data.get('chapters', []):
                if isinstance(ch, dict):
                    summary_word_count += len(ch.get('summary', '').split())
                    summary_word_count += len(ch.get('title', '').split())
            for q in summary_data.get('notable_quotes', []):
                if isinstance(q, dict):
                    summary_word_count += len(q.get('text', '').split())

            summary_result = Summary(
                success=True,
                content_type=content_type,
                executive_summary=summary_data.get('executive_summary', ''),
                key_takeaways=summary_data.get('key_takeaways', []),
                chapters=[],
                notable_quotes=[],
                topics=summary_data.get('topics', []),
                action_items=summary_data.get('action_items', []),
                word_count_original=word_count,
                word_count_summary=summary_word_count,
                duration_minutes=duration_min,
                title=video_title,
            )

            for ch in summary_data.get('chapters', []):
                if isinstance(ch, dict):
                    summary_result.chapters.append(Chapter(
                        title=ch.get('title', ''),
                        timestamp=ch.get('timestamp', ''),
                        summary=ch.get('summary', ''),
                    ))

            for q in summary_data.get('notable_quotes', []):
                if isinstance(q, dict):
                    summary_result.notable_quotes.append(Quote(
                        text=q.get('text', ''),
                        speaker=q.get('speaker', ''),
                        context=q.get('context', ''),
                    ))

            if progress_callback:
                progress_callback("Complete!", 1.0)

            t_total = time.time() - t0
            logger.info(
                f"✅ Done in {t_total:.1f}s: {word_count:,} words → "
                f"{summary_word_count} word summary, "
                f"{len(summary_result.key_takeaways)} takeaways, "
                f"{len(summary_result.chapters)} chapters"
            )

            return {
                'success': True,
                'transcript': transcript_result,
                'summary': summary_result,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            logger.warning(f"Raw response (first 500 chars): {summary_raw[:500] if summary_raw else 'empty'}")
            return None
        except Exception as e:
            logger.error(f"Combined Gemini failed: {type(e).__name__}: {e}")
            return None

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
        
        summary_result = self.summarizer.summarize(
            transcript=transcript_result.text,
            title=title or Path(file_path).stem,
            format=SummaryFormat.from_string(format),
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

        # Title
        lines.append(f"# {summary.title or 'Summary'}")
        lines.append("")

        # Executive Summary first (the most important content)
        if summary.executive_summary:
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
            lines.append(" ".join([f"`{t}`" for t in summary.topics]))
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

        # Metadata at the bottom (reference info, not primary content)
        lines.append("---")
        lines.append("")
        lines.append(
            f"*{transcript.source.value} | {transcript.platform} | "
            f"{transcript.duration_minutes:.1f} min | "
            f"{summary.word_count_original:,} words | "
            f"{summary.compression_ratio:.1f}x compression | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        )
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
  GEMINI_API_KEY  - Your Gemini API key (required)
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

    # Check API key
    if not args.transcript_only and not gemini_api_key:
        print("⚠️  No API key set!")
        print("")
        print("  1. Go to https://aistudio.google.com/apikey")
        print("  2. Create a free Gemini API key")
        print("  3. Run: export GEMINI_API_KEY='your-key'")
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
