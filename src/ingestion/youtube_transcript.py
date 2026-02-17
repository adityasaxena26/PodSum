"""
Podcast Summarizer MVP - YouTube Transcript Fetcher
====================================================
Fetches existing YouTube captions - FREE and instant!

No audio download, no Whisper, no GPU needed.
Just grab the transcript that YouTube already has.

Author: hobby-dev
Version: MVP 1.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from urllib.parse import urlparse, parse_qs
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

@dataclass
class TranscriptSegment:
    """Single segment of transcript with timing"""
    text: str
    start: float  # seconds
    duration: float
    
    @property
    def end(self) -> float:
        return self.start + self.duration
    
    def __str__(self) -> str:
        return f"[{self._format_time(self.start)}] {self.text}"
    
    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"


@dataclass 
class TranscriptResult:
    """Result from transcript fetch"""
    success: bool
    video_id: str = ""
    title: str = ""
    
    # Transcript data
    segments: List[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    
    # Metadata
    language: str = "en"
    is_auto_generated: bool = False
    duration_seconds: float = 0
    
    # Error info
    error: Optional[str] = None
    
    @property
    def word_count(self) -> int:
        return len(self.full_text.split())
    
    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60


class YouTubeTranscriptFetcher:
    """
    Fetch transcripts from YouTube videos.
    
    Uses youtube-transcript-api which:
    - Requires NO API key
    - Is completely FREE
    - Returns results instantly
    - Works with most YouTube videos
    
    Example:
        >>> fetcher = YouTubeTranscriptFetcher()
        >>> result = fetcher.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        >>> if result.success:
        ...     print(f"Got {result.word_count} words")
        ...     print(result.full_text[:500])
    """
    
    def __init__(self, preferred_languages: List[str] = None):
        """
        Initialize fetcher.
        
        Args:
            preferred_languages: List of language codes in preference order.
                                 Default: ['en', 'en-US', 'en-GB']
        """
        self.preferred_languages = preferred_languages or ['en', 'en-US', 'en-GB']
        self._api = None
    
    def _get_api(self):
        """Lazy load the API"""
        if self._api is None:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                self._api = YouTubeTranscriptApi()
            except ImportError:
                raise ImportError(
                    "youtube-transcript-api not installed. "
                    "Run: pip install youtube-transcript-api"
                )
        return self._api
    
    def fetch(self, url_or_id: str) -> TranscriptResult:
        """
        Fetch transcript from YouTube video.
        
        Args:
            url_or_id: YouTube URL or video ID
                       Supports: youtube.com/watch?v=ID, youtu.be/ID, or just ID
        
        Returns:
            TranscriptResult with transcript data or error
        """
        # Extract video ID
        video_id = self._extract_video_id(url_or_id)
        if not video_id:
            return TranscriptResult(
                success=False,
                error=f"Could not extract video ID from: {url_or_id}"
            )
        
        api = self._get_api()
        
        try:
            # Get available transcripts
            transcript_list = api.list(video_id)
            
            # Try to find best transcript
            transcript, is_auto = self._select_best_transcript(transcript_list)
            
            if transcript is None:
                return TranscriptResult(
                    success=False,
                    video_id=video_id,
                    error="No transcript available for this video"
                )
            
            # Fetch transcript segments
            segments_raw = transcript.fetch()

            # Convert to our format
            segments = [
                TranscriptSegment(
                    text=seg.text,
                    start=seg.start,
                    duration=seg.duration
                )
                for seg in segments_raw
            ]
            
            # Build full text
            full_text = ' '.join(seg.text for seg in segments)
            
            # Clean up text (remove artifacts)
            full_text = self._clean_text(full_text)
            
            # Calculate duration
            duration = segments[-1].end if segments else 0
            
            return TranscriptResult(
                success=True,
                video_id=video_id,
                segments=segments,
                full_text=full_text,
                language=transcript.language_code,
                is_auto_generated=is_auto,
                duration_seconds=duration
            )
        
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages
            if "Subtitles are disabled" in error_msg:
                error_msg = "Subtitles/captions are disabled for this video"
            elif "No transcript" in error_msg:
                error_msg = "No transcript available for this video"
            elif "Video unavailable" in error_msg:
                error_msg = "Video is unavailable (private, deleted, or restricted)"
            
            return TranscriptResult(
                success=False,
                video_id=video_id,
                error=error_msg
            )
    
    def _extract_video_id(self, url_or_id: str) -> Optional[str]:
        """
        Extract YouTube video ID from various formats.
        
        Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - VIDEO_ID (direct)
        """
        url_or_id = url_or_id.strip()
        
        # Check if it's already just an ID (11 characters, alphanumeric + dash/underscore)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
            return url_or_id
        
        # Parse URL
        parsed = urlparse(url_or_id)
        
        # youtube.com/watch?v=ID
        if 'youtube.com' in parsed.netloc:
            if '/watch' in parsed.path:
                query = parse_qs(parsed.query)
                video_ids = query.get('v', [])
                if video_ids:
                    return video_ids[0]
            # youtube.com/embed/ID
            elif '/embed/' in parsed.path:
                return parsed.path.split('/embed/')[-1].split('/')[0]
        
        # youtu.be/ID
        if 'youtu.be' in parsed.netloc:
            return parsed.path.strip('/')
        
        return None
    
    def _select_best_transcript(self, transcript_list) -> tuple:
        """
        Select best available transcript.
        
        Priority:
        1. Manual transcript in preferred language
        2. Auto-generated in preferred language
        3. Any manual transcript
        4. Any auto-generated transcript
        
        Returns:
            (transcript, is_auto_generated)
        """
        manual_transcripts = []
        auto_transcripts = []
        
        for transcript in transcript_list:
            if transcript.is_generated:
                auto_transcripts.append(transcript)
            else:
                manual_transcripts.append(transcript)
        
        # Try manual transcripts first
        for lang in self.preferred_languages:
            for t in manual_transcripts:
                if t.language_code.startswith(lang.split('-')[0]):
                    return (t, False)
        
        # Try auto-generated
        for lang in self.preferred_languages:
            for t in auto_transcripts:
                if t.language_code.startswith(lang.split('-')[0]):
                    return (t, True)
        
        # Fall back to any available
        if manual_transcripts:
            return (manual_transcripts[0], False)
        if auto_transcripts:
            return (auto_transcripts[0], True)
        
        return (None, False)
    
    def _clean_text(self, text: str) -> str:
        """Clean transcript text of common artifacts"""
        # Remove [Music], [Applause], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def fetch_with_timestamps(self, url_or_id: str) -> TranscriptResult:
        """
        Fetch transcript and format with timestamps.
        
        Same as fetch() but full_text includes timestamps.
        """
        result = self.fetch(url_or_id)
        
        if result.success:
            # Rebuild full_text with timestamps
            timestamped_lines = [str(seg) for seg in result.segments]
            result.full_text = '\n'.join(timestamped_lines)
        
        return result


# Convenience function
def get_youtube_transcript(url: str) -> TranscriptResult:
    """
    Quick function to get YouTube transcript.
    
    Args:
        url: YouTube URL or video ID
    
    Returns:
        TranscriptResult
    
    Example:
        >>> result = get_youtube_transcript("https://youtube.com/watch?v=...")
        >>> if result.success:
        ...     print(result.full_text)
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.fetch(url)


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("YouTube Transcript Fetcher")
        print("=" * 40)
        print("\nUsage: python youtube_transcript.py <URL or VIDEO_ID>")
        print("\nExamples:")
        print("  python youtube_transcript.py https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  python youtube_transcript.py dQw4w9WgXcQ")
        print("  python youtube_transcript.py https://youtu.be/dQw4w9WgXcQ")
        sys.exit(1)
    
    url = sys.argv[1]
    print(f"Fetching transcript for: {url}\n")
    
    fetcher = YouTubeTranscriptFetcher()
    result = fetcher.fetch(url)
    
    if result.success:
        print("✅ SUCCESS!")
        print(f"   Video ID: {result.video_id}")
        print(f"   Language: {result.language}")
        print(f"   Auto-generated: {result.is_auto_generated}")
        print(f"   Duration: {result.duration_minutes:.1f} minutes")
        print(f"   Word count: {result.word_count}")
        print(f"   Segments: {len(result.segments)}")
        print("\n" + "=" * 40)
        print("TRANSCRIPT (first 1000 characters):")
        print("=" * 40)
        print(result.full_text[:1000])
        if len(result.full_text) > 1000:
            print(f"\n... [{len(result.full_text) - 1000} more characters]")
    else:
        print(f"❌ FAILED: {result.error}")
