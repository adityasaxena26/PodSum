"""
Podcast Summarizer v2.0 - Multi-Platform Transcript Fetcher
============================================================
Smart transcript acquisition with tiered fallback:

TIER 1: Platform transcripts (FREE, instant)
    - YouTube captions
    - (Future: Spotify, Apple Podcasts, RSS feeds)

TIER 2: Audio download + Whisper transcription (fallback)
    - Supports 1000+ sites via yt-dlp
    - Uses faster-whisper for efficient transcription
llama-3.1-70b-versatile
Version: 2.0.0
"""

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TranscriptSource(Enum):
    """Where the transcript came from"""
    YOUTUBE_CAPTIONS = "youtube_captions"
    SPOTIFY = "spotify"
    RSS_FEED = "rss_feed"
    WHISPER = "whisper"
    USER_PROVIDED = "user_provided"


@dataclass
class TranscriptSegment:
    """Single segment with timing"""
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptResult:
    """Result from transcript acquisition"""
    success: bool
    source: TranscriptSource = TranscriptSource.YOUTUBE_CAPTIONS
    
    # Content
    text: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)
    
    # Metadata
    title: str = ""
    duration_seconds: float = 0
    language: str = "en"
    is_auto_generated: bool = False
    platform: str = ""
    url: str = ""
    
    # Error
    error: Optional[str] = None
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60


class Platform(Enum):
    """Supported platforms"""
    YOUTUBE = "youtube"
    SPOTIFY = "spotify"
    APPLE_PODCASTS = "apple_podcasts"
    SOUNDCLOUD = "soundcloud"
    VIMEO = "vimeo"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    TWITCH = "twitch"
    FACEBOOK = "facebook"
    GENERIC = "generic"  # Any other site yt-dlp supports


class MultiPlatformFetcher:
    """
    Smart transcript fetcher with multi-platform support.
    
    Strategy:
    1. Detect platform from URL
    2. Try platform-specific transcript API (FREE, instant)
    3. Fall back to audio download + Whisper if needed
    
    Supports:
    - YouTube (captions API + audio fallback)
    - Spotify, Vimeo, Twitter, TikTok, etc. (audio fallback)
    - 1000+ sites via yt-dlp
    
    Example:
        >>> fetcher = MultiPlatformFetcher()
        >>> result = fetcher.fetch("https://youtube.com/watch?v=...")
        >>> print(f"Source: {result.source}, Words: {result.word_count}")
    """
    
    def __init__(
        self,
        whisper_model: str = "small",
        whisper_device: str = "auto",
        prefer_transcripts: bool = True,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize fetcher.
        
        Args:
            whisper_model: Whisper model size for audio fallback
                          Options: tiny, base, small, medium, large-v2, large-v3
            whisper_device: Device for Whisper - "cuda", "cpu", or "auto"
            prefer_transcripts: Try platform transcripts before audio
            temp_dir: Directory for temporary audio files
        """
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.prefer_transcripts = prefer_transcripts
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Lazy-loaded components
        self._whisper = None
        self._yt_dlp = None
    
    def fetch(
        self,
        url: str,
        force_audio: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> TranscriptResult:
        """
        Fetch transcript from URL using best available method.
        
        Args:
            url: URL to video/podcast
            force_audio: Skip transcript APIs, go straight to audio
            progress_callback: Optional callback for progress updates
                              Signature: callback(step: str, progress: float)
        
        Returns:
            TranscriptResult with transcript and metadata
        """
        platform = self._detect_platform(url)
        logger.info(f"Detected platform: {platform.value}")
        
        if progress_callback:
            progress_callback("Detecting platform", 0.05)
        
        # TIER 1: Try platform transcripts (if enabled and not forced to audio)
        if self.prefer_transcripts and not force_audio:
            if progress_callback:
                progress_callback("Checking for existing transcript", 0.1)
            
            result = self._try_platform_transcript(url, platform)
            if result.success:
                logger.info(f"✅ Got transcript from {result.source.value}")
                if progress_callback:
                    progress_callback("Transcript fetched", 1.0)
                return result
            
            logger.info(f"No platform transcript available: {result.error}")
        
        # TIER 2: Audio fallback
        logger.info("Falling back to audio download + transcription...")
        return self._audio_fallback(url, platform, progress_callback)
    
    def _detect_platform(self, url: str) -> Platform:
        """Detect platform from URL"""
        parsed = urlparse(url.lower())
        domain = parsed.netloc
        
        platform_patterns = {
            Platform.YOUTUBE: ['youtube.com', 'youtu.be', 'youtube-nocookie.com'],
            Platform.SPOTIFY: ['spotify.com', 'open.spotify.com'],
            Platform.APPLE_PODCASTS: ['podcasts.apple.com', 'itunes.apple.com'],
            Platform.SOUNDCLOUD: ['soundcloud.com'],
            Platform.VIMEO: ['vimeo.com'],
            Platform.TWITTER: ['twitter.com', 'x.com'],
            Platform.TIKTOK: ['tiktok.com'],
            Platform.TWITCH: ['twitch.tv'],
            Platform.FACEBOOK: ['facebook.com', 'fb.com', 'fb.watch'],
        }
        
        for platform, patterns in platform_patterns.items():
            if any(p in domain for p in patterns):
                return platform
        
        return Platform.GENERIC
    
    # =========================================================================
    # TIER 1: Platform-Specific Transcript APIs
    # =========================================================================
    
    def _try_platform_transcript(self, url: str, platform: Platform) -> TranscriptResult:
        """Try to get transcript from platform-specific API"""
        
        if platform == Platform.YOUTUBE:
            return self._fetch_youtube_transcript(url)
        
        elif platform == Platform.SPOTIFY:
            return self._fetch_spotify_transcript(url)
        
        # Add more platforms as APIs become available
        
        return TranscriptResult(
            success=False,
            platform=platform.value,
            url=url,
            error=f"No transcript API available for {platform.value}"
        )
    
    def _fetch_youtube_transcript(self, url: str) -> TranscriptResult:
        """Fetch YouTube captions"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return TranscriptResult(
                success=False,
                error="youtube-transcript-api not installed"
            )
        
        video_id = self._extract_youtube_id(url)
        if not video_id:
            return TranscriptResult(
                success=False,
                url=url,
                error="Could not extract YouTube video ID"
            )
        
        try:
            # v1.x uses instance-based API: YouTubeTranscriptApi().list(video_id)
            # v0.x used class methods: YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.list(video_id)
            except TypeError:
                # Fallback for older versions (<0.6)
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Find best transcript
            transcript = None
            is_auto = False

            # Prefer manual captions
            try:
                transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
                is_auto = False
            except:
                pass

            if not transcript:
                try:
                    transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                    is_auto = True
                except:
                    pass

            if not transcript:
                # Try any available
                for t in transcript_list:
                    transcript = t
                    is_auto = t.is_generated
                    break

            if not transcript:
                return TranscriptResult(
                    success=False,
                    platform="youtube",
                    url=url,
                    error="No transcript available"
                )

            # Fetch segments
            # v1.x returns FetchedTranscript (iterable of FetchedTranscriptSnippet objects)
            # v0.x returned a list of dicts
            raw_segments = transcript.fetch()

            segments = []
            text_parts = []
            for seg in raw_segments:
                if hasattr(seg, 'text'):
                    # v1.x: FetchedTranscriptSnippet with .text, .start, .duration
                    text, start, duration = seg.text, seg.start, seg.duration
                else:
                    # v0.x: dict with 'text', 'start', 'duration' keys
                    text, start, duration = seg['text'], seg['start'], seg['duration']
                segments.append(TranscriptSegment(text=text, start=start, end=start + duration))
                text_parts.append(text)

            full_text = ' '.join(text_parts)
            full_text = self._clean_text(full_text)
            
            duration = segments[-1].end if segments else 0
            
            return TranscriptResult(
                success=True,
                source=TranscriptSource.YOUTUBE_CAPTIONS,
                text=full_text,
                segments=segments,
                duration_seconds=duration,
                language=transcript.language_code,
                is_auto_generated=is_auto,
                platform="youtube",
                url=url
            )
        
        except Exception as e:
            return TranscriptResult(
                success=False,
                platform="youtube",
                url=url,
                error=str(e)
            )
    
    def _fetch_spotify_transcript(self, url: str) -> TranscriptResult:
        """Fetch Spotify transcript (placeholder for future implementation)"""
        # Spotify has been adding transcripts but API access is limited
        return TranscriptResult(
            success=False,
            platform="spotify",
            url=url,
            error="Spotify transcript API not yet available"
        )
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Try query parameter
        parsed = urlparse(url)
        if 'youtube.com' in parsed.netloc:
            query = parse_qs(parsed.query)
            if 'v' in query:
                return query['v'][0]
        
        return None
    
    # =========================================================================
    # TIER 2: Audio Download + Whisper Transcription
    # =========================================================================
    
    def _audio_fallback(
        self,
        url: str,
        platform: Platform,
        progress_callback: Optional[Callable]
    ) -> TranscriptResult:
        """Download audio and transcribe with Whisper"""
        
        audio_path = None
        
        try:
            # Step 1: Download audio
            if progress_callback:
                progress_callback("Downloading audio", 0.2)
            
            audio_path, metadata = self._download_audio(url)
            
            if not audio_path:
                return TranscriptResult(
                    success=False,
                    platform=platform.value,
                    url=url,
                    error="Failed to download audio"
                )
            
            logger.info(f"Downloaded: {metadata.get('title', 'Unknown')}")
            
            # Step 2: Transcribe with Whisper
            if progress_callback:
                progress_callback("Transcribing audio", 0.5)
            
            result = self._transcribe_audio(audio_path, progress_callback)
            
            # Add metadata
            result.title = metadata.get('title', '')
            result.platform = platform.value
            result.url = url
            result.duration_seconds = metadata.get('duration', result.duration_seconds)
            
            return result
        
        finally:
            # Cleanup temp audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
    
    def _get_ffmpeg_location(self) -> Optional[str]:
        """Try to find ffmpeg binary from imageio-ffmpeg (pip) or system PATH."""
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            if path and os.path.exists(path):
                return os.path.dirname(path)
        except (ImportError, Exception):
            pass

        import shutil
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return os.path.dirname(ffmpeg_path)

        return None

    def _download_audio(self, url: str) -> tuple:
        """Download audio using yt-dlp.

        Downloads in the best native audio format (m4a/webm/opus) without
        requiring an FFmpeg postprocessor. faster-whisper decodes these formats
        via its bundled PyAV/libav, so no system ffmpeg is needed.
        """
        try:
            import yt_dlp
        except ImportError:
            raise ImportError("yt-dlp not installed. Run: pip install yt-dlp")

        import glob as glob_module

        # Use %(ext)s so yt-dlp writes the correct extension automatically
        output_template = os.path.join(
            self.temp_dir,
            f"podcast_audio_{os.getpid()}.%(ext)s"
        )

        ydl_opts = {
            # Prefer m4a, then webm, then any audio — no ffmpeg conversion needed
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }

        # If ffmpeg happens to be available, let yt-dlp use it for better format selection
        ffmpeg_dir = self._get_ffmpeg_location()
        if ffmpeg_dir:
            ydl_opts['ffmpeg_location'] = ffmpeg_dir

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                # Handle playlists
                if 'entries' in info:
                    info = info['entries'][0]

                # Find the downloaded file (extension is determined by yt-dlp)
                pattern = os.path.join(self.temp_dir, f"podcast_audio_{os.getpid()}.*")
                files = glob_module.glob(pattern)
                actual_path = files[0] if files else None

                if not actual_path or not os.path.exists(actual_path):
                    return None, {}

                metadata = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'description': info.get('description', '')[:500],
                }

                return actual_path, metadata

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None, {}
    
    def _transcribe_audio(
        self,
        audio_path: str,
        progress_callback: Optional[Callable]
    ) -> TranscriptResult:
        """Transcribe audio with faster-whisper"""
        
        if self._whisper is None:
            if progress_callback:
                progress_callback("Loading Whisper model", 0.4)
            self._load_whisper()
        
        try:
            segments_raw, info = self._whisper.transcribe(
                audio_path,
                beam_size=5,
                language=None,  # Auto-detect
                word_timestamps=True,
                vad_filter=True,
            )
            
            # Convert generator to list and build segments
            segments = []
            text_parts = []
            
            for seg in segments_raw:
                segments.append(TranscriptSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end
                ))
                text_parts.append(seg.text.strip())
                
                if progress_callback:
                    # Estimate progress based on time
                    progress = min(0.5 + (seg.end / info.duration) * 0.45, 0.95)
                    progress_callback("Transcribing", progress)
            
            full_text = ' '.join(text_parts)
            full_text = self._clean_text(full_text)
            
            if progress_callback:
                progress_callback("Complete", 1.0)
            
            return TranscriptResult(
                success=True,
                source=TranscriptSource.WHISPER,
                text=full_text,
                segments=segments,
                duration_seconds=info.duration,
                language=info.language,
                is_auto_generated=True  # Whisper is auto-generated
            )
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptResult(
                success=False,
                error=f"Transcription failed: {e}"
            )
    
    def _load_whisper(self):
        """Load faster-whisper model"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
        
        device = self.whisper_device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                device = "cpu"
        
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Loading Whisper model: {self.whisper_model} on {device}")
        
        self._whisper = WhisperModel(
            self.whisper_model,
            device=device,
            compute_type=compute_type
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean transcript text"""
        # Remove [Music], [Applause], etc.
        text = re.sub(r'\[.*?\]', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # =========================================================================
    # File Upload Support
    # =========================================================================
    
    def transcribe_file(
        self,
        file_path: str,
        progress_callback: Optional[Callable] = None
    ) -> TranscriptResult:
        """
        Transcribe a local audio/video file.
        
        Args:
            file_path: Path to audio/video file
            progress_callback: Optional progress callback
        
        Returns:
            TranscriptResult
        """
        if not os.path.exists(file_path):
            return TranscriptResult(
                success=False,
                error=f"File not found: {file_path}"
            )
        
        if progress_callback:
            progress_callback("Loading file", 0.1)
        
        result = self._transcribe_audio(file_path, progress_callback)
        result.title = Path(file_path).stem
        result.platform = "local_file"
        result.url = file_path
        
        return result


# Convenience functions
def fetch_transcript(url: str, **kwargs) -> TranscriptResult:
    """
    Fetch transcript from any supported URL.
    
    Args:
        url: URL to video/podcast
        **kwargs: Arguments passed to MultiPlatformFetcher
    
    Returns:
        TranscriptResult
    """
    fetcher = MultiPlatformFetcher(**kwargs)
    return fetcher.fetch(url)


def transcribe_file(file_path: str, **kwargs) -> TranscriptResult:
    """
    Transcribe a local audio/video file.
    
    Args:
        file_path: Path to file
        **kwargs: Arguments passed to MultiPlatformFetcher
    
    Returns:
        TranscriptResult
    """
    fetcher = MultiPlatformFetcher(**kwargs)
    return fetcher.transcribe_file(file_path)


# CLI
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Multi-Platform Transcript Fetcher")
        print("=" * 40)
        print("\nUsage:")
        print("  python multi_platform.py <URL>")
        print("  python multi_platform.py <URL> --force-audio")
        print("  python multi_platform.py --file <path>")
        print("\nSupported platforms:")
        print("  YouTube, Spotify, Vimeo, Twitter/X, TikTok,")
        print("  SoundCloud, Twitch, Facebook, and 1000+ more!")
        sys.exit(1)
    
    force_audio = "--force-audio" in sys.argv
    is_file = "--file" in sys.argv
    
    # Get URL or file path
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Error: No URL or file provided")
        sys.exit(1)
    
    target = args[0]
    
    def progress(step, pct):
        print(f"  [{pct*100:5.1f}%] {step}")
    
    fetcher = MultiPlatformFetcher(whisper_model="large-v3")
    
    if is_file:
        print(f"\nTranscribing file: {target}")
        result = fetcher.transcribe_file(target, progress_callback=progress)
    else:
        print(f"\nFetching transcript for: {target}")
        result = fetcher.fetch(target, force_audio=force_audio, progress_callback=progress)
    
    print()
    if result.success:
        print("✅ SUCCESS!")
        print(f"   Source: {result.source.value}")
        print(f"   Platform: {result.platform}")
        print(f"   Duration: {result.duration_minutes:.1f} min")
        print(f"   Words: {result.word_count}")
        print(f"   Language: {result.language}")
        print(f"\n--- First 500 characters ---\n{result.text[:500]}")
    else:
        print(f"❌ FAILED: {result.error}")
