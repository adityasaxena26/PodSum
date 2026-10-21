"""
Podcast Summarizer v2.0 - Multi-Platform Transcript Fetcher
============================================================
Smart transcript acquisition with tiered fallback:

TIER 1: Gemini transcription (YouTube only, works from cloud IPs)

TIER 2: Platform transcripts (FREE, instant)
    - YouTube captions
    - (Future: Spotify, Apple Podcasts, RSS feeds)

TIER 3: Audio download + Whisper transcription (fallback)
    - Supports 1000+ sites via yt-dlp
    - Uses faster-whisper for efficient transcription
Version: 2.2.0
"""

import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Callable
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
        # Cache the count — .split() on 50K+ char strings is not free
        if not hasattr(self, '_word_count_cache') or self._word_count_cache is None:
            self._word_count_cache = len(self.text.split())
        return self._word_count_cache

    def __setattr__(self, name, value):
        # Invalidate word count cache when text changes
        if name == 'text':
            object.__setattr__(self, '_word_count_cache', None)
        object.__setattr__(self, name, value)
    
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
        self.cookies_path = None  # Optional path to cookies.txt
        os.makedirs(self.temp_dir, exist_ok=True)

        # Lazy-loaded components (singletons)
        self._whisper = None
        self._yt_dlp = None
        self._gemini_client = None
        self._yt_transcript_api = None
        self._http_client = None  # Reusable httpx client with connection pooling
    
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
        # Validate URL: reject non-HTTP schemes (file://, ftp://, etc.) to prevent SSRF
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return TranscriptResult(
                success=False,
                url=url,
                error=f"Unsupported URL scheme: {parsed.scheme}. Only http:// and https:// are allowed."
            )

        platform = self._detect_platform(url)
        logger.info(f"Detected platform: {platform.value}")

        if progress_callback:
            progress_callback("Detecting platform", 0.05)
        
        # Track tier errors so the final error message shows what was tried
        tier_errors = []

        # ── Smart tier ordering ──────────────────────────────────
        # On cloud servers (GCP, AWS, etc.), YouTube blocks captions API
        # and yt-dlp audio download. Gemini transcription (googleapis.com)
        # is the ONLY reliable method. So we try it FIRST to avoid
        # wasting 30-60s on doomed requests.
        #
        # Order:
        #   1. Gemini transcription (always works on cloud — fastest path)
        #   2. YouTube captions API (works locally, blocked on cloud)
        #   3. Audio download + Whisper (works locally, blocked on cloud)

        if self.prefer_transcripts and not force_audio:
            # TIER 1: Gemini transcription (googleapis.com — never blocked)
            # Try this FIRST because it's the only reliable method on cloud.
            if platform == Platform.YOUTUBE and os.environ.get('GEMINI_API_KEY'):
                if progress_callback:
                    progress_callback("Transcribing via Gemini...", 0.1)
                logger.info("Tier 1: Gemini YouTube transcription (googleapis.com)")
                result = self._fetch_transcript_via_gemini(url)
                if result.success:
                    logger.info(f"✅ Got transcript via Gemini: {result.word_count:,} words")
                    if progress_callback:
                        progress_callback("Transcript fetched", 1.0)
                    return result
                tier_errors.append(f"Gemini transcription: {result.error}")
                logger.warning(f"Tier 1 (Gemini) failed: {result.error}")

            # TIER 2: YouTube captions API (free, instant — but blocked on cloud IPs)
            if progress_callback:
                progress_callback("Trying captions API...", 0.2)
            logger.info("Tier 2: YouTube captions API")

            result = self._try_platform_transcript(url, platform)
            if result.success:
                logger.info(f"✅ Got transcript from {result.source.value}")
                if progress_callback:
                    progress_callback("Transcript fetched", 1.0)
                return result

            tier_errors.append(f"Captions API: {result.error}")
            logger.info(f"Tier 2 (captions) failed: {result.error}")

        # TIER 3: Audio fallback (yt-dlp + Whisper — blocked on cloud)
        logger.info("Tier 3: Audio download + Whisper transcription")
        result = self._audio_fallback(url, platform, progress_callback)
        if result.success:
            return result

        tier_errors.append(f"Audio download: {result.error}")

        # All tiers failed — return a combined error so the user knows what was tried
        combined = " | ".join(tier_errors)
        logger.error(f"All transcript tiers failed: {combined}")
        return TranscriptResult(
            success=False,
            platform=platform.value,
            url=url,
            error=f"All methods failed: {combined}"
        )
    
    def _get_gemini_client(self):
        """Lazy-load and reuse a single Gemini client (singleton per fetcher)."""
        if self._gemini_client is None:
            from google import genai
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def _get_http_client(self):
        """Reusable httpx client with connection pooling.

        A single client reuses TCP connections across requests,
        eliminating per-request connection setup overhead (~50-100ms each).
        This matters because the fallback tiers can make 5-8 HTTP requests.
        """
        if self._http_client is None:
            import httpx
            self._http_client = httpx.Client(
                follow_redirects=True,
                timeout=30,
                # Connection pooling: keep connections alive for reuse
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30,
                ),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
        return self._http_client

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
            # youtube-transcript-api v1.2.x: reuse singleton
            if self._yt_transcript_api is None:
                self._yt_transcript_api = YouTubeTranscriptApi()
            ytt_api = self._yt_transcript_api
            transcript_list = ytt_api.list(video_id)

            # Find best transcript
            transcript = None
            is_auto = False

            # Prefer manual captions
            try:
                transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
                is_auto = False
            except Exception:
                pass

            if not transcript:
                try:
                    transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                    is_auto = True
                except Exception:
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
            for seg in raw_segments:
                if hasattr(seg, 'text'):
                    # v1.x: FetchedTranscriptSnippet with .text, .start, .duration
                    text, start, duration = seg.text, seg.start, seg.duration
                else:
                    # v0.x: dict with 'text', 'start', 'duration' keys
                    text, start, duration = seg['text'], seg['start'], seg['duration']
                segments.append(TranscriptSegment(text=text, start=start, end=start + duration))

            # Deduplicate overlapping segments before joining.
            # YouTube auto-captions use rolling text that can inflate
            # word count by 10-15x if naively concatenated.
            full_text = self._deduplicate_segments(segments)
            full_text = self._clean_text(full_text)

            naive_count = sum(len(s.text.split()) for s in segments)
            deduped_count = len(full_text.split())
            if naive_count > 0 and deduped_count < naive_count:
                logger.info(
                    f"Transcript deduplication: {naive_count:,} → {deduped_count:,} words "
                    f"({(1 - deduped_count/naive_count)*100:.0f}% reduction)"
                )
            
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
    
    # =========================================================================
    # TIER 1.5: Gemini YouTube Transcription (googleapis.com — never blocked)
    # =========================================================================

    def _fetch_transcript_via_gemini(self, url: str) -> TranscriptResult:
        """
        Use Gemini to transcribe a YouTube video directly.

        Gemini natively understands YouTube URLs via Google's internal
        infrastructure. All requests go through googleapis.com, which is
        NEVER blocked on GCP or any cloud provider IP.

        This is the most reliable method for cloud-deployed applications.
        """
        try:
            from google import genai
        except ImportError:
            return TranscriptResult(
                success=False, platform="youtube", url=url,
                error="google-genai not installed"
            )

        if not os.environ.get('GEMINI_API_KEY'):
            return TranscriptResult(
                success=False, platform="youtube", url=url,
                error="GEMINI_API_KEY not set"
            )

        video_id = self._extract_youtube_id(url)
        if not video_id:
            return TranscriptResult(
                success=False, platform="youtube", url=url,
                error="Could not extract YouTube video ID"
            )

        canonical_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            client = self._get_gemini_client()

            # Step 1: Get metadata via YouTube Data API v3 if available
            metadata = {}
            yt_api_key = os.environ.get('YOUTUBE_API_KEY')
            if yt_api_key:
                try:
                    metadata = self._yt_data_api_metadata(video_id, yt_api_key)
                except Exception:
                    pass

            # Step 2: Ask Gemini to transcribe the video verbatim
            logger.info(f"Sending YouTube URL to Gemini for transcription: {canonical_url}")

            response = client.models.generate_content(
                model=os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'),
                contents=[
                    genai.types.Part.from_uri(
                        file_uri=canonical_url,
                        mime_type='video/x-youtube',
                    ),
                    (
                        "Transcribe every word spoken in this video verbatim. "
                        "Return ONLY the transcript text — no timestamps, no speaker labels, "
                        "no commentary, no markdown formatting. "
                        "Include every sentence from start to finish. "
                        "Do not summarize or paraphrase."
                    ),
                ],
            )

            transcript_text = response.text.strip()

            if not transcript_text or len(transcript_text) < 50:
                return TranscriptResult(
                    success=False, platform="youtube", url=canonical_url,
                    error="Gemini returned empty or very short transcript"
                )

            # Clean up any accidental markdown that Gemini might add
            transcript_text = self._clean_text(transcript_text)

            # Build a single segment (Gemini doesn't provide timestamps)
            duration = metadata.get('duration', 0)
            segments = [TranscriptSegment(
                text=transcript_text,
                start=0.0,
                end=float(duration) if duration else 0.0,
            )]

            logger.info(
                f"Gemini transcription complete: {len(transcript_text.split())} words"
            )

            return TranscriptResult(
                success=True,
                source=TranscriptSource.YOUTUBE_CAPTIONS,
                text=transcript_text,
                segments=segments,
                title=metadata.get('title', ''),
                duration_seconds=duration,
                language="en",
                is_auto_generated=False,
                platform="youtube",
                url=canonical_url,
            )

        except Exception as e:
            logger.error(f"Gemini transcription failed: {type(e).__name__}: {e}")
            return TranscriptResult(
                success=False, platform="youtube", url=canonical_url,
                error=f"Gemini transcription: {type(e).__name__}: {e}"
            )

    def _yt_data_api_metadata(self, video_id: str, api_key: str) -> dict:
        """Fetch video metadata via YouTube Data API v3 (works with API key)."""
        url = (
            "https://www.googleapis.com/youtube/v3/videos"
            f"?part=snippet,contentDetails&id={video_id}&key={api_key}"
        )
        http = self._get_http_client()
        resp = http.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get('items'):
            return {}

        item = data['items'][0]
        snippet = item.get('snippet', {})
        content_details = item.get('contentDetails', {})

        return {
            'title': snippet.get('title', ''),
            'duration': self._parse_iso8601_duration(content_details.get('duration', '')),
            'channel': snippet.get('channelTitle', ''),
            'description': snippet.get('description', '')[:500],
        }

    @staticmethod
    def _parse_iso8601_duration(duration: str) -> int:
        """Parse ISO 8601 duration like PT1H2M3S to total seconds."""
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration or '')
        if not match:
            return 0
        h, m, s = (int(g) if g else 0 for g in match.groups())
        return h * 3600 + m * 60 + s

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
                dl_error = metadata.get("error", "unknown error")
                return TranscriptResult(
                    success=False,
                    platform=platform.value,
                    url=url,
                    error=f"Failed to download audio: {dl_error}"
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
                except OSError:
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
            f"podcast_audio_{os.getpid()}_{threading.get_ident()}.%(ext)s"
        )

        ydl_opts = {
            # Prefer m4a, then webm, then any audio — no ffmpeg conversion needed
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,
        }

        if self.cookies_path:
            ydl_opts['cookiefile'] = self.cookies_path

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
                pattern = os.path.join(self.temp_dir, f"podcast_audio_{os.getpid()}_{threading.get_ident()}.*")
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
            return None, {"error": str(e)}
    
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
            except (ImportError, Exception):
                device = "cpu"
        
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Loading Whisper model: {self.whisper_model} on {device}")
        
        self._whisper = WhisperModel(
            self.whisper_model,
            device=device,
            compute_type=compute_type
        )
    
    def _deduplicate_segments(self, segments: list) -> str:
        """
        Deduplicate overlapping YouTube caption segments.

        YouTube auto-generated captions use rolling/overlapping segments
        where each new segment repeats words from the previous one.
        A 30-min video can inflate from ~4,500 words to 50,000+ if
        segments are naively concatenated.

        Algorithm: word-level suffix/prefix matching. For each new segment,
        find the longest overlap between the end of the accumulated text
        and the start of the new segment, then append only the new portion.
        """
        if not segments:
            return ""

        result_words = segments[0].text.split()

        for i in range(1, len(segments)):
            new_words = segments[i].text.split()
            if not new_words:
                continue

            # Find longest overlap: check if end of result matches start of new
            best_overlap = 0
            max_check = min(len(result_words), len(new_words), 40)
            for overlap in range(1, max_check + 1):
                if result_words[-overlap:] == new_words[:overlap]:
                    best_overlap = overlap

            # Only add the non-overlapping portion
            if best_overlap > 0:
                result_words.extend(new_words[best_overlap:])
            else:
                # No word-level overlap detected — check for time-based overlap
                # If segments overlap in time by >50%, likely duplicate; skip
                if i > 0 and hasattr(segments[i], 'start') and hasattr(segments[i-1], 'end'):
                    overlap_time = segments[i-1].end - segments[i].start
                    seg_duration = segments[i].end - segments[i].start
                    if seg_duration > 0 and overlap_time / seg_duration > 0.5:
                        continue  # Skip heavily overlapping segment
                result_words.extend(new_words)

        return ' '.join(result_words)

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
