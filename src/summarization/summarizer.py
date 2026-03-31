"""
Podcast Summarizer v2.0 - Enhanced LLM Summarizer
=================================================
Summarizes transcripts with intelligent content-type detection.
Uses Google Gemini as the sole LLM provider (1M token context).

Adapts summary style based on content:
- Podcasts/Interviews -> Focus on speakers, key discussions
- Tutorials/Educational -> Focus on steps, concepts, takeaways
- News/Commentary -> Focus on topics, opinions, facts
- Entertainment -> Focus on highlights, moments

Version: 2.1.0
"""

import os
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Detected content type"""
    PODCAST = "podcast"
    INTERVIEW = "interview"
    TUTORIAL = "tutorial"
    LECTURE = "lecture"
    NEWS = "news"
    COMMENTARY = "commentary"
    ENTERTAINMENT = "entertainment"
    GENERAL = "general"


class SummaryFormat(Enum):
    """Output format options"""
    DETAILED = "detailed"
    QUICK = "quick"
    BULLETS = "bullets"
    CHAPTERS = "chapters"
    EXECUTIVE = "executive"

    @classmethod
    def from_string(cls, name: str) -> 'SummaryFormat':
        """Convert string name to SummaryFormat, defaulting to DETAILED."""
        _map = {
            "detailed": cls.DETAILED,
            "quick": cls.QUICK,
            "bullets": cls.BULLETS,
            "chapters": cls.CHAPTERS,
            "executive": cls.EXECUTIVE,
        }
        return _map.get(name, cls.DETAILED)


@dataclass
class Chapter:
    """A chapter/section"""
    title: str
    timestamp: str
    summary: str
    start_seconds: float = 0


@dataclass
class Quote:
    """A notable quote"""
    text: str
    speaker: str = ""
    context: str = ""
    timestamp: str = ""


@dataclass
class Summary:
    """Complete summary output"""
    success: bool

    # Core content
    title: str = ""
    executive_summary: str = ""
    key_takeaways: List[str] = field(default_factory=list)

    # Structure
    chapters: List[Chapter] = field(default_factory=list)

    # Details
    notable_quotes: List[Quote] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    # Metadata
    content_type: ContentType = ContentType.GENERAL
    word_count_original: int = 0
    word_count_summary: int = 0
    duration_minutes: float = 0

    # Error handling
    error: Optional[str] = None

    @property
    def compression_ratio(self) -> float:
        if self.word_count_summary == 0:
            return 0
        return self.word_count_original / self.word_count_summary


class EnhancedSummarizer:
    """
    Intelligent summarizer using Google Gemini (1M token context).

    Features:
    - Auto-detects content type (podcast, tutorial, news, etc.)
    - Adapts prompts based on content type
    - Handles transcripts of any length in a single API call
    - Exponential backoff retry logic (no blocking hardcoded sleeps)
    - Robust JSON parsing

    Example:
        >>> summarizer = EnhancedSummarizer()
        >>> result = summarizer.summarize(transcript, title="My Video")
        >>> print(result.executive_summary)
    """

    # Exponential backoff configuration
    _INITIAL_RETRY_DELAY = 1.0   # Start at 1 second
    _MAX_RETRY_DELAY = 30.0      # Cap at 30 seconds
    _BACKOFF_MULTIPLIER = 2.0    # Double each retry

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        **kwargs  # Accept and ignore legacy params for backwards compat
    ):
        """
        Initialize summarizer.

        Args:
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.gemini_api_key = gemini_api_key or os.environ.get('GEMINI_API_KEY')

        if not self.gemini_api_key:
            raise ValueError(
                "No Gemini API key found!\n"
                "Get a free key at: https://aistudio.google.com/apikey\n"
                "Then: export GEMINI_API_KEY='your-key'"
            )

        self.provider = "gemini"
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self._gemini_client = None

        logger.info(f"Using provider: gemini ({self.gemini_model})")

    def _get_gemini_client(self):
        """Lazy load Gemini client (singleton)."""
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)
        return self._gemini_client

    def _backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for a given attempt number."""
        delay = self._INITIAL_RETRY_DELAY * (self._BACKOFF_MULTIPLIER ** attempt)
        return min(delay, self._MAX_RETRY_DELAY)

    def summarize(
        self,
        transcript: str,
        title: str = "",
        format: SummaryFormat = SummaryFormat.DETAILED,
        content_type: Optional[ContentType] = None,
        duration_minutes: float = 0,
        max_retries: int = 3
    ) -> Summary:
        """
        Generate intelligent summary using Gemini.

        Gemini's 1M token context window handles transcripts of any length
        in a single API call — no chunking needed.

        Args:
            transcript: Full transcript text
            title: Title of the content
            format: Output format
            content_type: Override auto-detected content type
            duration_minutes: Duration for context
            max_retries: Retry attempts for API errors

        Returns:
            Summary object with all components
        """
        if not transcript or not transcript.strip():
            return Summary(success=False, error="Empty transcript")

        word_count = len(transcript.split())

        # Auto-detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(transcript, title)
            logger.info(f"Detected content type: {content_type.value}")

        # Build prompt — Gemini handles up to 1M tokens, no size check needed
        prompt = self._build_prompt(
            transcript, title, format, content_type, duration_minutes
        )

        logger.info(f"Using Gemini ({len(prompt):,} chars prompt)")

        from google.genai import types

        for attempt in range(max_retries):
            try:
                client = self._get_gemini_client()
                response = client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=8192,
                    ),
                )

                raw = response.text
                result = self._parse_response(raw, content_type)

                if result.success:
                    result.word_count_original = word_count
                    # Count ALL summary output words for accurate compression ratio
                    summary_words = len(result.executive_summary.split())
                    for t in result.key_takeaways:
                        summary_words += len(str(t).split())
                    for ch in result.chapters:
                        summary_words += len(ch.summary.split())
                        summary_words += len(ch.title.split())
                    for q in result.notable_quotes:
                        summary_words += len(q.text.split())
                    result.word_count_summary = summary_words
                    result.duration_minutes = duration_minutes
                    result.title = title
                    return result

                logger.warning(f"Gemini attempt {attempt + 1} parse failed, retrying...")
                prompt = self._add_format_hint(prompt)

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if "limit: 0" in error_str or "quota exceeded" in error_str.lower():
                        logger.error("Gemini quota fully exhausted")
                        return Summary(
                            success=False,
                            error="Gemini quota exhausted. Wait for quota reset or upgrade your plan."
                        )
                    # Temporary rate limit — use exponential backoff
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        f"Gemini rate limited, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini API error: {e}")
                    if attempt == max_retries - 1:
                        return Summary(success=False, error=str(e))

        return Summary(success=False, error="Gemini failed after retries")

    def _detect_content_type(self, transcript: str, title: str) -> ContentType:
        """Auto-detect content type from transcript and title"""
        text = (title + " " + transcript[:3000]).lower()

        # Check for interview patterns
        interview_signals = [
            "interview", "guest", "joining us", "welcome to the show",
            "tell us about", "what do you think", "in your experience"
        ]
        if sum(1 for s in interview_signals if s in text) >= 2:
            return ContentType.INTERVIEW

        # Check for podcast patterns
        podcast_signals = [
            "podcast", "episode", "today we're", "thanks for listening",
            "subscribe", "welcome back"
        ]
        if sum(1 for s in podcast_signals if s in text) >= 2:
            return ContentType.PODCAST

        # Check for tutorial patterns
        tutorial_signals = [
            "tutorial", "how to", "step by step", "let me show you",
            "first we'll", "next we", "click on", "install"
        ]
        if sum(1 for s in tutorial_signals if s in text) >= 2:
            return ContentType.TUTORIAL

        # Check for lecture patterns
        lecture_signals = [
            "lecture", "class", "students", "professor", "university",
            "let's examine", "as we discussed", "the theory"
        ]
        if sum(1 for s in lecture_signals if s in text) >= 2:
            return ContentType.LECTURE

        # Check for news patterns
        news_signals = [
            "breaking", "reported", "according to", "officials say",
            "news", "today's", "update"
        ]
        if sum(1 for s in news_signals if s in text) >= 2:
            return ContentType.NEWS

        return ContentType.GENERAL

    @staticmethod
    def get_format_schema(format: 'SummaryFormat', content_type: 'ContentType' = None) -> str:
        """Return the format-specific JSON schema for summary output.

        Extracted as a public method so it can be reused by the combined
        Gemini transcribe+summarize path in main.py.
        """
        # Content-type specific focus (appended to any format)
        ct_focus = ""
        if content_type:
            ct_map = {
                ContentType.PODCAST: (
                    "This is a PODCAST. Focus on: main discussion topics between "
                    "hosts/guests, key opinions and perspectives, interesting anecdotes, "
                    "and any disagreements or debates."
                ),
                ContentType.INTERVIEW: (
                    "This is an INTERVIEW. Focus on: the interviewee's background and "
                    "expertise, key insights and experiences shared, memorable quotes, "
                    "and advice or recommendations given."
                ),
                ContentType.TUTORIAL: (
                    "This is a TUTORIAL. Focus on: step-by-step instructions, key "
                    "concepts explained, prerequisites, common mistakes to avoid, "
                    "and tips/best practices."
                ),
                ContentType.LECTURE: (
                    "This is a LECTURE. Focus on: main concepts and theories, key "
                    "definitions and frameworks, examples and case studies, and "
                    "connections between ideas."
                ),
                ContentType.NEWS: (
                    "This is NEWS/COMMENTARY. Focus on: key facts and events, "
                    "who/what/when/where/why, different perspectives, implications, "
                    "and what remains unknown."
                ),
            }
            ct_focus = ct_map.get(content_type, "")

        if format == SummaryFormat.QUICK:
            schema = '''JSON schema:
{
    "executive_summary": "2-3 paragraphs: (1) what & why it matters, (2) main ideas with specifics, (3) conclusions",
    "key_takeaways": ["5-7 concrete insights with names/numbers/examples — no vague generalities"],
    "topics": ["specific topics"]
}'''
        elif format == SummaryFormat.BULLETS:
            schema = '''JSON schema:
{
    "key_takeaways": ["7-10 self-contained insights. Each: one distinct point + concrete detail (who/what/number)."]
}'''
        elif format == SummaryFormat.CHAPTERS:
            schema = '''JSON schema (5-10 chapters following natural flow):
{
    "chapters": [{"title": "Name", "timestamp": "MM:SS", "summary": "2-3 sentences with specific points"}],
    "executive_summary": "1-2 paragraph overview"
}'''
        else:  # DETAILED
            schema = '''JSON schema (5-10 chapters, 3-5 quotes, 5-8 topics):
{
    "executive_summary": "3 paragraphs: (1) context — who/what/why it matters, (2) main arguments/ideas with specific details, (3) conclusions/implications",
    "key_takeaways": ["5-7 insights — each: '[Speaker/Source] argues/shows/recommends [specific claim] because [reason]'"],
    "chapters": [{"title": "Section Name", "timestamp": "MM:SS", "summary": "2-3 sentences with specific points and examples"}],
    "notable_quotes": [{"text": "exact words spoken", "speaker": "name", "context": "significance"}],
    "topics": ["specific topic names"],
    "action_items": ["specific recommendations or next steps mentioned"]
}'''

        if ct_focus:
            schema += f"\n{ct_focus}"

        return schema

    @staticmethod
    def detect_content_type_from_title(title: str) -> 'ContentType':
        """Quick content-type detection from title alone (for fast path)."""
        t = title.lower()
        if any(w in t for w in ['interview', 'guest', 'conversation with', 'talks to']):
            return ContentType.INTERVIEW
        if any(w in t for w in ['podcast', 'episode', 'ep.', 'ep ']):
            return ContentType.PODCAST
        if any(w in t for w in ['tutorial', 'how to', 'guide', 'learn', 'course']):
            return ContentType.TUTORIAL
        if any(w in t for w in ['lecture', 'class', 'lesson']):
            return ContentType.LECTURE
        if any(w in t for w in ['news', 'breaking', 'update', 'report']):
            return ContentType.NEWS
        return ContentType.GENERAL

    def _build_prompt(
        self,
        transcript: str,
        title: str,
        format: SummaryFormat,
        content_type: ContentType,
        duration: float
    ) -> str:
        """Build prompt adapted to content type"""

        # Content-specific instructions
        content_instructions = {
            ContentType.PODCAST: """
Focus on:
- Main topics discussed between hosts/guests
- Key opinions and perspectives shared
- Interesting anecdotes or stories
- Disagreements or debates""",

            ContentType.INTERVIEW: """
Focus on:
- The interviewee's background and expertise
- Key insights and experiences shared
- Memorable quotes and stories
- Advice or recommendations given""",

            ContentType.TUTORIAL: """
Focus on:
- Step-by-step instructions
- Key concepts explained
- Prerequisites and requirements
- Common mistakes to avoid
- Tips and best practices""",

            ContentType.LECTURE: """
Focus on:
- Main concepts and theories presented
- Key definitions and frameworks
- Examples and case studies
- Connections between ideas
- Questions raised or addressed""",

            ContentType.NEWS: """
Focus on:
- Key facts and events reported
- Who, what, when, where, why
- Different perspectives presented
- Implications and analysis
- What remains unknown""",

            ContentType.GENERAL: """
Focus on:
- Main topics covered
- Key points and arguments
- Notable moments or statements
- Conclusions or outcomes"""
        }

        type_instruction = content_instructions.get(
            content_type, content_instructions[ContentType.GENERAL]
        )

        structure = self.get_format_schema(format, content_type)


        return f"""You are an expert content analyst. Produce a high-quality, insightful summary of this {content_type.value} transcript.

{"TITLE: " + title if title else ""}
{"DURATION: " + f"{duration:.0f} minutes" if duration else ""}
CONTENT TYPE: {content_type.value}

{type_instruction}

TRANSCRIPT:
{transcript}

{structure}

QUALITY RULES:
1. Return ONLY valid JSON - no markdown, no extra text
2. Start with {{ and end with }}
3. Be SPECIFIC: reference actual names, numbers, claims, and examples from the content
4. NEVER use generic filler like "interesting discussion" or "various topics" — always state WHAT was discussed
5. For executive_summary: explain what this is about, the key arguments/ideas, and why it matters
6. For key_takeaways: each must be a standalone insight with concrete details
7. For quotes: use EXACT words from the transcript, not paraphrases"""

    def _add_format_hint(self, prompt: str) -> str:
        """Add formatting hint after failed parse"""
        return prompt + """

IMPORTANT: Previous response had JSON errors.
- Start immediately with {
- End with }
- No markdown code blocks
- Escape quotes in strings with \\"""

    def _parse_response(self, raw: str, content_type: ContentType) -> Summary:
        """Parse LLM response into Summary"""

        data = self._extract_json(raw)
        if data is None:
            return Summary(success=False, error="JSON parse failed")

        try:
            summary = Summary(
                success=True,
                content_type=content_type,
                executive_summary=data.get('executive_summary', ''),
                key_takeaways=data.get('key_takeaways', []),
                topics=data.get('topics', []),
                action_items=data.get('action_items', [])
            )

            # Parse chapters
            for ch in data.get('chapters', []):
                summary.chapters.append(Chapter(
                    title=ch.get('title', ''),
                    timestamp=ch.get('timestamp', '00:00'),
                    summary=ch.get('summary', '')
                ))

            # Parse quotes
            for q in data.get('notable_quotes', []):
                if isinstance(q, dict):
                    summary.notable_quotes.append(Quote(
                        text=q.get('text', ''),
                        speaker=q.get('speaker', ''),
                        context=q.get('context', '')
                    ))
                elif isinstance(q, str):
                    summary.notable_quotes.append(Quote(text=q))

            return summary

        except Exception as e:
            return Summary(success=False, error=f"Parse error: {e}")

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON with multiple strategies"""
        text = text.strip()

        # Fast path: direct parse (most common for well-formed responses)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown code blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        for p in patterns:
            match = re.search(p, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except (json.JSONDecodeError, ValueError):
                    continue

        # Last resort: balanced brace matching with trailing comma fix
        start = text.find('{')
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
                        return json.loads(fixed)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

        return None


# Convenience function
def summarize(
    transcript: str,
    title: str = "",
    format: str = "detailed",
    **kwargs
) -> Summary:
    """
    Summarize a transcript.

    Args:
        transcript: Transcript text
        title: Content title
        format: "detailed", "quick", "bullets", "chapters"
        **kwargs: Additional args for EnhancedSummarizer

    Returns:
        Summary object
    """
    summarizer = EnhancedSummarizer(**kwargs)
    return summarizer.summarize(
        transcript,
        title=title,
        format=SummaryFormat.from_string(format)
    )
