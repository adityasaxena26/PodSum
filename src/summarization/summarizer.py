"""
Podcast Summarizer v2.0 - Enhanced LLM Summarizer
=================================================
Summarizes transcripts with intelligent content-type detection.

Adapts summary style based on content:
- Podcasts/Interviews → Focus on speakers, key discussions
- Tutorials/Educational → Focus on steps, concepts, takeaways
- News/Commentary → Focus on topics, opinions, facts
- Entertainment → Focus on highlights, moments

Version: 2.0.0
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
    Intelligent summarizer that adapts to content type.
    
    Features:
    - Auto-detects content type (podcast, tutorial, news, etc.)
    - Adapts prompts based on content type
    - Handles long transcripts with map-reduce
    - Robust JSON parsing
    
    Example:
        >>> summarizer = EnhancedSummarizer()
        >>> result = summarizer.summarize(transcript, title="My Video")
        >>> print(result.executive_summary)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        provider: str = "auto",
        model: str = "llama-3.3-70b-versatile",
        fast_model: str = "llama-3.1-8b-instant"
    ):
        """
        Initialize summarizer.

        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
            provider: "gemini", "groq", or "auto" (picks based on available keys)
            model: Primary Groq model for summaries
            fast_model: Faster Groq model for chunked processing
        """
        self.groq_api_key = api_key or os.environ.get('GROQ_API_KEY')
        self.gemini_api_key = gemini_api_key or os.environ.get('GEMINI_API_KEY')

        # Auto-select provider
        if provider == "auto":
            if self.gemini_api_key:
                self.provider = "gemini"
            elif self.groq_api_key:
                self.provider = "groq"
            else:
                raise ValueError(
                    "No API key found!\n"
                    "Option 1 (recommended): export GEMINI_API_KEY='your-key'\n"
                    "  Get free key at: https://aistudio.google.com/apikey\n"
                    "Option 2: export GROQ_API_KEY='your-key'\n"
                    "  Get free key at: https://console.groq.com/keys"
                )
        else:
            self.provider = provider

        logger.info(f"Using provider: {self.provider}")

        self.model = model
        self.fast_model = fast_model
        self._groq_client = None
        self._gemini_client = None

    def _get_client(self):
        """Lazy load Groq client"""
        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=self.groq_api_key)
        return self._groq_client

    def _get_gemini_client(self):
        """Lazy load Gemini client"""
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)
        return self._gemini_client

    def _summarize_gemini(
        self,
        prompt: str,
        word_count: int,
        title: str,
        content_type: ContentType,
        duration_minutes: float,
        max_retries: int = 3
    ) -> Summary:
        """Summarize using Google Gemini (1M token context, no chunking needed)."""
        from google.genai import types

        logger.info(f"Using Gemini ({len(prompt):,} chars prompt)")

        for attempt in range(max_retries):
            try:
                client = self._get_gemini_client()
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
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
                    result.word_count_summary = len(result.executive_summary.split())
                    result.duration_minutes = duration_minutes
                    result.title = title
                    return result

                logger.warning(f"Gemini attempt {attempt + 1} parse failed, retrying...")
                prompt = self._add_format_hint(prompt)

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Check if it's a short retry delay vs quota fully exhausted
                    if "limit: 0" in error_str or "quota exceeded" in error_str.lower():
                        # Quota fully exhausted — fall back to Groq if available
                        if self.groq_api_key:
                            logger.warning("Gemini quota exhausted, falling back to Groq")
                            self.provider = "groq"
                            return None  # Signal caller to retry with Groq
                        logger.error("Gemini quota exhausted and no Groq key available")
                        return Summary(success=False, error="Gemini quota exhausted. Set GROQ_API_KEY as fallback or wait for quota reset.")
                    # Temporary rate limit — retry after delay
                    logger.warning(f"Gemini rate limited, waiting 5s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(5)
                else:
                    logger.error(f"Gemini API error: {e}")
                    if attempt == max_retries - 1:
                        return Summary(success=False, error=str(e))

        return Summary(success=False, error="Gemini failed after retries")

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
        Generate intelligent summary.
        
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

        # Build prompt first to check actual size
        prompt = self._build_prompt(
            transcript, title, format, content_type, duration_minutes
        )

        # Gemini path: 1M token context, send everything in one call
        if self.provider == "gemini":
            result = self._summarize_gemini(
                prompt, word_count, title, content_type, duration_minutes, max_retries
            )
            if result is not None:
                return result
            # None means Gemini quota exhausted, provider switched to groq — continue below
            logger.info("Falling back to Groq after Gemini quota exhaustion")

        # Groq path: check if chunking is needed
        # Groq free-tier TPM limits are much lower than the model's 128K context
        # window, so sending huge prompts in one call causes repeated rate-limit
        # errors. Chunk early: ~30K tokens input ≈ 100K chars is a safe threshold.
        MAX_PROMPT_CHARS = 100000

        if len(prompt) > MAX_PROMPT_CHARS:
            logger.info(f"Prompt too large ({len(prompt):,} chars), using chunked processing")
            return self._summarize_long(
                transcript, title, format, content_type, duration_minutes
            )

        # Call Groq API with retries
        for attempt in range(max_retries):
            try:
                client = self._get_client()

                # Calculate safe max_tokens based on input length
                # Conservative estimate: 1 token ≈ 3.5 chars (accounts for special tokens)
                estimated_input_tokens = int(len(prompt) / 3.5)

                # Groq's limit: total tokens (input + output) must be < 128K
                # Reserve tokens for output based on format
                if format == SummaryFormat.QUICK:
                    desired_output = 2048
                elif format == SummaryFormat.BULLETS:
                    desired_output = 1024
                else:  # DETAILED or CHAPTERS
                    desired_output = 8192

                max_output_tokens = min(desired_output, 128000 - estimated_input_tokens - 2000)  # 2K safety margin

                if max_output_tokens < 512:
                    # This shouldn't happen if MAX_PROMPT_CHARS check worked, but just in case
                    logger.warning(f"Prompt too large ({estimated_input_tokens} tokens), falling back to chunking")
                    return self._summarize_long(
                        transcript, title, format, content_type, duration_minutes
                    )

                logger.debug(f"Estimated tokens - Input: {estimated_input_tokens:,}, Output: {max_output_tokens:,}")

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=max_output_tokens
                )

                raw = response.choices[0].message.content
                result = self._parse_response(raw, content_type)
                
                if result.success:
                    result.word_count_original = word_count
                    result.word_count_summary = len(result.executive_summary.split())
                    result.duration_minutes = duration_minutes
                    result.title = title
                    return result
                
                logger.warning(f"Attempt {attempt + 1} parse failed, retrying...")
                prompt = self._add_format_hint(prompt)
                
            except Exception as e:
                error_str = str(e).lower()

                if "rate_limit" in error_str:
                    # If the prompt itself is large, rate-limit retries won't help —
                    # the same request will exceed TPM again. Fall back to chunking.
                    if len(prompt) > 50000:
                        logger.warning("Rate limited on large prompt, switching to chunked processing")
                        return self._summarize_long(
                            transcript, title, format, content_type, duration_minutes
                        )
                    logger.warning("Rate limited, waiting 10s...")
                    time.sleep(10)
                elif "payload" in error_str or "context_length" in error_str or "too large" in error_str:
                    # Payload too large - force chunking even if under MAX_CONTEXT_CHARS
                    logger.warning(f"Payload too large, using chunked processing")
                    return self._summarize_long(
                        transcript, title, format, content_type, duration_minutes
                    )
                else:
                    logger.error(f"API error: {e}")
                    if attempt == max_retries - 1:
                        return Summary(success=False, error=str(e))
        
        return Summary(success=False, error="Failed after retries")
    
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
    def get_format_schema(format: 'SummaryFormat') -> str:
        """Return the format-specific JSON schema for summary output.

        Extracted as a public method so it can be reused by the combined
        Gemini transcribe+summarize path in main.py.
        """
        if format == SummaryFormat.QUICK:
            return '''Return JSON:
{
    "executive_summary": "2-3 paragraphs",
    "key_takeaways": ["5-7 key points"],
    "topics": ["main topics"]
}'''
        elif format == SummaryFormat.BULLETS:
            return '''Return JSON:
{
    "key_takeaways": ["7-10 key points as complete sentences"]
}'''
        elif format == SummaryFormat.CHAPTERS:
            return '''Return JSON:
{
    "chapters": [
        {"title": "Chapter Name", "timestamp": "MM:SS", "summary": "Description"}
    ],
    "executive_summary": "Brief overview"
}'''
        else:  # DETAILED
            return '''Return JSON:
{
    "executive_summary": "Comprehensive 2-3 paragraph summary",
    "key_takeaways": [
        "First key insight",
        "Second key insight",
        "Include 5-7 total"
    ],
    "chapters": [
        {"title": "Section Title", "timestamp": "MM:SS", "summary": "What's covered"}
    ],
    "notable_quotes": [
        {"text": "Exact quote", "speaker": "Who said it", "context": "Why notable"}
    ],
    "topics": ["topic1", "topic2"],
    "action_items": ["Recommendations or calls to action mentioned"]
}'''

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
        
        structure = self.get_format_schema(format)

        
        return f"""You are an expert content summarizer. Analyze this {content_type.value} transcript.

{"TITLE: " + title if title else ""}
{"DURATION: " + f"{duration:.0f} minutes" if duration else ""}
CONTENT TYPE: {content_type.value}

{type_instruction}

TRANSCRIPT:
{transcript}

{structure}

RULES:
1. Return ONLY valid JSON - no markdown, no extra text
2. Start with {{ and end with }}
3. Be specific and detailed, not generic
4. Include actual content from the transcript
5. For quotes, use exact words from the transcript"""
    
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
        
        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Try extracting from markdown
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        for p in patterns:
            match = re.search(p, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    continue
        
        # Try finding JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                # Fix trailing commas
                fixed = re.sub(r',\s*([}\]])', r'\1', match.group(0))
                return json.loads(fixed)
            except:
                pass
        
        return None
    
    def _summarize_long(
        self,
        transcript: str,
        title: str,
        format: SummaryFormat,
        content_type: ContentType,
        duration: float
    ) -> Summary:
        """
        Incremental refinement for long transcripts - faster than map-reduce.
        Processes transcript in overlapping windows, building up summary progressively.
        """

        logger.info(f"Long transcript ({len(transcript)} chars), using incremental summarization")

        # Groq free-tier TPM limit is 6,000 tokens/min for the fast model.
        # Keep each chunk under ~4K tokens (~14K chars) to leave room for
        # prompt overhead (~500 tokens) and output tokens (1500).
        chunk_size = 14000
        overlap = 1000

        chunks = []
        start = 0
        while start < len(transcript):
            end = min(start + chunk_size, len(transcript))
            # Try to end at sentence boundary
            if end < len(transcript):
                period = transcript.rfind('.', end - 500, end)
                if period > start:
                    end = period + 1
            chunks.append(transcript[start:end])
            if end >= len(transcript):
                break
            start = end - overlap

        logger.info(f"Processing {len(chunks)} chunks incrementally")

        # Incremental refinement
        client = self._get_client()
        running_summary = {
            'key_points': [],
            'topics': [],
            'quotes': [],
            'chapters': []
        }

        for i, chunk in enumerate(chunks):
            logger.info(f"Refining with chunk {i+1}/{len(chunks)}")

            # Build incremental prompt
            previous_context = ""
            if i > 0:
                previous_context = f"""
Previous summary so far:
- Topics: {', '.join(running_summary['topics'][:5])}
- Key points found: {len(running_summary['key_points'])}
"""

            prompt = f"""You are analyzing a {content_type.value} titled "{title}".
{previous_context}

Now process this {'next' if i > 0 else 'first'} section (part {i+1}/{len(chunks)}):

{chunk}

Extract and return JSON:
{{
    "new_topics": ["any new main topics in this section"],
    "new_key_points": ["2-3 new important points from this section"],
    "quotes": [{{"text": "notable quote", "context": "why it matters"}}],
    "section_summary": "Brief summary of this part"
}}

Focus on NEW information not already covered."""

            # Retry loop for rate limits
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=self.fast_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=1500
                    )

                    result = self._extract_json(response.choices[0].message.content)
                    if result:
                        # Accumulate findings
                        running_summary['topics'].extend(result.get('new_topics', []))
                        running_summary['key_points'].extend(result.get('new_key_points', []))
                        running_summary['quotes'].extend(result.get('quotes', []))

                        if result.get('section_summary'):
                            running_summary['chapters'].append({
                                'title': f"Part {i+1}",
                                'summary': result['section_summary']
                            })
                    break  # Success, exit retry loop

                except Exception as e:
                    error_str = str(e).lower()
                    if "rate_limit" in error_str and attempt < 2:
                        logger.warning(f"Chunk {i+1} rate limited, waiting 10s (attempt {attempt+1}/3)...")
                        time.sleep(10)
                    else:
                        logger.error(f"Chunk {i+1} failed: {e}")
                        break

            # Wait for TPM rolling window (6K TPM limit on Groq free tier).
            if i < len(chunks) - 1:
                logger.info(f"Waiting 15s for rate limit window...")
                time.sleep(15)

        # Final synthesis
        logger.info("Creating final summary from accumulated insights...")

        # Deduplicate and synthesize
        unique_topics = list(dict.fromkeys(running_summary['topics']))[:8]
        all_points = running_summary['key_points']

        synthesis_prompt = f"""Create a final executive summary for this {content_type.value}: "{title}"

Based on analysis of {len(chunks)} sections, here are the accumulated findings:

TOPICS: {', '.join(unique_topics)}

KEY POINTS:
{chr(10).join([f"- {p}" for p in all_points[:15]])}

SECTIONS:
{chr(10).join([f"{i+1}. {ch['summary']}" for i, ch in enumerate(running_summary['chapters'])])}

Return JSON:
{{
    "executive_summary": "Comprehensive 2-3 paragraph overview",
    "key_takeaways": ["Select and refine the 5-7 MOST important insights"],
    "topics": {unique_topics},
    "notable_quotes": ["Select 2-3 best quotes"],
    "chapters": [{{"title": "Section name", "timestamp": "00:00", "summary": "What's covered"}}]
}}"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3,
                max_tokens=4096
            )

            result = self._parse_response(response.choices[0].message.content, content_type)
            if result.success:
                result.title = title
                result.duration_minutes = duration
                result.word_count_original = len(transcript.split())
                return result

        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")

        # Fallback: return basic summary from accumulated data
        return Summary(
            success=True,
            title=title,
            content_type=content_type,
            executive_summary=f"Analysis of {len(chunks)} sections covering: " + ", ".join(unique_topics[:5]),
            key_takeaways=all_points[:7],
            topics=unique_topics,
            word_count_original=len(transcript.split()),
            duration_minutes=duration
        )


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
    format_map = {
        "detailed": SummaryFormat.DETAILED,
        "quick": SummaryFormat.QUICK,
        "bullets": SummaryFormat.BULLETS,
        "chapters": SummaryFormat.CHAPTERS,
    }
    
    summarizer = EnhancedSummarizer(**kwargs)
    return summarizer.summarize(
        transcript,
        title=title,
        format=format_map.get(format, SummaryFormat.DETAILED)
    )
