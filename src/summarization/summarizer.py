"""
Podcast Summarizer MVP - LLM Summarizer
========================================
Summarize transcripts using Llama 3.1 via Groq API.

Groq free tier:
- ~30 requests/minute
- ~14,400 requests/day
- Fast inference (500+ tokens/sec)

Author: hobby-dev
Version: MVP 1.0
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

class SummaryStyle(Enum):
    """Different summary output styles"""
    DETAILED = "detailed"      # Full summary with all sections
    QUICK = "quick"            # Just executive summary + takeaways
    BULLETS = "bullets"        # Bullet points only
    CHAPTERS = "chapters"      # Focus on chapter breakdown


@dataclass
class Chapter:
    """A chapter/section of the podcast"""
    title: str
    timestamp: str  # "MM:SS" or "HH:MM:SS"
    summary: str


@dataclass
class Quote:
    """A notable quote from the podcast"""
    text: str
    context: str = ""
    timestamp: str = ""


@dataclass
class PodcastSummary:
    """Complete summary output"""
    success: bool
    
    # Core summary
    executive_summary: str = ""
    key_takeaways: List[str] = field(default_factory=list)
    
    # Structure
    chapters: List[Chapter] = field(default_factory=list)
    
    # Extras
    notable_quotes: List[Quote] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    
    # Metadata
    word_count_original: int = 0
    word_count_summary: int = 0
    compression_ratio: float = 0.0
    
    # Error
    error: Optional[str] = None
    raw_response: Optional[str] = None


class PodcastSummarizer:
    """
    Summarize podcast transcripts using Groq API.
    
    Example:
        >>> summarizer = PodcastSummarizer()
        >>> result = summarizer.summarize(transcript_text, title="My Podcast")
        >>> if result.success:
        ...     print(result.executive_summary)
        ...     for takeaway in result.key_takeaways:
        ...         print(f"• {takeaway}")
    """
    
    # Token limits (approximate)
    MAX_CONTEXT_TOKENS = 6000  # Leave room for response
    MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 4  # ~4 chars per token
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile"
    ):
        """
        Initialize summarizer.

        Args:
            api_key: Groq API key. If not provided, reads from groq_api_key env var.
            model: Model to use. Options:
                   - llama-3.3-70b-versatile (best quality)
                   - llama-3.1-8b-instant (faster, still good)
        """
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy load Groq client"""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("groq not installed. Run: pip install groq")
        return self._client
    
    def summarize(
        self,
        transcript: str,
        title: str = "Podcast",
        style: SummaryStyle = SummaryStyle.DETAILED,
        max_retries: int = 3
    ) -> PodcastSummary:
        """
        Generate summary from transcript.
        
        Args:
            transcript: Full transcript text
            title: Title of the podcast/video
            style: Summary style (detailed, quick, bullets, chapters)
            max_retries: Number of retry attempts for API/parsing errors
        
        Returns:
            PodcastSummary with all summary components
        """
        if not transcript or not transcript.strip():
            return PodcastSummary(success=False, error="Empty transcript")
        
        word_count = len(transcript.split())
        
        # Handle long transcripts
        if len(transcript) > self.MAX_CONTEXT_CHARS:
            print(f"Long transcript ({word_count} words), using chunked summarization...")
            return self._summarize_long(transcript, title, style)
        
        # Build prompt based on style
        prompt = self._build_prompt(transcript, title, style)
        
        # Try API call with retries
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=4096
                )
                
                raw_response = response.choices[0].message.content
                
                # Parse response
                result = self._parse_response(raw_response, style)
                
                if result.success:
                    # Add metadata
                    result.word_count_original = word_count
                    result.word_count_summary = len(result.executive_summary.split())
                    if result.word_count_summary > 0:
                        result.compression_ratio = word_count / result.word_count_summary
                    return result
                
                # Parsing failed, retry with modified prompt
                print(f"Attempt {attempt + 1} failed to parse, retrying...")
                prompt = self._add_format_hint(prompt)
                
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    import time
                    print("Rate limited, waiting 30 seconds...")
                    time.sleep(30)
                else:
                    print(f"API error: {e}")
                    if attempt == max_retries - 1:
                        return PodcastSummary(
                            success=False,
                            error=f"API error after {max_retries} attempts: {e}"
                        )
        
        return PodcastSummary(
            success=False,
            error="Failed after all retry attempts"
        )
    
    def _build_prompt(
        self,
        transcript: str,
        title: str,
        style: SummaryStyle
    ) -> str:
        """Build the summarization prompt"""
        
        if style == SummaryStyle.QUICK:
            return self._build_quick_prompt(transcript, title)
        elif style == SummaryStyle.BULLETS:
            return self._build_bullets_prompt(transcript, title)
        elif style == SummaryStyle.CHAPTERS:
            return self._build_chapters_prompt(transcript, title)
        else:  # DETAILED
            return self._build_detailed_prompt(transcript, title)
    
    def _build_detailed_prompt(self, transcript: str, title: str) -> str:
        """Full detailed summary prompt"""
        return f"""You are an expert podcast summarizer. Analyze this transcript and create a comprehensive summary.

PODCAST: {title}

TRANSCRIPT:
{transcript}

Create a JSON response with this exact structure:
{{
    "executive_summary": "A 2-3 paragraph summary capturing the main discussion, key arguments, and conclusions.",
    "key_takeaways": [
        "First key insight (one sentence)",
        "Second key insight",
        "Third key insight",
        "Fourth key insight",
        "Fifth key insight"
    ],
    "chapters": [
        {{"title": "Introduction", "timestamp": "00:00", "summary": "Brief description"}},
        {{"title": "Main Topic", "timestamp": "05:00", "summary": "What was discussed"}}
    ],
    "notable_quotes": [
        {{"text": "Exact quote from transcript", "context": "Why it's notable"}}
    ],
    "topics": ["topic1", "topic2", "topic3"],
    "action_items": ["Any recommendations or calls to action mentioned"]
}}

IMPORTANT RULES:
1. Return ONLY valid JSON - no markdown, no extra text
2. Start with {{ and end with }}
3. Use double quotes for all strings
4. Escape any quotes inside strings
5. Include 5-7 key takeaways
6. Include 3-6 chapters
7. Include 2-4 notable quotes"""
    
    def _build_quick_prompt(self, transcript: str, title: str) -> str:
        """Quick summary prompt"""
        return f"""Summarize this podcast transcript concisely.

PODCAST: {title}

TRANSCRIPT:
{transcript}

Return JSON:
{{
    "executive_summary": "2-3 paragraph summary",
    "key_takeaways": ["takeaway1", "takeaway2", "takeaway3", "takeaway4", "takeaway5"],
    "topics": ["topic1", "topic2"]
}}

Return ONLY valid JSON, starting with {{ and ending with }}."""
    
    def _build_bullets_prompt(self, transcript: str, title: str) -> str:
        """Bullet points only prompt"""
        return f"""Extract the key points from this podcast as bullet points.

PODCAST: {title}

TRANSCRIPT:
{transcript}

Return JSON:
{{
    "key_takeaways": [
        "First key point",
        "Second key point",
        "Continue for 7-10 key points total"
    ]
}}

Return ONLY valid JSON."""
    
    def _build_chapters_prompt(self, transcript: str, title: str) -> str:
        """Chapters focus prompt"""
        return f"""Break down this podcast into chapters/sections.

PODCAST: {title}

TRANSCRIPT:
{transcript}

Return JSON:
{{
    "chapters": [
        {{"title": "Chapter Title", "timestamp": "MM:SS", "summary": "What's covered"}},
        {{"title": "Next Chapter", "timestamp": "MM:SS", "summary": "Description"}}
    ],
    "executive_summary": "One paragraph overview"
}}

Return ONLY valid JSON."""
    
    def _add_format_hint(self, prompt: str) -> str:
        """Add extra formatting hints after failed attempt"""
        return prompt + """

PREVIOUS ATTEMPT HAD FORMATTING ERRORS. Please ensure:
- Start response with { character
- End response with } character  
- No markdown code blocks (no ```)
- No text before or after the JSON
- All strings use double quotes"""
    
    def _parse_response(self, raw_response: str, style: SummaryStyle) -> PodcastSummary:
        """Parse LLM response into PodcastSummary"""
        
        # Try to extract JSON
        json_data = self._extract_json(raw_response)
        
        if json_data is None:
            return PodcastSummary(
                success=False,
                error="Could not parse JSON from response",
                raw_response=raw_response[:500]
            )
        
        try:
            # Build summary from parsed JSON
            summary = PodcastSummary(success=True)
            
            # Executive summary
            summary.executive_summary = json_data.get('executive_summary', '')
            
            # Key takeaways
            summary.key_takeaways = json_data.get('key_takeaways', [])
            
            # Chapters
            chapters_raw = json_data.get('chapters', [])
            summary.chapters = [
                Chapter(
                    title=ch.get('title', 'Untitled'),
                    timestamp=ch.get('timestamp', '00:00'),
                    summary=ch.get('summary', '')
                )
                for ch in chapters_raw
            ]
            
            # Notable quotes
            quotes_raw = json_data.get('notable_quotes', [])
            summary.notable_quotes = [
                Quote(
                    text=q.get('text', q) if isinstance(q, dict) else str(q),
                    context=q.get('context', '') if isinstance(q, dict) else ''
                )
                for q in quotes_raw
            ]
            
            # Topics
            summary.topics = json_data.get('topics', [])
            
            # Action items
            summary.action_items = json_data.get('action_items', [])
            
            return summary
            
        except Exception as e:
            return PodcastSummary(
                success=False,
                error=f"Error building summary: {e}",
                raw_response=raw_response[:500]
            )
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text with multiple strategies"""
        text = text.strip()
        
        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Strategy 2: Find JSON in markdown code block
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    continue
        
        # Strategy 3: Find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        # Strategy 4: Try to fix common issues
        fixed = self._fix_json(text)
        if fixed:
            try:
                return json.loads(fixed)
            except:
                pass
        
        return None
    
    def _fix_json(self, text: str) -> Optional[str]:
        """Try to fix common JSON errors"""
        # Find the JSON part
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start == -1 or end == 0:
            return None
        
        json_str = text[start:end]
        
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _summarize_long(
        self,
        transcript: str,
        title: str,
        style: SummaryStyle
    ) -> PodcastSummary:
        """Handle long transcripts with map-reduce"""
        
        # Split into chunks
        chunk_size = 5000  # characters
        chunks = []
        
        words = transcript.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"Split into {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = []
        client = self._get_client()
        
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            
            prompt = f"""Summarize this section of a podcast (part {i+1} of {len(chunks)}):

{chunk}

Provide a brief summary (2-3 sentences) of the main points discussed."""
            
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",  # Use faster model for chunks
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                chunk_summaries.append(response.choices[0].message.content)
            except Exception as e:
                chunk_summaries.append(f"[Error summarizing chunk {i+1}]")
        
        # Combine chunk summaries
        combined = "\n\n".join([
            f"PART {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
        ])
        
        # Final summary
        print("Generating final summary...")
        return self.summarize(combined, title=title, style=style)


def summarize_transcript(
    transcript: str,
    title: str = "Podcast",
    style: str = "detailed",
    api_key: Optional[str] = None
) -> PodcastSummary:
    """
    Convenience function to summarize a transcript.
    
    Args:
        transcript: Transcript text
        title: Podcast title
        style: "detailed", "quick", "bullets", or "chapters"
        api_key: Groq API key (or set groq_api_key env var)
    
    Returns:
        PodcastSummary
    """
    style_map = {
        "detailed": SummaryStyle.DETAILED,
        "quick": SummaryStyle.QUICK,
        "bullets": SummaryStyle.BULLETS,
        "chapters": SummaryStyle.CHAPTERS
    }
    
    summarizer = PodcastSummarizer(api_key=api_key)
    return summarizer.summarize(
        transcript,
        title=title,
        style=style_map.get(style, SummaryStyle.DETAILED)
    )


# CLI interface
if __name__ == "__main__":
    # Test with sample transcript
    sample_transcript = """
    Welcome to the AI Podcast! Today we're discussing the future of artificial intelligence
    with our guest Dr. Sarah Chen, a leading researcher in machine learning.
    
    Host: Sarah, what do you think is the most exciting development in AI right now?
    
    Dr. Chen: I think large language models have really changed the game. We're seeing
    capabilities that were science fiction just five years ago. The ability to understand
    and generate human-like text opens up so many possibilities.
    
    Host: Are there concerns about these technologies?
    
    Dr. Chen: Absolutely. We need to think carefully about safety, bias, and misuse.
    I always tell my students that with great power comes great responsibility.
    The key is building safeguards from the start, not as an afterthought.
    
    Host: What advice would you give to someone wanting to get into AI?
    
    Dr. Chen: Start with the fundamentals - linear algebra, statistics, Python.
    Then pick a specific area and dive deep. The field is vast, so finding your
    niche is important. And always keep learning - things change fast!
    
    Host: Thanks so much for joining us, Sarah!
    """
    
    print("Testing PodcastSummarizer...")
    print("=" * 50)
    
    try:
        summarizer = PodcastSummarizer()
        result = summarizer.summarize(sample_transcript, title="AI Podcast with Dr. Chen")
        
        if result.success:
            print("\n✅ SUCCESS!\n")
            print("EXECUTIVE SUMMARY:")
            print("-" * 40)
            print(result.executive_summary)
            print("\nKEY TAKEAWAYS:")
            print("-" * 40)
            for i, takeaway in enumerate(result.key_takeaways, 1):
                print(f"{i}. {takeaway}")
            print("\nTOPICS:", ", ".join(result.topics))
        else:
            print(f"\n❌ FAILED: {result.error}")
            
    except ValueError as e:
        print(f"\n⚠️ {e}")
        print("\nTo test, set your Groq API key:")
        print("  export groq_api_key='your-key-here'")
