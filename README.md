# Podcast Summarizer v2.0

> Multi-platform video & podcast summarizer with intelligent fallback and dual LLM support.

Summarize content from **YouTube, Vimeo, Twitter, TikTok, Spotify, and 1000+ more sites** using AI. Supports both **Google Gemini** (recommended) and **Groq** as LLM providers.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set API Key

**Option A — Google Gemini (recommended):**

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```bash
export GEMINI_API_KEY='your-key-here'
```

**Option B — Groq:**

Get a free key at [console.groq.com](https://console.groq.com/keys)

```bash
export GROQ_API_KEY='your-key-here'
```

Or add them to a `.env` file in the project root.

### 3. Run

```bash
# YouTube video (uses captions — instant)
python -m src.main https://www.youtube.com/watch?v=VIDEO_ID

# Any supported site
python -m src.main https://vimeo.com/123456789

# Local file
python -m src.main --file podcast.mp3

# Save output
python -m src.main URL --output summary.md
```

**Web UI:**
```bash
python -m src.app
# Open http://localhost:7860
```

---

## LLM Providers

The app auto-selects the best available provider, or you can force one with `--provider`:

| Provider | Model | Context Window | Speed | Free Tier | Notes |
|----------|-------|---------------|-------|-----------|-------|
| **Gemini** | `gemini-2.5-flash-lite` | 1M tokens | Fast | 15 RPM, 1M TPM | Handles 4+ hour podcasts in a single call |
| **Groq** | `llama-3.3-70b-versatile` (primary) / `llama-3.1-8b-instant` (chunked) | 128K tokens | Fast | 6K TPM | Uses chunked processing for long content |

```bash
# Auto-select (default — prefers Gemini)
python -m src.main URL

# Force a specific provider
python -m src.main URL --provider gemini
python -m src.main URL --provider groq
```

If Gemini quota is exhausted, the app automatically falls back to Groq.

---

## CLI Options

```bash
python -m src.main <URL> [options]

Options:
  --format FORMAT        Summary format: detailed, quick, bullets, chapters (default: detailed)
  --output FILE, -o      Save to file (.md or .json)
  --file FILE, -f        Summarize a local audio/video file
  --title TITLE, -t      Custom title override
  --provider PROVIDER    LLM provider: auto, gemini, groq (default: auto)
  --force-audio          Skip platform captions, use Whisper
  --transcript-only      Only fetch transcript, no summarization
  --whisper-model MODEL  Whisper model: tiny, base, small, medium, large-v2, large-v3 (default: large-v3)
  --quiet, -q            Minimal output
```

### Examples

```bash
# Quick summary
python -m src.main URL --format quick

# Bullet points saved to file
python -m src.main URL --format bullets --output notes.md

# JSON output
python -m src.main URL --output data.json

# Transcript only (no API key needed)
python -m src.main URL --transcript-only

# Force audio transcription with a smaller Whisper model
python -m src.main URL --force-audio --whisper-model small
```

---

## Python API

```python
from src.main import PodcastSummarizerV2

app = PodcastSummarizerV2()

# Summarize a URL
result = app.summarize_url("https://youtube.com/watch?v=...", format="detailed")

if result['success']:
    summary = result['summary']
    print(summary.executive_summary)
    for takeaway in summary.key_takeaways:
        print(f"- {takeaway}")

# Summarize a local file
result = app.summarize_file("podcast.mp3", format="quick")

# Export as markdown
markdown = app.format_markdown(result['transcript'], result['summary'])
```

---

## Supported Platforms

**Tier 1 — Platform Transcripts (free, instant):**
- YouTube (captions API)

**Tier 2 — Audio Download + Whisper (fallback):**
- Vimeo, Twitter/X, TikTok, Spotify, SoundCloud, Twitch, Facebook, Instagram, Dailymotion, and 1000+ more via yt-dlp

**Local Files:**
- Audio: MP3, WAV, M4A, FLAC, OGG
- Video: MP4, MKV, WebM, AVI
- Any format supported by FFmpeg

---

## Summary Formats

| Format | Output | Best For |
|--------|--------|----------|
| `detailed` | Full summary, chapters, quotes, takeaways | Deep analysis |
| `quick` | Executive summary + key takeaways | Quick overview |
| `bullets` | Key points as bullet list | Scanning |
| `chapters` | Timeline with chapter breakdown | Navigation |

---

## Content Type Detection

The summarizer auto-detects content type and adapts its output:

| Type | Focus Areas |
|------|-------------|
| Podcast | Host/guest dynamics, main topics |
| Interview | Interviewee insights, quotes |
| Tutorial | Steps, concepts, how-to |
| Lecture | Theories, definitions, examples |
| News | Facts, sources, implications |
| Commentary | Opinions, arguments, perspectives |
| Entertainment | Highlights, moments, reactions |
| General | Main points, key takeaways |

---

## How It Works

```
INPUT: URL or local file
         |
         v
  TIER 1: Platform Check
  YouTube? -> Fetch captions (free, instant)
         |
     Found? --YES--> SUMMARIZE
         |
        NO
         v
  TIER 2: Audio Fallback
  1. Download audio (yt-dlp)
  2. Transcribe (faster-whisper)
         |
         v
  SUMMARIZE
  1. Detect content type
  2. Gemini: single call (1M context)
     Groq: chunked processing if needed
         |
         v
  OUTPUT: Markdown / JSON
  Summary, chapters, quotes, takeaways
```

---

## Project Structure

```
PodsumV2/
├── src/
│   ├── main.py                    # CLI application
│   ├── app.py                     # Gradio web UI
│   ├── ingestion/
│   │   └── multi_platform.py      # Multi-platform transcript fetcher
│   └── summarization/
│       └── summarizer.py          # LLM summarizer (Gemini + Groq)
├── requirements.txt
├── .env                           # API keys
└── README.md
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | One of these | Google Gemini API key (recommended) |
| `GROQ_API_KEY` | is required | Groq API key (fallback) |

### Whisper Models

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| `tiny` | Fastest | Low | ~1GB |
| `base` | Fast | Medium | ~1GB |
| `small` | Medium | Good | ~2GB |
| `medium` | Slow | High | ~5GB |
| `large-v3` | Slowest | Best | ~5GB |

CLI default: `large-v3` | Web UI default: `base`

---

## Cost

| Component | Cost |
|-----------|------|
| YouTube transcripts | Free |
| Audio download (yt-dlp) | Free |
| Whisper transcription | Free (local) |
| Gemini API | Free tier available |
| Groq API | Free tier available |

---

## Troubleshooting

**"No API key set"** — Set at least one:
```bash
export GEMINI_API_KEY='your-key'   # or
export GROQ_API_KEY='your-key'
```

**"Gemini quota exhausted"** — The app auto-falls back to Groq. Or wait for daily quota reset.

**"No transcript available"** — The app automatically falls back to audio download + Whisper transcription.

**"FFmpeg not found"** — Install it:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

**"CUDA out of memory"** — Use a smaller Whisper model:
```bash
python -m src.main URL --whisper-model small
```

**Slow summarization on Groq** — Groq's free tier has tight rate limits (6K TPM), requiring delays between chunks for long podcasts. Use Gemini instead, or upgrade to Groq's Dev tier.

---

## License

MIT
