# Podcast Summarizer v2.0

> Multi-platform video & podcast summarizer with intelligent fallback, powered by Google Gemini.

Summarize content from **YouTube, Vimeo, Twitter, TikTok, SoundCloud, and 1000+ more sites** using AI, or summarize a local audio/video file.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

You also need FFmpeg for audio downloads and local files (see [Troubleshooting](#troubleshooting)).

### 2. Set API Key

Get a free Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```bash
export GEMINI_API_KEY='your-key-here'
```

Or add it to a `.env` file in the project root.

### 3. Run

```bash
# YouTube video
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

## LLM

| Provider | Model | Context Window | Notes |
|----------|-------|---------------|-------|
| **Gemini** | `gemini-2.5-flash` (override with `GEMINI_MODEL`) | 1M tokens | Can read YouTube videos directly when captions are missing |

---

## CLI Options

```bash
python -m src.main <URL> [options]

Options:
  --format FORMAT        Summary format: detailed, quick, bullets, chapters (default: detailed)
  --output FILE, -o      Save to file (.md or .json)
  --file FILE, -f        Summarize a local audio/video file
  --title TITLE, -t      Custom title override
  --force-audio          Skip transcript APIs, download audio and use Whisper
  --transcript-only      Only fetch transcript, no summarization
  --whisper-model MODEL  Whisper model: tiny, base, small, medium, large-v2, large-v3 (default: small)
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

# Transcript only
python -m src.main URL --transcript-only

# Force audio transcription with a larger Whisper model
python -m src.main URL --force-audio --whisper-model medium
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

**YouTube:**
- Captions via `youtube-transcript-api` (free, instant — works locally, usually blocked from cloud servers)
- Gemini transcription / video understanding (works from cloud servers)
- Audio download + Whisper as a last resort

**Everything else — Audio Download + Whisper:**
- Vimeo, Twitter/X, TikTok, SoundCloud, Twitch, Facebook, Instagram, Dailymotion, and 1000+ more via yt-dlp
- Spotify only works for episodes yt-dlp can download (no DRM-protected content, no transcript API)

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
  YouTube URL? ──YES──> FAST PATH (one Gemini call)
         |              1. Fetch captions + video metadata in parallel
         |              2. Captions good (>50 words/min)?
         |                   YES -> Gemini summarizes the text
         |                          (very long videos: Gemini watches the video)
         |                   NO  -> Gemini watches the video directly
         |              Success? ──YES──> OUTPUT
         |                 |
         NO                NO (fall back)
         v                 v
  TRANSCRIPT FETCH (tiers tried in order)
  1. Gemini transcription  (YouTube only)
  2. YouTube captions API
  3. Download audio (yt-dlp) + transcribe (faster-whisper)
         |
         v
  SUMMARIZE
  1. Detect content type
  2. Long transcripts (>8,000 words): sample intro, middle sections, outro
  3. Gemini single call (1M context), retries with backoff on rate limits
         |
         v
  OUTPUT: Markdown / JSON
  Summary, chapters, quotes, takeaways
```

Results are cached in memory (last 32 URL + format combinations), so repeat requests return instantly.

---

## Project Structure

```
PodSum/
├── src/
│   ├── main.py                    # Orchestrator + CLI
│   ├── app.py                     # Gradio web UI
│   ├── ingestion/
│   │   └── multi_platform.py      # Multi-platform transcript fetcher
│   └── summarization/
│       └── summarizer.py          # Gemini summarizer
├── Dockerfile
├── docker-compose.yml
├── deploy.sh                      # Deploy to Google Cloud Run
├── requirements.txt
├── .env                           # API keys (not committed)
└── README.md
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `GEMINI_MODEL` | No | Gemini model to use (default: `gemini-2.5-flash`) |
| `YOUTUBE_API_KEY` | No | YouTube Data API v3 key, used to fetch video title and duration |
| `WHISPER_MODEL` | No | Whisper model for the web UI (default: `small`) |
| `PORT` | No | Web UI port (default: `7860`) |

### Whisper Models

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| `tiny` | Fastest | Low | ~1GB |
| `base` | Fast | Medium | ~1GB |
| `small` | Medium | Good | ~2GB |
| `medium` | Slow | High | ~5GB |
| `large-v3` | Slowest | Best | ~5GB |

Default: `small` (CLI and web UI)

---

## Deployment

**Docker:**
```bash
docker-compose up --build
# Open http://localhost:7860
```
The `small` Whisper model is baked into the image to avoid a slow first request.

**Google Cloud Run:**
```bash
./deploy.sh
```
Requires the `gcloud` CLI, a selected GCP project, and `GEMINI_API_KEY` in your environment or `.env`.

---

## Cost

| Component | Cost |
|-----------|------|
| YouTube transcripts | Free |
| Audio download (yt-dlp) | Free |
| Whisper transcription | Free (local) |
| Gemini API | Free tier available |

---

## Troubleshooting

**"No API key set"** — Set your Gemini key:
```bash
export GEMINI_API_KEY='your-key'
```

**"Gemini quota exhausted"** — Wait for the quota to reset or upgrade your Gemini plan.

**"No transcript available"** — The app automatically falls back to audio download + Whisper transcription. On cloud servers YouTube often blocks captions and audio downloads; Gemini transcription is the reliable path there.

**"FFmpeg not found"** — Install it:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

**"CUDA out of memory"** — Use a smaller Whisper model:
```bash
python -m src.main URL --whisper-model base
```

---

## License

MIT
