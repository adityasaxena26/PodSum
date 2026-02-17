# 🎙️ Podcast Summarizer MVP

> **Tier 1 Implementation:** YouTube transcripts → AI Summary  
> **Cost:** FREE | **Speed:** Instant | **GPU:** Not needed!

---

## ✨ What This Does

1. **Fetches** existing YouTube captions (no audio download!)
2. **Summarizes** using Llama 3.1 70B via Groq (free tier)
3. **Outputs** structured summary with takeaways, chapters, quotes

```
YouTube URL → Transcript (free, instant) → AI Summary → Markdown
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install youtube-transcript-api groq python-dotenv
```

Or:
```bash
pip install -r requirements.txt
```

### 2. Get Groq API Key (FREE)

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Create a free account
3. Generate an API key
4. Set it:

```bash
export GROQ_API_KEY='your-key-here'
```

### 3. Run It!

```bash
# Basic usage
python main.py https://www.youtube.com/watch?v=VIDEO_ID

# Quick summary (shorter output)
python main.py VIDEO_ID --style quick

# Save to file
python main.py VIDEO_ID --output summary.md

# Just get transcript (no API key needed)
python main.py VIDEO_ID --transcript-only
```

---

## 📖 Usage Examples

### CLI

```bash
# Full detailed summary
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Quick summary (executive summary + takeaways only)
python main.py dQw4w9WgXcQ --style quick

# Chapter breakdown
python main.py dQw4w9WgXcQ --style chapters

# Bullet points only
python main.py dQw4w9WgXcQ --style bullets

# Save output
python main.py dQw4w9WgXcQ -o my_summary.md

# Custom title
python main.py dQw4w9WgXcQ --title "My Favorite Video"

# Get transcript without summarizing (no API key needed)
python main.py dQw4w9WgXcQ --transcript-only
```

### Python API

```python
from main import PodcastSummarizerApp

# Initialize
app = PodcastSummarizerApp()

# Summarize
result = app.summarize_url(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    style="detailed"
)

if result['success']:
    summary = result['summary']
    print(summary.executive_summary)
    
    for takeaway in summary.key_takeaways:
        print(f"• {takeaway}")
```

---

## 📊 Summary Styles

| Style | What You Get | Best For |
|-------|--------------|----------|
| `detailed` | Everything - summary, takeaways, chapters, quotes | Deep analysis |
| `quick` | Summary + takeaways only | Quick overview |
| `bullets` | Just key points as bullets | Skimming |
| `chapters` | Chapter breakdown with timestamps | Navigation |

---

## 📁 Project Structure

```
podcast-summarizer-mvp/
├── main.py                           # CLI and main app
├── requirements.txt                  # Dependencies
├── README.md                         # This file
│
├── src/
│   ├── ingestion/
│   │   └── youtube_transcript.py     # YouTube transcript fetcher
│   │
│   └── summarization/
│       └── summarizer.py             # LLM summarizer
│
└── outputs/                          # Your saved summaries
```

---

## 🎯 Output Example

```markdown
# My Podcast Summary

**Video ID:** dQw4w9WgXcQ
**Duration:** 15.2 minutes
**Language:** en

---

## Executive Summary

This podcast discusses the fundamentals of machine learning...
[2-3 paragraphs]

---

## Key Takeaways

1. Start with the fundamentals before diving into advanced topics
2. Practice is more important than theory
3. ...

---

## Chapters

### [00:00] Introduction
Brief overview of what will be covered...

### [03:45] Main Topic
Deep dive into the core concepts...

---

## Notable Quotes

> "The best way to learn is by doing"
> *On the importance of hands-on practice*

---

## Topics

`machine learning` • `python` • `data science`
```

---

## ⚡ Why Tier 1?

| Approach | Speed | Cost | Compute |
|----------|-------|------|---------|
| **Tier 1 (This!)** | ~5 seconds | FREE | None |
| Tier 2 (Audio) | ~5 minutes | GPU time | High |

We use YouTube's existing transcripts instead of downloading audio and running Whisper. Same result, 60x faster, zero compute cost!

---

## 🔜 Coming in Tier 2

- [ ] Audio fallback (when no transcript available)
- [ ] Speaker diarization
- [ ] Prosody analysis (emphasis, emotion)
- [ ] Spotify, Apple Podcasts support
- [ ] Web UI

---

## 🐛 Troubleshooting

**"No transcript available"**
- Some videos have captions disabled
- Try another video, or wait for Tier 2 (audio fallback)

**"GROQ_API_KEY not set"**
- Get free key at [console.groq.com](https://console.groq.com)
- Run: `export GROQ_API_KEY='your-key'`

**"Rate limited"**
- Groq free tier: ~30 requests/minute
- Wait a minute and try again

**JSON parsing errors**
- Usually resolves on retry
- The code automatically retries 3 times

---

## 📄 License

MIT - Do whatever you want with it!

---

**Built for learning, shipped for using 🚀**
