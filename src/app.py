"""
Podcast Summarizer v2.0 - Web Interface
========================================
Gradio-based web UI for the podcast summarizer.

Features:
- URL input for any supported platform
- File upload for local audio/video
- Real-time progress tracking
- Multiple output formats
- Download results as markdown

Version: 2.0.0
"""

import gradio as gr
import os
import sys
import tempfile
import threading
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from main import PodcastSummarizerV2


# ── Thread-safe app initialization ──────────────────────────────────
_app_lock = threading.Lock()
_app = None
_app_whisper_model = None

# Read default whisper model from env (matches Dockerfile/docker-compose)
_DEFAULT_WHISPER = os.environ.get('WHISPER_MODEL', 'small')


def get_app(whisper_model=None):
    whisper_model = whisper_model or _DEFAULT_WHISPER
    global _app, _app_whisper_model
    with _app_lock:
        if _app is None or _app_whisper_model != whisper_model:
            _app = PodcastSummarizerV2(whisper_model=whisper_model)
            _app_whisper_model = whisper_model
        return _app


# ── Helpers ─────────────────────────────────────────────────────────

def _check_api_keys():
    """Validate API keys are set. Returns error string or None."""
    if not os.environ.get('GEMINI_API_KEY') and not os.environ.get('GROQ_API_KEY'):
        return (
            "**No API key configured.**\n\n"
            "Set one of these environment variables before starting the app:\n\n"
            "- `GEMINI_API_KEY` (recommended, free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))\n"
            "- `GROQ_API_KEY` (free at [console.groq.com](https://console.groq.com))"
        )
    return None


_ERROR_MAP = {
    "quota exhausted": "The AI service is temporarily at capacity. Please try again in a few minutes.",
    "could not extract youtube video id": "This doesn't look like a valid YouTube URL. Please check and try again.",
    "no transcript available": "This video has no captions available. Try enabling **Force audio transcription**.",
    "unsupported url scheme": "Only http:// and https:// URLs are supported.",
    "gemini failed after retries": "The AI service failed to respond. Please try again shortly.",
    "json parse failed": "The AI returned an unexpected response. Please try again.",
    "empty transcript": "Could not extract any text from this content.",
}


def _friendly_error(raw_error: str) -> str:
    """Convert technical errors to user-friendly messages."""
    lower = raw_error.lower()
    for pattern, friendly in _ERROR_MAP.items():
        if pattern in lower:
            return friendly
    return raw_error


def _format_result(result, app_instance):
    """Format a successful result into (status, summary_md, transcript, download_path)."""
    transcript = result['transcript']
    summary = result['summary']

    status = (
        f"**Done!**\n\n"
        f"| | |\n|---|---|\n"
        f"| Source | {transcript.source.value} |\n"
        f"| Platform | {transcript.platform} |\n"
        f"| Duration | {transcript.duration_minutes:.1f} min |\n"
        f"| Words | {transcript.word_count:,} |\n"
        f"| Content Type | {summary.content_type.value} |\n"
        f"| Compression | {summary.compression_ratio:.1f}x |"
    )

    summary_md = app_instance.format_markdown(transcript, summary)
    transcript_text = transcript.text

    # Write markdown to a temp file for download
    download_path = None
    try:
        safe_title = "".join(
            c for c in (summary.title or "summary") if c.isalnum() or c in " -_"
        ).strip()[:60] or "summary"
        fd, download_path = tempfile.mkstemp(suffix=".md", prefix=f"{safe_title}_")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(summary_md)
    except Exception:
        download_path = None

    return status, summary_md, transcript_text, download_path


# ── Processing functions ────────────────────────────────────────────

def process_url(
    url: str,
    format_choice: str,
    custom_title: str,
    force_audio: bool,
    whisper_model: str,
    progress=gr.Progress()
):
    """Process URL and generate summary."""
    if not url or not url.strip():
        return "Enter a URL above to get started.", "", "", None

    # Basic URL format check
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        return "Please enter a valid URL starting with http:// or https://", "", "", None

    key_err = _check_api_keys()
    if key_err:
        return key_err, "", "", None

    app_instance = get_app(whisper_model=whisper_model)

    def update_progress(msg, pct):
        progress(pct, desc=msg)

    try:
        result = app_instance.summarize_url(
            url=url,
            format=format_choice,
            title=custom_title.strip() if custom_title and custom_title.strip() else None,
            force_audio=force_audio,
            progress_callback=update_progress
        )

        if not result['success']:
            return _friendly_error(result['error']), "", "", None

        return _format_result(result, app_instance)

    except Exception as e:
        return _friendly_error(str(e)), "", "", None


def process_file(
    file,
    format_choice: str,
    custom_title: str,
    whisper_model: str,
    progress=gr.Progress()
):
    """Process uploaded file."""
    if file is None:
        return "Upload an audio or video file to get started.", "", "", None

    key_err = _check_api_keys()
    if key_err:
        return key_err, "", "", None

    app_instance = get_app(whisper_model=whisper_model)

    def update_progress(msg, pct):
        progress(pct, desc=msg)

    try:
        file_path = file.name if hasattr(file, 'name') else file

        result = app_instance.summarize_file(
            file_path=file_path,
            format=format_choice,
            title=custom_title.strip() if custom_title and custom_title.strip() else None,
            progress_callback=update_progress
        )

        if not result['success']:
            return _friendly_error(result['error']), "", "", None

        return _format_result(result, app_instance)

    except Exception as e:
        return _friendly_error(str(e)), "", "", None


# ── Custom CSS ──────────────────────────────────────────────────────
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}
footer {
    visibility: hidden;
}
"""

# ── Build interface ─────────────────────────────────────────────────
with gr.Blocks(
    title="Podcast Summarizer v2.0",
) as demo:

    gr.Markdown("""
    # Podcast Summarizer v2.0
    Summarize any video or podcast with AI. Paste a URL or upload a file.
    """)

    with gr.Row():
        with gr.Column(scale=3):

            with gr.Tabs() as input_tabs:

                with gr.Tab("From URL", id="url_tab"):
                    url_input = gr.Textbox(
                        label="Video / Podcast URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1,
                    )
                    with gr.Row():
                        force_audio = gr.Checkbox(
                            label="Force audio transcription",
                            value=False,
                            info="Skip captions, use Whisper speech recognition"
                        )
                    url_btn = gr.Button("Summarize", variant="primary", size="lg")

                with gr.Tab("Upload File", id="file_tab"):
                    file_input = gr.File(
                        label="Audio or video file",
                        file_types=["audio", "video"],
                        type="filepath"
                    )
                    file_btn = gr.Button("Summarize", variant="primary", size="lg")

            with gr.Accordion("Options", open=False):
                format_choice = gr.Radio(
                    choices=["detailed", "quick", "bullets", "chapters"],
                    value="detailed",
                    label="Summary Format",
                    info="detailed = full summary with chapters & quotes | quick = executive summary only | bullets = key points | chapters = timeline"
                )
                custom_title = gr.Textbox(
                    label="Custom Title (optional)",
                    placeholder="Leave empty to auto-detect from video"
                )
                whisper_model = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    value=_DEFAULT_WHISPER,
                    label="Whisper Model",
                    info="For audio transcription. tiny = fastest, large-v3 = best quality"
                )

        # Status sidebar
        with gr.Column(scale=1, min_width=250):
            status_output = gr.Markdown(
                value="**Ready.** Enter a URL or upload a file.",
                label="Status",
            )

    # ── Output area ─────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("Summary"):
            summary_output = gr.Markdown(label="Summary")

        with gr.Tab("Full Transcript"):
            transcript_output = gr.Textbox(
                label="Transcript",
                lines=20,
                max_lines=50,
            )

        with gr.Tab("Download"):
            download_output = gr.File(label="Download Summary (.md)")

    # ── Event handlers ──────────────────────────────────────────
    url_btn.click(
        fn=process_url,
        inputs=[url_input, format_choice, custom_title, force_audio, whisper_model],
        outputs=[status_output, summary_output, transcript_output, download_output],
    )

    file_btn.click(
        fn=process_file,
        inputs=[file_input, format_choice, custom_title, whisper_model],
        outputs=[status_output, summary_output, transcript_output, download_output],
    )

    # ── Footer ──────────────────────────────────────────────────
    gr.Markdown("""
    ---
    **How it works:**
    YouTube with captions (~5-10 s) | YouTube cloud / no captions (~30-35 s) | Other sites: audio download + Whisper + AI

    **Powered by** yt-dlp, faster-whisper, Google Gemini, Groq
    """)


# ── Entry point ─────────────────────────────────────────────────────

def main():
    """Launch the web interface."""
    print("=" * 50)
    print("Podcast Summarizer v2.0")
    print("=" * 50)

    if not os.environ.get('GEMINI_API_KEY') and not os.environ.get('GROQ_API_KEY'):
        print("\nWarning: No API key set!")
        print("  Set GEMINI_API_KEY (recommended) or GROQ_API_KEY\n")

    print("Starting server...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="purple"),
        css=css,
    )


if __name__ == "__main__":
    main()
