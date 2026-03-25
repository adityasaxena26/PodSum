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
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from main import PodcastSummarizerV2


# Initialize app (lazy loaded, per whisper model)
_app = None
_app_whisper_model = None

def get_app(whisper_model="base"):
    global _app, _app_whisper_model
    if _app is None or _app_whisper_model != whisper_model:
        _app = PodcastSummarizerV2(whisper_model=whisper_model)
        _app_whisper_model = whisper_model
    return _app


def process_url(
    url: str,
    format_choice: str,
    custom_title: str,
    force_audio: bool,
    whisper_model: str,
    progress=gr.Progress()
):
    """Process URL and generate summary"""
    
    if not url or not url.strip():
        return "❌ Please enter a URL", "", "", ""
    
    # Check API key
    if not os.environ.get('GEMINI_API_KEY') and not os.environ.get('GROQ_API_KEY'):
        return (
            "❌ No API key set!\n\n"
            "Option 1 (recommended): export GEMINI_API_KEY='your-key'\n"
            "Get free key at: aistudio.google.com/apikey\n\n"
            "Option 2: export GROQ_API_KEY='your-key'\n"
            "Get free key at: console.groq.com",
            "", "", ""
        )

    app = get_app(whisper_model=whisper_model)

    # Progress callback
    def update_progress(msg, pct):
        progress(pct, desc=msg)

    try:
        result = app.summarize_url(
            url=url.strip(),
            format=format_choice,
            title=custom_title.strip() if custom_title and custom_title.strip() else None,
            force_audio=force_audio,
            progress_callback=update_progress
        )
        
        if not result['success']:
            return f"❌ Error: {result['error']}", "", "", ""
        
        transcript = result['transcript']
        summary = result['summary']
        
        # Format outputs
        status = (
            f"✅ **Success!**\n\n"
            f"- **Source:** {transcript.source.value}\n"
            f"- **Platform:** {transcript.platform}\n"
            f"- **Duration:** {transcript.duration_minutes:.1f} minutes\n"
            f"- **Words:** {transcript.word_count:,}\n"
            f"- **Content Type:** {summary.content_type.value}\n"
            f"- **Compression:** {summary.compression_ratio:.1f}x"
        )
        
        summary_md = app.format_markdown(transcript, summary)
        transcript_text = transcript.text
        
        # Create downloadable content
        download_content = summary_md
        
        return status, summary_md, transcript_text, download_content
    
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""


def process_file(
    file,
    format_choice: str,
    custom_title: str,
    whisper_model: str,
    progress=gr.Progress()
):
    """Process uploaded file"""
    
    if file is None:
        return "❌ Please upload a file", "", "", ""
    
    if not os.environ.get('GEMINI_API_KEY') and not os.environ.get('GROQ_API_KEY'):
        return (
            "❌ No API key set!\n\n"
            "Option 1 (recommended): export GEMINI_API_KEY='your-key'\n"
            "Option 2: export GROQ_API_KEY='your-key'",
            "", "", ""
        )
    
    app = get_app(whisper_model=whisper_model)

    def update_progress(msg, pct):
        progress(pct, desc=msg)

    try:
        # Get file path
        file_path = file.name if hasattr(file, 'name') else file
        
        result = app.summarize_file(
            file_path=file_path,
            format=format_choice,
            title=custom_title.strip() if custom_title and custom_title.strip() else None,
            progress_callback=update_progress
        )
        
        if not result['success']:
            return f"❌ Error: {result['error']}", "", "", ""
        
        transcript = result['transcript']
        summary = result['summary']
        
        status = (
            f"✅ **Success!**\n\n"
            f"- **Source:** {transcript.source.value}\n"
            f"- **Duration:** {transcript.duration_minutes:.1f} minutes\n"
            f"- **Words:** {transcript.word_count:,}\n"
            f"- **Content Type:** {summary.content_type.value}\n"
            f"- **Compression:** {summary.compression_ratio:.1f}x"
        )
        
        summary_md = app.format_markdown(transcript, summary)
        transcript_text = transcript.text
        download_content = summary_md
        
        return status, summary_md, transcript_text, download_content
    
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""


# Custom CSS
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}
.status-box {
    padding: 10px;
    border-radius: 8px;
}
footer {
    visibility: hidden;
}
"""

# Build interface
with gr.Blocks(
    title="🎙️ Podcast Summarizer v2.0",
) as app:
    
    gr.Markdown("""
    # 🎙️ Podcast Summarizer v2.0
    
    **Summarize any video or podcast with AI**
    
    Supports: YouTube, Vimeo, Twitter, TikTok, Spotify, SoundCloud, and 1000+ more sites!
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            
            with gr.Tabs() as input_tabs:
                
                # URL Tab
                with gr.Tab("🔗 From URL", id="url_tab"):
                    url_input = gr.Textbox(
                        label="Enter URL",
                        placeholder="https://www.youtube.com/watch?v=... or any supported URL",
                        lines=1
                    )
                    
                    with gr.Row():
                        force_audio = gr.Checkbox(
                            label="Force audio transcription",
                            value=False,
                            info="Skip platform captions, use Whisper"
                        )
                    
                    url_btn = gr.Button("🚀 Summarize URL", variant="primary", size="lg")
                
                # File Tab
                with gr.Tab("📁 Upload File", id="file_tab"):
                    file_input = gr.File(
                        label="Upload audio or video file",
                        file_types=["audio", "video"],
                        type="filepath"
                    )
                    
                    file_btn = gr.Button("🚀 Summarize File", variant="primary", size="lg")
            
            # Common options
            with gr.Accordion("⚙️ Options", open=False):
                format_choice = gr.Radio(
                    choices=["detailed", "quick", "bullets", "chapters"],
                    value="detailed",
                    label="Summary Format",
                    info="detailed=full, quick=brief, bullets=points only, chapters=timeline"
                )
                
                custom_title = gr.Textbox(
                    label="Custom Title (optional)",
                    placeholder="Leave empty to auto-detect"
                )

                whisper_model = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    value="base",
                    label="Whisper Model",
                    info="For audio transcription. tiny=fastest, large-v3=best quality"
                )
        
        # Status
        with gr.Column(scale=1):
            status_output = gr.Markdown(
                label="Status",
                value="Ready! Enter a URL or upload a file."
            )
    
    gr.Markdown("---")
    
    # Output tabs
    with gr.Tabs():
        with gr.Tab("📝 Summary"):
            summary_output = gr.Markdown(label="Summary")
        
        with gr.Tab("📄 Full Transcript"):
            transcript_output = gr.Textbox(
                label="Transcript",
                lines=20,
                max_lines=50
            )
        
        with gr.Tab("💾 Download"):
            download_output = gr.Textbox(
                label="Markdown (copy or save)",
                lines=20
            )
    
    # Event handlers
    url_btn.click(
        fn=process_url,
        inputs=[url_input, format_choice, custom_title, force_audio, whisper_model],
        outputs=[status_output, summary_output, transcript_output, download_output]
    )
    
    file_btn.click(
        fn=process_file,
        inputs=[file_input, format_choice, custom_title, whisper_model],
        outputs=[status_output, summary_output, transcript_output, download_output]
    )
    
    # Examples
    gr.Markdown("---")
    gr.Markdown("### 💡 Examples")
    gr.Examples(
        examples=[
            ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "quick", ""],
        ],
        inputs=[url_input, format_choice, custom_title],
        label=""
    )
    
    # Footer
    gr.Markdown("""
    ---
    
    **Tips:**
    - YouTube videos use existing captions (instant!)
    - Other sites download audio and transcribe with Whisper
    - Use "Force audio" if captions are poor quality
    - Set `GROQ_API_KEY` environment variable before running
    
    **Powered by:** yt-dlp, faster-whisper, Google Gemini / Groq
    """)


def main():
    """Launch the web interface"""
    
    print("=" * 50)
    print("🎙️ Podcast Summarizer v2.0 - Web UI")
    print("=" * 50)
    
    # Check API key
    if not os.environ.get('GEMINI_API_KEY') and not os.environ.get('GROQ_API_KEY'):
        print("\n⚠️  Warning: No API key set!")
        print("   Set GEMINI_API_KEY (recommended) or GROQ_API_KEY")
        print("   Gemini key: aistudio.google.com/apikey")
        print("   Groq key: console.groq.com\n")
    
    print("\nStarting server...")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,  # Set True for public link
        show_error=True,
        theme=gr.themes.Soft(primary_hue="purple"),
        css=css
    )


if __name__ == "__main__":
    main()
