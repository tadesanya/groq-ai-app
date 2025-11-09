import streamlit as st
import os
from dotenv import load_dotenv
from yt_vid_summarizer import YouTubeTranscriptSummarizer

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF0000;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #FF0000;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Model type selection
st.sidebar.markdown("### 🤖 LLM Settings")
llm_type = st.sidebar.radio(
    "Select LLM Type",
    options=["openai", "ollama"],
    index=0,
    help="Choose between OpenAI (cloud) or Ollama (local) models"
)

# Model name selection based on type
if llm_type == "openai":
    llm_model_name = st.sidebar.selectbox(
        "Model Name",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )

    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key"
    )

    if not api_key:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
else:
    llm_model_name = st.sidebar.text_input(
        "Model Name",
        value="llama3.2",
        help="Enter the Ollama model name (e.g., llama3.2, gemma3:270m)"
    )
    st.sidebar.info("ℹ️ Make sure Ollama is running locally")

st.sidebar.markdown("---")

# Embedding settings
st.sidebar.markdown("### 🔤 Embedding Settings")
embedding_type = st.sidebar.selectbox(
    "Embedding Model",
    options=["openai", "huggingface", "nomic"],
    index=0,
    help="Choose embedding model for vector store"
)

if embedding_type == "openai" and llm_type != "openai":
    st.sidebar.warning("⚠️ OpenAI embeddings require an API key")

st.sidebar.markdown("---")

# Summary type selection
st.sidebar.markdown("### 📝 Summary Settings")
summary_type = st.sidebar.radio(
    "Summary Type",
    options=["detailed", "concise"],
    index=0,
    help="Choose between detailed or concise summary"
)

st.sidebar.markdown("---")

# About section
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This app fetches YouTube transcripts and generates AI summaries. "
    "You can also ask questions about the video content!"
)

# Main content
st.markdown('<div class="main-header">🎥 YouTube Video Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by LangChain & AI</div>', unsafe_allow_html=True)
st.markdown("---")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📹 Summarize Video", "💬 Ask Questions", "📄 View Transcript"])

with tab1:
    st.markdown("### Enter YouTube Video URL")

    # URL input
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the URL of the YouTube video you want to summarize",
            label_visibility="collapsed"
        )

    with col2:
        summarize_button = st.button("🚀 Summarize", type="primary", use_container_width=True)

    # Example URLs
    with st.expander("📎 Try Example Videos"):
        st.markdown("""
        - [AI Explanation](https://www.youtube.com/watch?v=aircAruvnKk)
        - [Tech Tutorial](https://www.youtube.com/watch?v=kqtD5dpn9C8)
        """)

    # Process summarization
    if summarize_button:
        if not url:
            st.error("❌ Please enter a YouTube URL")
        elif llm_type == "openai" and not api_key:
            st.error("❌ Please enter your OpenAI API key in the sidebar")
        else:
            # Show progress
            with st.spinner("🔄 Processing video... This may take a moment..."):
                try:
                    # Initialize summarizer
                    summarizer = YouTubeTranscriptSummarizer(
                        llm_type=llm_type,
                        llm_model_name=llm_model_name,
                        embedding_type=embedding_type
                    )

                    # Process video
                    result = summarizer.process_video(url, summary_type=summary_type)

                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                        st.session_state.video_processed = False
                    else:
                        # Store in session state
                        st.session_state.summary_result = result
                        st.session_state.qa_chain = result['qa_chain']
                        st.session_state.video_processed = True
                        st.session_state.chat_history = []  # Reset chat history
                        st.success("✅ Video processed successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.video_processed = False

    # Display results
    if st.session_state.summary_result and not isinstance(st.session_state.summary_result, dict) or (
            isinstance(st.session_state.summary_result, dict) and "error" not in st.session_state.summary_result):
        result = st.session_state.summary_result

        st.markdown("---")

        # Video metadata in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📺 Video Info")
            st.markdown(f"**Title:** {result.get('title', 'N/A')}")
            st.markdown(f"**Author:** {result.get('author', 'N/A')}")

        with col2:
            st.markdown("### 🤖 Model Info")
            model_info = result.get('model_info', {})
            st.markdown(f"**LLM:** {model_info.get('llm_type', 'N/A')} ({model_info.get('llm_model', 'N/A')})")
            st.markdown(f"**Embeddings:** {model_info.get('embedding_type', 'N/A')}")

        with col3:
            st.markdown("### 📝 Summary Info")
            st.markdown(f"**Type:** {summary_type.title()}")
            st.markdown(f"**Video ID:** {result.get('video_id', 'N/A')}")

        st.markdown("---")

        # Display summary
        st.markdown("### 📋 Summary")
        summary_text = result.get('summary', '')

        st.markdown(
            f'<div style="background-color: #333333; padding: 1.5rem; border-radius: 10px; '
            f'border-left: 5px solid #FF0000; line-height: 1.6;">'
            f'{summary_text}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            st.link_button("🔗 Watch Video", result.get('url', ''))

        with col2:
            # Download summary
            download_content = (
                f"YouTube Video Summary\n"
                f"{'=' * 50}\n\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"Author: {result.get('author', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n\n"
                f"Model Used:\n"
                f"- LLM: {model_info.get('llm_type', 'N/A')} ({model_info.get('llm_model', 'N/A')})\n"
                f"- Embeddings: {model_info.get('embedding_type', 'N/A')}\n\n"
                f"Summary ({summary_type}):\n"
                f"{'-' * 50}\n"
                f"{summary_text}\n"
            )

            st.download_button(
                label="💾 Download Summary",
                data=download_content,
                file_name=f"summary_{result.get('video_id', 'video')}.txt",
                mime="text/plain"
            )

        with col3:
            if st.button("🔄 Process New Video"):
                st.session_state.summary_result = None
                st.session_state.qa_chain = None
                st.session_state.chat_history = []
                st.session_state.video_processed = False
                st.rerun()

with tab2:
    st.markdown("### 💬 Ask Questions About the Video")

    if not st.session_state.video_processed:
        st.info("👈 Please process a video first in the 'Summarize Video' tab")
    else:
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("#### Conversation History")
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(
                        f'<div class="chat-message user-message" style="background-color: #333333;">'
                        f'<strong>You:</strong> {msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message assistant-message" style="background-color: #333333;">'
                        f'<strong>Assistant:</strong> {msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            st.markdown("---")

        # Question input
        with st.form(key="qa_form", clear_on_submit=True):
            question = st.text_input(
                "Ask a question",
                placeholder="What is the main topic discussed in the video?",
                label_visibility="collapsed"
            )
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                ask_button = st.form_submit_button("🔍 Ask Question", type="primary", use_container_width=True)
            with col2:
                if st.form_submit_button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

        if ask_button and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"question": question})
                    answer = response.get('answer', 'No answer generated')

                    # Add to chat history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer
                    })

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Suggested questions
        if not st.session_state.chat_history:
            with st.expander("💡 Suggested Questions"):
                st.markdown("""
                - What is the main topic of this video?
                - Can you summarize the key points discussed?
                - What examples or demonstrations were shown?
                - What conclusions were drawn?
                - Are there any action items or recommendations?
                """)

with tab3:
    st.markdown("### 📄 Full Transcript")

    if not st.session_state.video_processed:
        st.info("👈 Please process a video first in the 'Summarize Video' tab")
    else:
        result = st.session_state.summary_result
        transcript = result.get('full_transcript', '')

        if transcript:
            # Transcript stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", len(transcript.split()))
            with col2:
                st.metric("Character Count", len(transcript))
            with col3:
                est_time = len(transcript.split()) / 150  # Average reading speed
                st.metric("Est. Reading Time", f"{est_time:.1f} min")

            st.markdown("---")

            # Display transcript
            st.text_area(
                "Transcript",
                value=transcript,
                height=400,
                label_visibility="collapsed"
            )

            # Download transcript
            st.download_button(
                label="💾 Download Transcript",
                data=transcript,
                file_name=f"transcript_{result.get('video_id', 'video')}.txt",
                mime="text/plain"
            )
        else:
            st.warning("No transcript available")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Built by 👾 Oluwatosin Adesanya for Project Week 3"
    "</div>",
    unsafe_allow_html=True
)