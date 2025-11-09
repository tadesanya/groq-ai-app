import streamlit as st
import os
from dotenv import load_dotenv
from news_summarizer import NewsArticleSummarizer

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="News Article Summarizer",
    page_icon="📰",
    layout="wide"
)

# Initialize session state
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = None

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Model type selection
model_type = st.sidebar.radio(
    "Select Model Type",
    options=["openai", "ollama"],
    index=0,
    help="Choose between OpenAI (cloud) or Ollama (local) models"
)

# Model name selection based on type
if model_type == "openai":
    st.sidebar.markdown("### OpenAI Settings")
    model_name = st.sidebar.selectbox(
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
    st.sidebar.markdown("### Ollama Settings")
    model_name = st.sidebar.text_input(
        "Model Name",
        value="gemma3:270m",
        help="Enter the Ollama model name (e.g., llama3.2, gemma3:270m)"
    )
    api_key = None
    st.sidebar.info("ℹ️ Make sure Ollama is running locally")

# Summary type selection
summary_type = st.sidebar.radio(
    "Summary Type",
    options=["detailed", "concise"],
    index=0,
    help="Choose between detailed or concise summary"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app uses LangChain to summarize news articles. "
    "Simply paste a URL and get an AI-generated summary!"
)

# Main content
st.title("📰 News Article Summarizer")
st.markdown("Powered by LangChain & AI")
st.markdown("---")

# URL input
url = st.text_input(
    "Enter Article URL",
    placeholder="https://example.com/news-article",
    help="Paste the URL of the news article you want to summarize"
)

# Summarize button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    summarize_button = st.button("🚀 Summarize Article", type="primary", use_container_width=True)

# Process summarization
if summarize_button:
    if not url:
        st.error("❌ Please enter a URL")
    elif model_type == "openai" and not api_key:
        st.error("❌ Please enter your OpenAI API key in the sidebar")
    else:
        # Show progress
        with st.spinner("🔄 Fetching and analyzing article..."):
            try:
                # Initialize summarizer
                summarizer = NewsArticleSummarizer(
                    api_key=api_key,
                    model_type=model_type,
                    model_name=model_name
                )

                # Get summary
                result = summarizer.summarize(url, summary_type=summary_type)

                # Store in session state
                st.session_state.summary_result = result

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.summary_result = None

# Display results
if st.session_state.summary_result:
    result = st.session_state.summary_result

    if "error" in result:
        st.error(f"❌ {result['error']}")
    else:
        st.success("✅ Summary generated successfully!")
        st.markdown("---")

        # Create two columns for metadata
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Article Details")
            st.markdown(f"**Title:** {result.get('title', 'N/A')}")

            authors = result.get('authors', [])
            if authors:
                st.markdown(f"**Authors:** {', '.join(authors)}")
            else:
                st.markdown("**Authors:** N/A")

            publish_date = result.get('publish_date', 'N/A')
            st.markdown(f"**Published:** {publish_date if publish_date else 'N/A'}")

        with col2:
            st.markdown("### 🤖 Model Information")
            model_info = result.get('model_info', {})
            st.markdown(f"**Model Type:** {model_info.get('type', 'N/A').title()}")
            st.markdown(f"**Model Name:** {model_info.get('name', 'N/A')}")
            st.markdown(f"**Summary Type:** {summary_type.title()}")

        st.markdown("---")

        # Display summary
        st.markdown("### 📝 Summary")

        # Extract summary text
        summary_text = result.get('summary', '')

        # Handle different summary formats
        if isinstance(summary_text, dict):
            summary_text = summary_text.get('output_text', str(summary_text))
        elif not isinstance(summary_text, str):
            summary_text = str(summary_text)

        # Display in an expandable container with streaming effect
        with st.container():
            st.markdown(
                f'<div style="background-color: #333333; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">'
                f'{summary_text}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Link to original article
        st.markdown(f"[📖 Read Original Article]({result.get('url', '')})")

        # Download summary option
        st.download_button(
            label="💾 Download Summary",
            data=f"Title: {result.get('title', 'N/A')}\n\n"
                 f"Authors: {', '.join(result.get('authors', [])) if result.get('authors') else 'N/A'}\n"
                 f"Published: {result.get('publish_date', 'N/A')}\n"
                 f"Model: {model_info.get('type', 'N/A')} - {model_info.get('name', 'N/A')}\n\n"
                 f"Summary:\n{summary_text}",
            file_name="article_summary.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built by 👾 Oluwatosin Adesanya for Project Week 3"
    "</div>",
    unsafe_allow_html=True
)
