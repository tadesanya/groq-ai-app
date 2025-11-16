import streamlit as st
from agentic_blog import GroqLLM, GraphBuilder, BlogState
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="AI Blog Generator",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .blog-content {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_blog' not in st.session_state:
    st.session_state.generated_blog = None
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False

# Header
st.markdown('<p class="main-header">✍️ AI Blog Generator</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key input (optional, if not in .env)
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key if not set in .env file")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    else:
        st.success("✅ API Key loaded from environment")

    st.divider()

    # About section
    st.header("ℹ️ About")
    st.markdown("""
    This AI Blog Generator uses:
    - **LangGraph** for workflow orchestration
    - **Groq** for fast LLM inference
    - **LangChain** for AI integration

    Generate blogs in English, Spanish, or French!
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input")

    # Topic input
    topic = st.text_input(
        "Blog Topic",
        placeholder="e.g., Artificial Intelligence in Healthcare",
        help="Enter the topic you want to write about"
    )

    # Language selection
    language_options = {
        "English (No Translation)": "english",
        "Spanish": "spanish",
        "French": "french"
    }

    selected_language = st.selectbox(
        "Target Language",
        options=list(language_options.keys()),
        help="Choose the language for your blog post"
    )

    # Generate button
    generate_button = st.button("🚀 Generate Blog", type="primary")

with col2:
    st.header("📊 Output")

    # Placeholder for output
    output_placeholder = st.empty()

# Generation logic
if generate_button:
    if not topic:
        st.error("⚠️ Please enter a blog topic!")
    elif not api_key and not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ Please provide a Groq API key!")
    else:
        st.session_state.is_generating = True

        with st.spinner("🤖 AI is crafting your blog post... This may take a moment."):
            try:
                # Initialize LLM
                llm = GroqLLM().get_llm()

                # Initialize graph builder
                graph_builder = GraphBuilder(llm)

                # Determine use case
                lang_code = language_options[selected_language]

                if lang_code == "english":
                    # Use topic-only graph
                    graph = graph_builder.setup_graph("topic")
                    initial_state = {
                        "topic": topic,
                        "blog": {"title": "", "content": ""},
                        "current_language": "english"
                    }
                else:
                    # Use language translation graph
                    graph = graph_builder.setup_graph("language")
                    initial_state = {
                        "topic": topic,
                        "blog": {"title": "", "content": ""},
                        "current_language": lang_code
                    }

                # Invoke the graph
                result = graph.invoke(initial_state)

                st.session_state.generated_blog = result
                st.session_state.is_generating = False

                st.success("✅ Blog generated successfully!")

            except Exception as e:
                st.error(f"❌ Error generating blog: {str(e)}")
                st.session_state.is_generating = False

# Display generated blog
if st.session_state.generated_blog:
    result = st.session_state.generated_blog

    with col2:
        with output_placeholder.container():
            st.markdown("### 📰 Generated Blog")

            # Display title
            if result.get('blog', {}).get('title'):
                st.markdown(f"## {result['blog']['title']}")

            st.divider()

            # Display content
            if result.get('blog', {}).get('content'):
                content = result['blog']['content']

                # Handle if content is a Blog object
                if hasattr(content, 'content'):
                    st.markdown(content.content)
                else:
                    st.markdown(content)

            st.divider()


