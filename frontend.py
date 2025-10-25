import streamlit as st
from openai import api_key

from app import LLMApp


# page configuration (app meta data)
st.set_page_config(
    page_title="Streamlit App",
    page_icon="",
    layout="centered",
)


# initialise session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "llm_app" not in st.session_state:
    st.session_state["llm_app"] = None


# Title and description
st.title("Streamlit App")
st.markdown("Chat with a powerful LLM from Groq")

# Implement sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # API key input
    api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API Key")
    # if not api_key:
    #     api_key = LLMApp().api_key

    # model selection
    model = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ],
        help="Select the model to use"
    )

    temperature = st.slider(
        "Temperature",
        max_value=2.0,
        min_value=0.0,
        value=0.5,
        step=0.1,
    )

    max_tokens = st.slider(
        "Maximum number of tokens",
        max_value=2048,
        min_value=256,
        value=1024,
        step=256,
        help="Set the response length"
    )

    system_prompt = st.text_area(
        "System prompt (Optional)",
        placeholder="You are a helpful assistant...",
        help="Set context and behaviou of the assistant"
    )

    # clear chat button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        # if st.session_state.llm_app:
        #     st.session_state.llm_app.clear_history()

        st.rerun()

if st.session_state.llm_app is None:
    try:
        st.session_state.llm_app = LLMApp(api_key, model=model)
    except Exception as e:
        st.error(f"Error: {str(e)}")


# display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    if not api_key or not LLMApp().api_key:
        st.warning("API Key not set")
    else:
        st.session_state.messages.append(
            {
                "content": f"{prompt}",
                "role": "user",
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_app.chat(
                        user_message=prompt,
                        system_prompt=system_prompt if system_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    st.markdown(response)
                    st.session_state.messages.append(
                        {
                            "content": f"{response}",
                            "role": "assistant",
                        }
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
