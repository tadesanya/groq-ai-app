import streamlit as st

from app import llm_factory

# page configuration (app meta data)
st.set_page_config(
    page_title="Streamlit App",
    page_icon="🤖",
    layout="centered",
)

# initialise session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Title and description
st.title("Streamlit App")
st.markdown("Chat with a powerful LLM from Either Groq or OpenAI")

# Implement sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # API key input
    api_key = st.sidebar.text_input("Enter your API Key (Optional)",
                                    type="password",
                                    help="Enter your Groq or OpenAI API key, depending on the model you wish to use")

    # model selection
    model = st.selectbox(
        "LLM Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ],
        help="Select the model to use",
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
        help="Set context and behaviour of the assistant"
    )

    # clear chat button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main App section
# display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container and add to chat history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display assistant response in chat message container and add to chat history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # get the right LLM API class based on selected model
                llm_client_params = {"model_name": model}
                if api_key:
                    llm_client_params["api_key"] = api_key

                llm_client = llm_factory(**llm_client_params)

                # post a message to the LLM API
                llm_chat_params = {
                    "user_message": st.session_state["messages"],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if system_prompt:
                    llm_chat_params["system_prompt"] = system_prompt

                response = llm_client.chat(**llm_chat_params)

                # print response
                st.markdown(response)

                # add response to chat history
                st.session_state["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
