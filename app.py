import streamlit as st
import replicate
import os
from transformers import AutoTokenizer

# image = open(r"C:\Users\Ishaan\Downloads\haze1k\Distributed_haze1k\train\target\885.png", "rb")
image = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"])
if image:
    st.image(image, caption="Satellite Image", use_column_width=True,)

    # Replicate Credentials
    with st.sidebar:
        st.title('Satellite VQA')
        os.environ['REPLICATE_API_TOKEN'] = replicate_api

        st.subheader("Models and parameters")
        model = "anthropic/claude-3.5-sonnet"
        
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.7, step=0.01, help="Randomness of generated output")
        if temperature >= 1:
            st.warning('Values exceeding 1 produces more creative and random output as well as increased likelihood of hallucination.')
        if temperature < 0.1:
            st.warning('Values approaching 0 produces deterministic output. Recommended starting value is 0.7')
        
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, help="Top p percentage of most likely tokens for output generation")

    # Store LLM-generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]

    st.sidebar.button('Clear chat history', on_click=clear_chat_history)

    @st.cache_resource(show_spinner=False)
    def get_tokenizer():
        """Get a tokenizer to make sure we're not sending too much text
        text to the Model. Eventually we will replace this with ArcticTokenizer
        """
        return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

    def get_num_tokens(prompt):
        """Get the number of tokens in a given prompt"""
        tokenizer = get_tokenizer()
        tokens = tokenizer.tokenize(prompt)
        return len(tokens)

    # Function for generating model response
    def generate_response():
        prompt = []
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
            else:
                prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")
        
        prompt.append("<|im_start|>assistant")
        prompt.append("")
        prompt_str = "\n".join(prompt)
        
        if get_num_tokens(prompt_str) >= 3072:
            st.error("Conversation length too long. Please keep it under 3072 tokens.")
            st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
            st.stop()

        for event in replicate.stream(model,
                            input={"image": image,
                                    "prompt": prompt_str,
                                    "prompt_template": r"{prompt}",
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    }):
            yield str(event)

    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = generate_response()
            full_response = st.write_stream(response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)