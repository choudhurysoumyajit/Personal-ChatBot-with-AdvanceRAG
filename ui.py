import streamlit as st
from rag_utils import process_files, ask_question


st.set_page_config(page_title="Advance Rag", layout="wide")
st.title("📄 Advance Rag | Chat with Your Files ")

st.sidebar.header("Configuration")
uploaded_files = st.sidebar.file_uploader(
    "Upload your documents (PDF, CSV, TXT)", 
    type=["pdf", "csv", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    file_flag = True
else:
    file_flag = False

if uploaded_files:
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=500, value=100, step=10)
    top_k = st.sidebar.number_input("Documents to Retrieve per Query", min_value=1, max_value=10, value=3)

if uploaded_files:
    if st.sidebar.button("Submit & Process"):
        if uploaded_files:
            with st.spinner("🔄 Processing documents... Please wait"):
                process_files(uploaded_files, chunk_size, chunk_overlap)
                
            popup = st.empty()

            with popup.container():
                st.warning("✅ Documents processed and stored in local vector DB")
                if st.button("OK"):
                    popup.empty()
        else:
            st.sidebar.warning("⚠️ Please upload at least one document.")

initial_message = """
    Hi there! I'm your PersonalBot 🤖 
    You can shoot any question you want or talk to your data from any document
"""        

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
  

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
      
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching the details for you..."):
            if uploaded_files:
                response = ask_question(prompt,top_k,file_flag)
            else:
                response = ask_question(prompt,5,file_flag)
            placeholder = st.empty()
            full_response = response  # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
st.button('Clear Chat', on_click=clear_chat_history)
