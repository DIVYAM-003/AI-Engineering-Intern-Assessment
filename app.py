# app.py

import streamlit as st
from dotenv import load_dotenv
from utils import get_text_from_file, get_text_chunks, get_vector_store, get_conversational_chain

def handle_user_input(user_question):
    """
    Processes user input and updates the chat display.
    """
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"**You:** {message.content}")
            else:
                st.write(f"**Bot:** {message.content}")
    else:
        st.write("Please upload a document and click 'Process' first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Talk to Your Document", page_icon="📄")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main UI layout
    st.header("Talk to Your Document (Hugging Face) 📄")
    user_question = st.text_input("Ask a question about your document:")
    
    if user_question:
        handle_user_input(user_question)

    # Sidebar for file upload and processing
    with st.sidebar:
        st.subheader("Your Document")
        uploaded_file = st.file_uploader(
            "Upload your PDF or Image here and click 'Process'", 
            type=["pdf", "png", "jpg", "jpeg"]
        )

        if st.button("Process"):
            if uploaded_file is not None:
                with st.spinner("Processing your document..."):
                    # Step 1: Extract text
                    raw_text = get_text_from_file(uploaded_file)
                    if not raw_text:
                        st.error("Could not extract text. The file might be corrupted.")
                        return

                    # Step 2: Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("Could not create text chunks.")
                        return
                    
                    # Step 3: Create vector store
                    vector_store = get_vector_store(text_chunks)
                    if not vector_store:
                        st.error("Could not create vector store. Check your API token and network connection.")
                        return

                    # Step 4: Create conversation chain
                    st.session_state.conversation = get_conversational_chain()
                    if st.session_state.conversation:
                        st.success("Processing complete! You can now ask questions.")
                    else:
                        st.error("Failed to create the conversational chain.")
            else:
                st.warning("Please upload a file first.")

if __name__ == '__main__':
    main()
