import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from huggingface_hub import InferenceClient

def get_text_from_file(uploaded_file):
    text = ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".pdf":
        try:
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf_doc:
                text += page.get_text()
            pdf_doc.close()
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None
    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    else:
        print("Unsupported file format.")
        return None
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_hf")
        print("✅ Vector store created and saved.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


def get_conversational_chain():
    try:
        print("\n--- INITIALIZING CONVERSATIONAL CHAIN ---")
        hf_client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        llm = HuggingFaceHub(
            client=hf_client,
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.6, "max_new_tokens": 1024}
        )
        print("✅ LLM initialized.")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vector_store = FAISS.load_local("faiss_index_hf", embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded.")

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        print("✅ Memory initialized.")

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
        )
        print("✅ Conversational chain created.\n")

        return conversation_chain
    except Exception as e:
        print(f"❌ Failed to create conversational chain: {e}")
        return None
