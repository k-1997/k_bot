import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import tempfile
import os

st.title("Document Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx"], accept_multiple_files=True)

question = st.chat_input("Ask a question about the documents:")

# Access the OpenAI API key from Streamlit's secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key


if st.button("Get Answer"):
    if uploaded_files is not None and len(uploaded_files) > 0:
        all_text = []
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(tmp_file_path)

            if loader:
                all_text.extend(loader.load())

            os.remove(tmp_file_path) # Clean up the temporary file

        if all_text:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            document_chunks = text_splitter.split_documents(all_text)
            st.write("Documents processed and split into chunks.")

            embeddings = OpenAIEmbeddings()
            knowledge_base = Chroma.from_documents(document_chunks, embeddings)
            st.write("Knowledge base created successfully.")

            llm = ChatOpenAI(model_name="gpt-3.5-turbo")
            retriever = knowledge_base.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            # Store the qa_chain in the session state for later use
            st.session_state['qa_chain'] = qa_chain

            st.write("Question answering chain set up.")


# Check if a question is entered and qa_chain exists
if question and 'qa_chain' in st.session_state:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    qa_chain = st.session_state['qa_chain']
    response = qa_chain.run(question)

    # Display bot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

elif question and 'qa_chain' not in st.session_state:
    st.write("Please upload and process documents first to create the knowledge base.")
elif not question and ('qa_chain' in st.session_state or (uploaded_files is not None and len(uploaded_files) > 0)):
     st.write("Please enter a question.")
