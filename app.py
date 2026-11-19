import streamlit as st

from rag_backend import (
    answer_question,
    build_collection,
    get_embedder,
    get_llm,
)


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="PDF",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #252B3B; }
    [data-testid="stChatMessage"] {
        background-color: #171C28;
        border: 1px solid #252B3B;
        border-radius: 12px;
        margin-bottom: 8px;
    }
    .upload-card {
        background: #171C28;
        border: 1px solid #252B3B;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "active_file" not in st.session_state:
    st.session_state.active_file = None


@st.cache_resource
def load_embedder():
    return get_embedder()


@st.cache_resource
def load_llm():
    return get_llm()


with st.sidebar:
    st.title("PDF RAG")
    st.caption("Upload a PDF, then ask questions about it.")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        file_signature = (uploaded_file.name, uploaded_file.size)
        if st.session_state.active_file != file_signature:
            try:
                with st.spinner("Reading and indexing your PDF..."):
                    collection, chunks = build_collection(
                        uploaded_file.name,
                        uploaded_file,
                        load_embedder(),
                    )
                st.session_state.collection = collection
                st.session_state.active_file = file_signature
                st.session_state.messages = []
                st.success(f"Indexed {len(chunks)} chunks")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.collection is not None:
        st.markdown(
            f'<div class="upload-card"><strong>Active document</strong><br>{st.session_state.active_file[0]}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

st.title("Chat with your PDF")

if st.session_state.collection is None:
    st.info("Upload a PDF from the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("evidence"):
                with st.expander("View retrieved evidence"):
                    for index, item in enumerate(message["evidence"], start=1):
                        st.markdown(f"**Source {index}**")
                        st.write(item)

    question = st.chat_input("Ask a question about the PDF")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching the document..."):
                    answer, evidence = answer_question(
                        st.session_state.collection,
                        load_embedder(),
                        load_llm(),
                        question,
                    )
                st.markdown(answer)
                with st.expander("View retrieved evidence"):
                    for index, item in enumerate(evidence, start=1):
                        st.markdown(f"**Source {index}**")
                        st.write(item)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "evidence": evidence}
            )
        except Exception as exc:
            st.error(f"Unable to answer: {exc}")
