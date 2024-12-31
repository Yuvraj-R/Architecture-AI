import os
import shutil
from git import Repo
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize Pinecone Connection
index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index("codebase-rag")

IGNORED_DIRS = {
    "node_modules",
    "venv",
    "env",
    "dist",
    "build",
    ".git",
    "__pycache__",
    ".next",
    ".vscode",
    "vendor",
}

EXTENSION_TO_LANGUAGE = {
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".kotlin": Language.KOTLIN,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".php": Language.PHP,
    ".proto": Language.PROTO,
    ".py": Language.PYTHON,
    ".rst": Language.RST,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".md": Language.MARKDOWN,
    ".tex": Language.LATEX,
    ".html": Language.HTML,
    ".cs": Language.CSHARP,
    ".cob": Language.COBOL,
    ".c": Language.C,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".hs": Language.HASKELL,
}


def should_load_file(file_path):
    """
    Determine whether a file should be loaded based on ignored directories.
    """
    for ignored_dir in IGNORED_DIRS:
        if f"/{ignored_dir}/" in file_path or file_path.startswith(f"{ignored_dir}/"):
            return False
    return True


def get_text_splitter(file_extension):
    """
    Returns the appropriate text splitter based on the file extension.
    """
    language = EXTENSION_TO_LANGUAGE.get(file_extension)
    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=1000, chunk_overlap=100
        )
    return None


def load_github_docs(repo_owner, repo_name, branch):
    """
    Load documents from the specified GitHub repository.
    (Refactored to clone locally and read files, rather than using GithubFileLoader.)
    """
    # Construct a local path to clone the repo
    local_repo_path = os.path.join("repos", f"{repo_owner}_{repo_name}")

    # If it already exists, remove it
    if os.path.exists(local_repo_path):
        shutil.rmtree(local_repo_path)

    Repo.clone_from(
        f"https://github.com/{repo_owner}/{repo_name}.git",
        local_repo_path,
        branch=branch,
    )

    docs = []
    for root, dirs, files in os.walk(local_repo_path):
        # Filter out ignored directories directly
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, local_repo_path)

            # Check if we should load this file
            if not should_load_file(relative_path):
                continue

            # Determine the file extension to decide text splitting
            file_extension = os.path.splitext(file_name)[1].lower()

            # Attempt to read the file as UTF-8. If it fails, skip it
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                doc = Document(page_content=content, metadata={"source": relative_path})
                docs.append(doc)

            except UnicodeDecodeError:
                # If the file can't be decoded as UTF-8, skip it
                continue
            except Exception as e:
                print(f"Skipping {relative_path} due to error: {e}")
                continue

    shutil.rmtree(local_repo_path)
    return docs


def split_documents(docs):
    """
    Split documents into smaller chunks based on file type.
    """
    split_documents_list = []
    skipped_files = 0

    for doc in docs:
        file_extension = doc.metadata.get("source", "").split(".")[-1].lower()
        file_extension = f".{file_extension}"
        text_splitter = get_text_splitter(file_extension)

        if text_splitter:
            split_docs = text_splitter.split_documents([doc])
            split_documents_list.extend(split_docs)
        else:
            skipped_files += 1

    return split_documents_list, skipped_files


def store_embeddings_in_pinecone(split_documents, namespace):
    """
    Generate vector embeddings for the split documents and store them in Pinecone.
    """
    # Check if the given namespace already exists, and clear it if so
    stats = index.describe_index_stats()
    if stats and "namespaces" in stats and namespace in stats["namespaces"]:
        index.delete(deleteAll=True, namespace=namespace)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(
        documents=split_documents,
        embedding=embeddings,
        index_name="codebase-rag",
        namespace=namespace,
    )


def perform_rag(query, namespace):
    """
    Query the Pinecone database and fetch results based on the query.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embeddings.embed_query(query)

    results = index.query(
        vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace
    )

    # Build a context string from top 5 matches
    contexts = []
    for match in results["matches"]:
        snippet = match["metadata"].get("text", "")
        contexts.append(snippet)

    augmented_query = (
        "<CONTEXT>\n"
        + "\n\n-------\n\n".join(contexts)
        + "\n-------\n</CONTEXT>\n\n\nMY QUESTION:\n"
        + query
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    messages = [
        SystemMessage(
            "You are a Senior Software Engineer. Answer based on the provided context. However, don't mention said context as the end user doesn't know about it."
        ),
        HumanMessage(augmented_query),
    ]

    response = llm.invoke(messages)

    return response.content


# Streamlit Interface
st.set_page_config(page_title="Codebase RAG Assistant", layout="wide")
st.title("Codebase RAG Assistant")

col1, col2 = st.columns([3, 1])
with col1:
    github_url = st.text_input("GitHub Repo URL")
with col2:
    branch = st.text_input("Branch", placeholder="main")

if st.button("Submit"):
    if github_url:
        try:
            # Parse GitHub URL
            url_parts = github_url.rstrip("/").split("/")
            if len(url_parts) < 5:
                st.error(
                    "Invalid GitHub URL. Please ensure it follows the format: https://github.com/owner/repo"
                )
            else:
                repo_owner = url_parts[3]
                repo_name = url_parts[4]
                branch = branch.strip() if branch.strip() else "main"

                # Load documents (now clones locally and reads files)
                docs = load_github_docs(repo_owner, repo_name, branch)
                st.success(f"Successfully loaded {len(docs)} documents.")

                # Split documents
                split_docs, skipped_files = split_documents(docs)
                st.success(f"Code successfully split into {len(split_docs)} chunks.")
                st.info(f"Skipped {skipped_files} unsupported files.")

                # Store embeddings in Pinecone
                namespace = f"{repo_owner}/{repo_name}"
                store_embeddings_in_pinecone(split_docs, namespace)
                st.success("Vector embeddings stored in Pinecone successfully.")

                # Store namespace for querying
                st.session_state.repo_path = f"{repo_owner}/{repo_name}"

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid GitHub repository URL.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the codebase..."):
    if "repo_path" not in st.session_state:
        st.error("Please process a repository first!")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Querying Pinecone database..."):
            try:
                answer = perform_rag(prompt, st.session_state.repo_path)

                st.chat_message("assistant").markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
