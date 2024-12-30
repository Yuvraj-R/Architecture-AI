import streamlit as st
import os
from langchain_community.document_loaders import GithubFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

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
    """
    loader = GithubFileLoader(
        repo=f"{repo_owner}/{repo_name}",
        branch=branch,
        github_api_url="https://api.github.com",
        file_filter=should_load_file,
    )
    return loader.load()


def split_documents(docs):
    """
    Split documents into smaller chunks based on file type.
    """
    split_documents = []
    skipped_files = 0

    for doc in docs:
        file_extension = doc.metadata.get("source", "").split(".")[-1].lower()
        file_extension = f".{file_extension}"
        text_splitter = get_text_splitter(file_extension)

        if text_splitter:
            split_docs = text_splitter.split_documents([doc])
            split_documents.extend(split_docs)
        else:
            skipped_files += 1

    return split_documents, skipped_files


def generate_embeddings(split_documents):
    """
    Generate vector embeddings for the split documents.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key is missing. Please set it in your environment."
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=split_documents)
    return vector_store


# Streamlit Interface
st.title("Codebase RAG Assistant: Generate Embeddings")

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

                # Load and process documents
                docs = load_github_docs(repo_owner, repo_name, branch)
                st.success(f"Successfully loaded {len(docs)} documents.")

                split_docs, skipped_files = split_documents(docs)
                st.success(f"Code successfully split into {len(split_docs)} chunks.")
                st.info(f"Skipped {skipped_files} unsupported files.")

                # Generate embeddings
                vector_store = generate_embeddings(split_docs)
                st.success("Vector embeddings generated successfully.")

                # Display the first embedding
                if split_docs:
                    first_chunk = split_docs[0].page_content
                    first_embedding = vector_store.embeddings.embed_query(first_chunk)
                    st.write(f"**First Chunk:**\n{first_chunk}")
                    st.write(
                        f"**First Embedding:**\n{first_embedding[:10]} (showing first 10 dimensions)"
                    )

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid GitHub repository URL.")
