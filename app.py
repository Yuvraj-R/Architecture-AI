import streamlit as st
from langchain_community.document_loaders import GithubFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from dotenv import load_dotenv

load_dotenv()

# Directories to ignore during loading
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


# General-purpose text splitter
general_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def should_load_file(file_path):
    for ignored_dir in IGNORED_DIRS:
        if f"/{ignored_dir}/" in file_path or file_path.startswith(f"{ignored_dir}/"):
            return False
    return True


def get_text_splitter(file_extension):
    language = EXTENSION_TO_LANGUAGE.get(file_extension)
    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=1000, chunk_overlap=100
        )
    return None


st.title("Codebase RAG Assistant")

col1, col2 = st.columns([3, 1])
with col1:
    github_url = st.text_input("GitHub Repo URL")
with col2:
    branch = st.text_input("Branch", placeholder="main")

if st.button("Submit"):
    if github_url:
        try:
            # Validate GitHub URL
            url_parts = github_url.rstrip("/").split("/")
            if len(url_parts) < 5:
                st.error(
                    "Invalid GitHub URL. Please ensure it follows the format: https://github.com/owner/repo"
                )
            else:
                repo_owner = url_parts[3]
                repo_name = url_parts[4]
                branch = branch.strip() if branch.strip() else "main"

                # Load files from GitHub repository
                loader = GithubFileLoader(
                    repo=f"{repo_owner}/{repo_name}",
                    branch=branch,
                    github_api_url="https://api.github.com",
                    file_filter=should_load_file,
                )

                docs = loader.load()
                st.success(f"Successfully loaded {len(docs)} documents.")

                split_documents = []
                skipped_files = 0
                for doc in docs:
                    file_extension = (
                        doc.metadata.get("source", "").split(".")[-1].lower()
                    )
                    file_extension = f".{file_extension}"

                    # Get the relevant splitter
                    text_splitter = get_text_splitter(file_extension)
                    if text_splitter:
                        split_docs = text_splitter.split_documents([doc])
                        split_documents.extend(split_docs)
                    else:
                        skipped_files += 1  # Count skipped files

                st.success(
                    f"Code successfully split into {len(split_documents)} chunks."
                )
                st.info(f"Skipped {skipped_files} unsupported files.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid GitHub repository URL.")
