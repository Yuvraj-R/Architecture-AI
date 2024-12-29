import streamlit as st
from langchain_community.document_loaders import GithubFileLoader
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


def should_load_file(file_path):
    for ignored_dir in IGNORED_DIRS:
        if f"/{ignored_dir}/" in file_path or file_path.startswith(f"{ignored_dir}/"):
            return False
    return True


st.title("Codebase RAG Assistant")

col1, col2 = st.columns([3, 1])
with col1:
    github_url = st.text_input("GitHub Repo URL")
with col2:
    branch = st.text_input("Branch", placeholder="main")

if st.button("Submit"):
    if github_url:
        try:
            url_parts = github_url.rstrip("/").split("/")
            if len(url_parts) < 5:
                st.error(
                    "Invalid GitHub URL. Please ensure it follows the format: https://github.com/owner/repo"
                )
            else:
                repo_owner = url_parts[3]
                repo_name = url_parts[4]
                branch = branch.strip() if branch.strip() else "main"

                loader = GithubFileLoader(
                    repo=f"{repo_owner}/{repo_name}",
                    branch=branch,
                    github_api_url="https://api.github.com",
                    file_filter=should_load_file,
                )

                docs = loader.load()

                st.success(f"Successfully loaded {len(docs)} documents.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid GitHub repository URL.")
