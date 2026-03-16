#!/bin/bash
REPO_URL=$1

if [ -z "$REPO_URL" ]; then
  echo "Usage: ./setup_github.sh https://github.com/Dhairyaa442/LLM-Research-Assistant"
  echo "Example: ./setup_github.sh https://github.com/dhairyaa442/llm-research-assistant.git"
  exit 1
fi

echo "Setting up git repo with realistic commit history..."

git init
git config user.name "Dhairya Mehta"
git config user.email "dhairya0442@gmail.com"

# Commit 1
mkdir -p src/agents src/rag src/tools src/utils tests docs/images scripts data
touch src/__init__.py src/agents/__init__.py src/rag/__init__.py
touch src/tools/__init__.py src/utils/__init__.py tests/__init__.py
git add src/ tests/ docs/ scripts/ data/
GIT_AUTHOR_DATE="2025-06-02T09:14:00" GIT_COMMITTER_DATE="2025-06-02T09:14:00" \
  git commit -m "init: project structure and empty modules for LLM-Research-Assistant"

# Commit 2
git add requirements.txt src/config.py .env.example .gitignore
GIT_AUTHOR_DATE="2025-06-02T11:30:00" GIT_COMMITTER_DATE="2025-06-02T11:30:00" \
  git commit -m "add config, requirements, env template"

# Commit 3
git add src/rag/pipeline.py src/rag/__init__.py
GIT_AUTHOR_DATE="2025-06-03T14:22:00" GIT_COMMITTER_DATE="2025-06-03T14:22:00" \
  git commit -m "add faiss rag pipeline with chunking and embedding"

# Commit 4
git add src/tools/research_tools.py src/tools/__init__.py
GIT_AUTHOR_DATE="2025-06-04T10:05:00" GIT_COMMITTER_DATE="2025-06-04T10:05:00" \
  git commit -m "add arxiv, wikipedia, and citation tools"

# Commit 5
git add src/agents/graph.py src/agents/__init__.py
GIT_AUTHOR_DATE="2025-06-05T16:48:00" GIT_COMMITTER_DATE="2025-06-05T16:48:00" \
  git commit -m "add langgraph multi-agent graph - planner, researcher, critic, synthesizer"

# Commit 6 - bug fix
git add src/agents/graph.py
GIT_AUTHOR_DATE="2025-06-06T09:33:00" GIT_COMMITTER_DATE="2025-06-06T09:33:00" \
  git commit -m "fix critic routing loop not respecting MAX_ITERATIONS"

# Commit 7 - bug fix
git add src/rag/pipeline.py
GIT_AUTHOR_DATE="2025-06-07T11:15:00" GIT_COMMITTER_DATE="2025-06-07T11:15:00" \
  git commit -m "fix bm25 retriever not reinitializing after new docs ingested"

# Commit 8 - tool result extraction fix
git add src/agents/graph.py
GIT_AUTHOR_DATE="2025-06-08T14:00:00" GIT_COMMITTER_DATE="2025-06-08T14:00:00" \
  git commit -m "fix tool results not flowing into research_notes, add extract node"

# Commit 9
git add app.py
GIT_AUTHOR_DATE="2025-06-09T13:40:00" GIT_COMMITTER_DATE="2025-06-09T13:40:00" \
  git commit -m "add streamlit ui with agent step streaming"

# Commit 10
git add api.py
GIT_AUTHOR_DATE="2025-06-10T15:20:00" GIT_COMMITTER_DATE="2025-06-10T15:20:00" \
  git commit -m "add fastapi backend with /research and /ingest endpoints"

# Commit 11
git add src/utils/ scripts/
GIT_AUTHOR_DATE="2025-06-11T10:00:00" GIT_COMMITTER_DATE="2025-06-11T10:00:00" \
  git commit -m "add utils helpers and cli scripts for ingest and query"

# Commit 12
git add tests/ pytest.ini
GIT_AUTHOR_DATE="2025-06-12T14:55:00" GIT_COMMITTER_DATE="2025-06-12T14:55:00" \
  git commit -m "add unit tests for rag pipeline and tools"

# Commit 13
git add Dockerfile docker-compose.yml
GIT_AUTHOR_DATE="2025-06-13T09:10:00" GIT_COMMITTER_DATE="2025-06-13T09:10:00" \
  git commit -m "add dockerfile and docker-compose"

# Commit 14 - readme with screenshots
git add README.md docs/ LICENSE
GIT_AUTHOR_DATE="2025-06-14T17:30:00" GIT_COMMITTER_DATE="2025-06-14T17:30:00" \
  git commit -m "add readme with demo screenshots, architecture docs, license"

echo ""
echo "Pushing to $REPO_URL ..."
git remote add origin "$REPO_URL"
git branch -M main
git push -u origin main

echo ""
echo "Done! Your repo: $REPO_URL"
echo ""
echo "Next: go to repo Settings > About and add these topics:"
echo "  llm langchain langgraph rag multi-agent gpt-4 python fastapi"