# Upload Instructions

## Option 1: Replace current repo contents from local machine

```bash
git clone https://github.com/vinod-ai-engineering/two-stage-rag-system.git
cd two-stage-rag-system

# Copy the generated files into this folder, then:
git add .
git commit -m "Redesign RAG project into production-style repo"
git push origin main
```

## Option 2: Create a new branch first

```bash
git checkout -b repo-redesign
git add .
git commit -m "Redesign RAG project structure and documentation"
git push origin repo-redesign
```

Then open a Pull Request in GitHub and review changes before merging.

## Important

Do not commit `.env`, API keys, dataset files, or notebook checkpoints.
