# AI Essay Assistant — Agentic AI TA2

An AI-powered essay writing assistant developed as an extension of our TA1 project. It uses LangGraph and a locally hosted Ollama LLM to generate, critique, and revise student-level paragraphs.

## Project Overview

The application follows a three-stage workflow:

1. **Draft:** Generates an approximately 80-word paragraph based on the given topic.
2. **Critique:** Evaluates the draft for grammar, clarity, relevance, and organisation.
3. **Revise:** Improves the draft when issues are identified.

A conditional edge determines whether the workflow proceeds to revision or ends with the original draft.

## Technologies Used

- Python
- LangGraph
- LangChain
- Ollama — Qwen 2.5:3B
- ChromaDB
- nomic-embed-text
- Retrieval-Augmented Generation (RAG)

## TA2 Extensions

- Retrieval of saved writing preferences using ChromaDB.
- Semantic search using nomic-embed-text embeddings.
- Integration of retrieved preferences into the draft-generation prompt.
- Separate TA1 and TA2 source-code folders.

## Project Structure

```text
AI-Essay-Assistant-TA2/
├── essay_assistant_TA1_working.py
└── ta2/
    ├── essay_assistant.py
    ├── rag_memory.py
    └── rag_test.py
```

## Setup and Execution

1. Install Python and Ollama.
2. Download the required Ollama models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

3. Install the required Python packages:

```bash
pip install langgraph langchain-ollama chromadb requests
```

4. Start Ollama and navigate to the TA2 folder:

```bash
cd ta2
```

5. Run the memory test:

```bash
python3 rag_test.py
```

6. Run the application:

```bash
python3 essay_assistant.py
```

## Academic Project

**Course:** Agentic AI — PE-IV, Semester VII  
**Assessment:** TA2 Project Extension  
**Project:** AI Essay Assistant

This repository contains the source code for the TA1 and TA2 project implementations.
