# Math Agent AI - Intelligent Math Problem Solver

An advanced AI-powered system designed to solve, explain, and verify mathematical problems using a multi-agent architecture, retrieval-augmented generation (RAG), and human-in-the-loop workflows. Built with Streamlit for an interactive web interface.

## Overview

Math Agent AI is an end-to-end solution that combines multiple specialized AI agents to handle complex mathematical problem-solving tasks. The system supports multimodal inputs (text, image, audio), maintains memory of solved problems, and provides verification and explanation of solutions.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for parsing, solving, verifying, explaining, and clarifying problems
- **Multimodal Input Support**: Accept problems via text, image (OCR), and audio (speech-to-text)
- **Retrieval-Augmented Generation (RAG)**: Leverage a knowledge base for context-aware problem solving
- **Semantic Memory**: Cache and retrieve previously solved similar problems using embeddings
- **Interactive Web UI**: Built with Streamlit for responsive user interaction
- **Human-in-the-Loop (HITL)**: Automatically trigger clarification when confidence is low
- **Conversational Reference Resolution**: Maintain context across multiple problem-solving sessions
- **Feedback and Logging**: Track user feedback and system performance with comprehensive logging

## Architecture

### Core Components

1. **app.py** - Streamlit web application
   - User interface and session management
   - Input handling (text, image, audio)
   - Message display and feedback collection
   - Agent orchestration

2. **agents.py** - Multi-agent system
   - Parser Agent: Extracts problem structure into JSON format
   - Clarification Agent: Generates clarifying questions for ambiguous problems
   - Solver Agent: Solves problems using context and memory
   - Verifier Agent: Validates solution correctness and assigns confidence scores
   - Explainer Agent: Formats solutions into clear, student-friendly explanations
   - Reference Agent: Resolves conversational references to past problems

3. **llm.py** - Language Model Interface
   - Mistral API integration
   - Embedding generation for semantic similarity
   - Retry logic for robust API calls

4. **rag.py** - Retrieval-Augmented Generation
   - FAISS index for semantic search
   - Sentence-Transformer embeddings (all-MiniLM-L6-v2)
   - Knowledge base document storage and retrieval
   - Index building and persistence

5. **memory.py** - Semantic Memory System
   - Problem caching using embeddings
   - Cosine similarity-based problem matching
   - Memory persistence in JSON format

6. **multimodal.py** - Multimodal Input Processing
   - OCR text extraction from images (EasyOCR)
   - Audio transcription (OpenAI Whisper tiny model)
   - Math term normalization
   - Confidence estimation

7. **hitl.py** - Human-in-the-Loop
   - Triggers human intervention when confidence is low
   - Confidence threshold: 0.5

8. **logger.py** - Unified Logging
   - Centralized logging configuration
   - Console output with timestamps

## System Workflow

1. **Input Reception**: User provides problem via text, image, or audio
2. **Parsing**: Parser Agent structures the problem
3. **Clarification Check**: If problem is ambiguous, Clarification Agent generates questions
4. **Context Retrieval**: RAG system retrieves relevant knowledge base documents
5. **Memory Lookup**: System checks for semantically similar cached problems
6. **Solving**: Solver Agent generates solution using context and memory
7. **Verification**: Verifier Agent validates correctness and assigns confidence
8. **Explanation**: Explainer Agent formats solution for student understanding
9. **Feedback**: User provides feedback (like/dislike) which triggers resolution if needed
10. **Memory Storage**: Problem and solution cached for future reference

## Technical Stack

### Dependencies
- **Web Framework**: Streamlit
- **LLM**: Mistral API (mistral-large-latest)
- **Vector Database**: FAISS with Sentence-Transformers
- **Text Processing**: EasyOCR, OpenAI Whisper
- **Audio Processing**: soundfile, librosa
- **Data**: numpy, faiss-cpu
- **Environment**: python-dotenv

### Knowledge Base
The system includes mathematical knowledge base documents:
- algebra.md
- calculus.md
- linear_algebra.md
- probability.md

## Installation

1. Clone the repository and navigate to the project directory
2. Create a virtual environment:
   ```
   python -m venv my_venv
   my_venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a `.env` file in the project root:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

The application will launch at `http://localhost:8501`

### Input Methods
- **Text**: Type your math problem directly
- **Image**: Click the image button to upload a photo of the problem
- **Audio**: Click the audio button to speak your problem

### Features in UI
- View confidence levels for solutions
- Inspect RAG documents used for context
- Review agent traces for transparency
- Like/Dislike feedback for continuous improvement
- Automatic clarification prompts when needed

## Data Management

### Memory System
- Problems and solutions cached in `data/memory.json`
- Semantic similarity threshold: 0.85 (configurable)
- Memory loaded on startup for faster retrieval

### RAG System
- FAISS index stored in `data/faiss.index`
- Document metadata in `data/docs.pkl`
- Knowledge base read from `knowledge_base/` directory
- Index automatically built on first run

### Vosk Models
- Speech recognition model: `vosk-model-small-en-us-0.15`
- Located in `models/vosk/`

## Configuration

### Solver Parameters
- LLM Model: mistral-large-latest
- Temperature: 0.3 (for deterministic solutions)
- Retry Attempts: 3

### Confidence Thresholds
- HITL Trigger: confidence < 0.5
- Problem Similarity: 0.85
- ASR Trigger: confidence < 0.2

### RAG Configuration
- Embedding Model: all-MiniLM-L6-v2
- Top-K Retrieved Docs: 3
- Index Type: FAISS Flat L2

## Logging

The system maintains comprehensive logs with timestamps. Log levels include:
- INFO: General information about operations
- WARNING: Potential issues (e.g., low confidence)
- ERROR: Operation failures

Logs output to stdout in the format: `TIMESTAMP | LEVEL | MESSAGE`

## Error Handling

- **Robust JSON Extraction**: Fallback parsing with brace matching for malformed JSON
- **API Retry Logic**: Exponential backoff for Mistral API calls
- **Multimodal Fallbacks**: Graceful degradation if OCR or ASR fails
- **Memory Recovery**: Automatic data recovery if memory files are corrupted

## Development Notes

### Guardrails
- Parser avoids over-flagging standard probability problems
- Solver sticks to provided context to prevent hallucination
- Explainer avoids mention of internal mechanics
- Verifier applies standard mathematical conventions

### Extensibility
- Easy to add new agents by following the established pattern
- Knowledge base can be expanded by adding markdown files
- RAG system supports custom embedders and indices

## Future Enhancements

- Support for more mathematical domains (Statistics, Discrete Math, etc.)
- Integration with popular exam preparation databases
- Multi-language support
- Collaborative problem-solving features
- Advanced analytics on problem-solving patterns
- Better UI Design
