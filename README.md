# GPT-2 Activation Patching Demo

An interactive web application that performs activation patching on GPT-2 to analyze how different layers in the transformer model contribute to grammatical number agreement (singular vs. plural verb forms).

🎥 Demo Video
[![Watch the Demo](https://img.youtube.com/vi/oahGAtX4pV8/maxresdefault.jpg)](https://youtu.be/oahGAtX4pV8)

## Overview

This project demonstrates mechanistic interpretability techniques by patching activations in GPT-2's attention layers. It uses **LangGraph** to orchestrate a multi-agent pipeline that analyzes results and generates human-readable explanations.

## Features

- **Interactive Web Interface**: Simple, clean UI for entering sentences
- **Real-time Analysis**: Processes sentences and performs layer-wise activation patching
- **LangGraph Agent Pipeline**: Multi-agent system with parser and explanation agents
- **AI-Powered Explanations**: Uses Groq API to generate human-readable explanations of results
- **Automated Verb Detection**: Automatically identifies verb pairs (has/have, is/are, was/were, does/do)
- **Layer Attribution**: Shows which transformer layers contribute most to correct grammatical predictions

## Technology Stack

- **Backend**: FastAPI (Python web framework)
- **ML Framework**: PyTorch + TransformerLens (for model manipulation)
- **Agent Framework**: LangGraph (for orchestrating multi-agent pipeline)
- **Model**: GPT-2 (via HuggingFace)
- **LLM API**: Groq (for generating explanations)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Deployment**: Designed for Hugging Face Spaces

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export PORT=7860  # Optional, defaults to 7860
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:7860
```

3. Enter a sentence containing a verb from these pairs:
   - has/have
   - is/are
   - was/were
   - does/do

4. Click "Run Patching" to see the analysis

## Example Sentences

- "The cat has fur."
- "The dogs have collars."
- "The student is studying."
- "The books are on the shelf."

## How It Works

1. **Sentence Processing**: The system detects the verb and its grammatical number
2. **Variant Generation**: Creates a grammatically incorrect version by swapping singular/plural
3. **Activation Patching**: For each layer, patches activations from the correct sentence into the incorrect one
4. **LangGraph Pipeline**: Executes a multi-agent workflow:
   - **Parser Agent**: Extracts structured data from raw patching results
   - **Explanation Agent**: Generates natural language explanations using Groq LLM
5. **Results Display**: Shows both technical results and human-readable explanations

## LangGraph Architecture

The project uses LangGraph to orchestrate a two-agent pipeline:

```
START → Parser Agent → Explanation Agent → END
         (extracts      (generates AI
          structured     explanation)
          data)
```

Each agent operates on a shared state object that flows through the graph, ensuring clean separation of concerns and making the pipeline easy to extend with additional agents.

## Project Structure

```
├── app.py                  # Server startup script
├── main.py                 # FastAPI application and routes
├── patching_logic.py       # Core activation patching implementation
├── explanation_utils.py    # LangGraph agent pipeline implementation
├── requirements.txt        # Python dependencies
└── static/
    └── index.html         # Frontend interface
```

## API Endpoints

### `GET /`
Serves the main web interface

### `POST /predict`
Performs activation patching analysis

**Request Body:**
```json
{
  "sentence": "The cat has fur."
}
```

**Response:**
```json
{
  "patching_results": {
    "user_sentence": "...",
    "verb_pair": ["singular", "plural"],
    "layer_probs_correct_after_patch": [...],
    ...
  },
  "explanation": "Natural language explanation..."
}
```

## Configuration

- **Model**: GPT-2 (loaded on CPU by default)
- **Context Length**: Automatically trims long sentences to fit GPT-2's context window
- **Layers Analyzed**: All 12 layers of GPT-2 by default
- **LangGraph**: Uses in-memory state management (no persistence needed)

## Deployment on Hugging Face Spaces

This project is configured for deployment on Hugging Face Spaces:

1. Create a new Space with Python SDK
2. Upload all project files
3. Add your `GROQ_API_KEY` in Space settings (Secrets)
4. The app will automatically start on port 7860

**Note**: LangGraph runs entirely in-memory without requiring external databases or checkpointers, making it perfect for Hugging Face Spaces deployment.

## Limitations

- Only supports specific verb pairs (has/have, is/are, was/were, does/do)
- Requires sentences where verb agreement is determinable from context
- Long sentences are automatically trimmed to fit model context window

## Credits

Built using:
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for model interpretability
- [Groq](https://groq.com/) for fast LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
