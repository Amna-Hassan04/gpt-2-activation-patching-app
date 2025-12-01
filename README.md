# GPT-2 Activation Patching Demo

## Overview
This project provides an interactive web application to explore the internal mechanisms of GPT-2, specifically focusing on how the model processes **verb agreement**. It leverages activation patching to pinpoint which layers of the GPT-2 model are crucial for maintaining grammatical correctness. To make complex mechanistic interpretability results accessible, the application integrates an AI agent (powered by the Groq API) that provides human-readable explanations of the patching outcomes.

## Features

*   **Interactive Input**: Users can input sentences containing common verb agreement pairs (e.g., 'is'/'are', 'has'/'have', 'was'/'were', 'does'/'do').
*   **Automated Sentence Variation**: The application automatically generates a 'wrong' version of the input sentence by swapping the verb to its incorrect form (e.g., "The cat *has* fur" becomes "The cat *have* fur").
*   **GPT-2 Activation Patching**: Performs layer-wise activation patching on a pre-trained GPT-2 model to understand the causal role of specific layers in verb agreement.
*   **Probability Analysis**: Displays raw GPT-2 next-token probabilities for both the actual and the 'wrong' verb, and layer-wise probabilities after patching.
*   **AI-Generated Explanations**: A Groq-powered AI agent interprets the patching results and generates clear, intuitive explanations, identifying influential layers and insights into number-agreement circuits.
*   **Modern Frontend**: A clean, responsive web interface for an enhanced user experience.

## Live Demo (Placeholder)

**[🚀 View Live Demo Here (Update Link After Deployment)](https://your-huggingface-space-url.hf.space/)**

*Please note: The live demo link above is a placeholder. You will need to replace it with the actual URL of your deployed Hugging Face Space.* 

## Technologies Used

*   **Backend**: FastAPI (Python)
*   **Activation Patching**: `transformer_lens` (Python)
*   **AI Explanation**: Groq API (`openai/gpt-oss-20b` model)
*   **Frontend**: HTML, CSS, JavaScript
*   **Model**: GPT-2 (via Hugging Face `transformers` and `transformer_lens`)

## Setup and Local Development

To run this application locally, follow these steps:

1.  **Clone the Repository (or download files):**
    ```bash
    git clone https://github.com/your-username/gpt2-patching-demo.git
    cd gpt2-patching-demo
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Groq API Key:**
    Obtain an API key from [Groq](https://console.groq.com/keys). Create a `.env` file in the root of your project and add your API key:
    ```
    GROQ_API_KEY="gsk_YOUR_GROQ_API_KEY_HERE"
    ```
    *Make sure to replace `gsk_YOUR_GROQ_API_KEY_HERE` with your actual key.* Alternatively, set it as an environment variable before running the app.

5.  **Run the FastAPI Backend:**
    Ensure your `index.html` is inside a `static/` directory.
    ```bash
    uvicorn main:app --reload
    ```
    The backend will typically run on `http://127.0.0.1:8000`.

6.  **Access the Frontend:**
    Open your web browser and navigate to `http://127.0.0.1:8000`. The FastAPI application will serve the `index.html` file, and you can interact with the demo.

## Deployment on Hugging Face Spaces

This application is well-suited for deployment on [Hugging Face Spaces](https://huggingface.co/spaces) as a single Python Space, serving both the FastAPI backend and the static HTML/JavaScript frontend.

1.  **File Structure**: Ensure your project has the following structure:
    ```
    your-space-repo/
    ├── main.py
    ├── patching_logic.py
    ├── explanation_utils.py
    ├── requirements.txt
    ├── static/
    │   └── index.html
    └── app.py
    ```

2.  **`requirements.txt`**: This file should list all Python dependencies:
    ```
    fastapi
uvicorn
pydantic
transformer_lens
torch
datasets
transformers
sentencepiece
einops
groq
    ```

3.  **`app.py` (Entry Point for Hugging Face Spaces)**:
    Create an `app.py` file in the root of your repository to tell Hugging Face how to run your FastAPI application. Hugging Face Spaces often expose port `7860` for Python apps.
    ```python
    import subprocess
    import os

    port = os.getenv("PORT", "7860") # HF Spaces typically use PORT 7860
    command = f"uvicorn main:app --host 0.0.0.0 --port {port}"
    subprocess.run(command, shell=True)
    ```

4.  **Hugging Face Space Setup**:
    *   Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
    *   Choose a **Space name**, **license**, and select `Gradio` or `Streamlit` as the **SDK** (this provides a Python environment).
    *   Set **Visibility** to Public (for free tier).
    *   Upload all your project files (`main.py`, `patching_logic.py`, `explanation_utils.py`, `requirements.txt`, `static/index.html`, `app.py`) to the Space's Git repository.

5.  **Environment Variables (Groq API Key on HF Spaces)**:
    *   On your Hugging Face Space page, navigate to "Settings" (usually under the "App" tab).
    *   Find the "Repository secrets" section and add a new secret named `GROQ_API_KEY` with your Groq API key as its value.

6.  **Update Frontend API Endpoint**: If you modified `index.html` to point to `http://127.0.0.1:8000`, you should change it back to a relative path `/predict` or your deployed HF Space URL. Since we're serving `index.html` from the *same* FastAPI app, a relative path is ideal. (The provided `static/index.html` uses `http://127.0.0.1:8000/predict` which you *must* change to `/predict` or your deployed space's URL if deployed separately).
    *   **If serving `index.html` from the *same* FastAPI app on HF Spaces**, ensure the fetch call in `static/index.html` looks like this:
        ```javascript
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sentence: inputValue })
        });
        ```

### Important Considerations for Free Tiers

GPT-2 is a relatively large language model (around 500MB). Deploying it on free hosting tiers (including Hugging Face Spaces' free tier) comes with significant limitations:

*   **Memory Constraints**: Free tiers often have strict RAM limits. Loading GPT-2, along with Python, dependencies, and caching, can easily lead to Out-Of-Memory (OOM) errors, causing your application to crash or fail to start.
*   **Cold Starts**: When your Space is inactive, free tiers may spin down your container. A 'cold start' will involve reloading the entire GPT-2 model, leading to very long initial response times or timeouts.
*   **CPU Limits**: Free tiers typically offer limited CPU resources, which can make the model inference process slow.

While this setup is excellent for demonstration and learning, you should expect performance and reliability challenges on free-tier infrastructure. For a production-ready application, upgrading to a paid tier or a platform optimized for ML model deployment with more generous resources would be necessary.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
