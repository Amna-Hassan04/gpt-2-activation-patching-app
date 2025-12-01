from typing import TypedDict, Dict, Any
from groq import Groq
import json
import re
import os

# ---------------------------
# 1. State definition
# ---------------------------
class PatchState(TypedDict):
    raw_output: str
    parsed: Dict[str, Any]
    explanation: str


# ---------------------------
# 2. Groq client
# ---------------------------
# It's better practice to load API key from environment variables
# For this example, replace 'YOUR_GROQ_API_KEY' with your actual key or load from .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------
# 3. Parser Agent
# ---------------------------
def parser_agent(state: PatchState) -> PatchState:
    raw = state["raw_output"]

    # Extract probabilities
    p_actual = re.findall(r"p\(actual\)\s*=\s*([\d\.eE+-]+)", raw)
    p_wrong = re.findall(r"p\(wrong\)\s*=\s*([\d\.eE+-]+)", raw)

    # Extract layer deltas (handling both positive and negative signs)
    layer_deltas = []
    for line in raw.split("\n"):
        m = re.search(r"layer\s+(\d+):\s+\u0394p\s*=\s*([\+?\-][\d\.eE+-]+)", line) # Adjusted regex for Δp with optional sign
        if m:
            layer_deltas.append({
                "layer": int(m.group(1)),
                "delta": float(m.group(2))
            })

    parsed = {
        "actual_prob": float(p_actual[0]) if p_actual else None,
        "wrong_prob": float(p_wrong[0]) if p_wrong else None,
        "layer_deltas": layer_deltas
    }

    state["parsed"] = parsed
    return state


# ---------------------------
# 4. Explanation Agent
# ---------------------------
def explanation_agent(state: PatchState) -> PatchState:
    parsed = json.dumps(state["parsed"], indent=2)

    prompt = f"""
You are a mechanistic interpretability expert.

Explain the following parsed activation-patching results in clear, intuitive language:

{parsed}

Explain:
- What the delta values mean
- Which layers are most important
- What this suggests about number-agreement circuits
- What overall conclusion we can draw

Avoid formulas. Be concise and clear.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You explain activation-patching outputs."},
            {"role": "user", "content": prompt}
        ]
    )

    state["explanation"] = response.choices[0].message.content
    return state

# Function to run the explanation pipeline, combining parsing and explanation
def generate_explanation(raw_output: str) -> Dict[str, Any]:
    initial_state: PatchState = {
        "raw_output": raw_output,
        "parsed": {},
        "explanation": ""
    }
    state_after_parsing = parser_agent(initial_state)
    final_state = explanation_agent(state_after_parsing)
    return final_state