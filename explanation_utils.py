from typing import TypedDict, Dict, Any
from groq import Groq
import json
import re
import os
from langgraph.graph import StateGraph, END

# ---------------------------
# 1. State definition
# ---------------------------
class PatchState(TypedDict):
    raw_output: str
    parsed: Dict[str, Any]
    explanation: str
    error: str


# ---------------------------
# 2. Groq client
# ---------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------
# 3. Parser Agent (Node)
# ---------------------------
def parser_agent(state: PatchState) -> PatchState:
    """
    Parses raw activation patching output and extracts structured data.
    """
    raw = state["raw_output"]

    try:
        # Extract probabilities
        p_actual = re.findall(r"p\(actual\)\s*=\s*([\d\.eE+-]+)", raw)
        p_wrong = re.findall(r"p\(wrong\)\s*=\s*([\d\.eE+-]+)", raw)

        # Extract layer deltas (handling both positive and negative signs)
        layer_deltas = []
        for line in raw.split("\n"):
            m = re.search(r"layer\s+(\d+):\s+\u0394p\s*=\s*([\+?\-][\d\.eE+-]+)", line)
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
        state["error"] = ""
        
    except Exception as e:
        state["error"] = f"Parser error: {str(e)}"
        state["parsed"] = {}
    
    return state


# ---------------------------
# 4. Explanation Agent (Node)
# ---------------------------
def explanation_agent(state: PatchState) -> PatchState:
    """
    Generates natural language explanation using Groq LLM.
    """
    # Check if parsing failed
    if state.get("error"):
        state["explanation"] = "Unable to generate explanation due to parsing error."
        return state
    
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

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You explain activation-patching outputs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        state["explanation"] = response.choices[0].message.content
        
    except Exception as e:
        state["explanation"] = f"Error generating explanation: {str(e)}"
    
    return state


# ---------------------------
# 5. Build LangGraph Pipeline
# ---------------------------
def build_explanation_graph():
    """
    Constructs the LangGraph workflow for explanation generation.
    """
    # Create a new graph
    workflow = StateGraph(PatchState)
    
    # Add nodes (agents)
    workflow.add_node("parser", parser_agent)
    workflow.add_node("explainer", explanation_agent)
    
    # Define edges (flow)
    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "explainer")
    workflow.add_edge("explainer", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


# ---------------------------
# 6. Main execution function
# ---------------------------
# Build the graph once at module import
explanation_graph = build_explanation_graph()

def generate_explanation(raw_output: str) -> Dict[str, Any]:
    """
    Runs the LangGraph pipeline to generate explanations.
    
    Args:
        raw_output: Raw text output from activation patching
        
    Returns:
        Dict containing parsed data and explanation
    """
    initial_state: PatchState = {
        "raw_output": raw_output,
        "parsed": {},
        "explanation": "",
        "error": ""
    }
    
    # Execute the graph
    final_state = explanation_graph.invoke(initial_state)
    
    return final_state