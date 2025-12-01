from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from patching_logic import run_user_activation_pipeline
from explanation_utils import generate_explanation # New import
import json

# Instantiate FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] ,
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)

# Mount static files directory
# Make sure to create a 'static' folder and place index.html inside it
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html from the root URL
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

# Define Pydantic model for the request body
class SentenceRequest(BaseModel):
    sentence: str

# Define the POST endpoint
@app.post("/predict")
async def predict_verb_agreement(request: SentenceRequest):
    user_sentence = request.sentence
    
    # Process the sentence using the activation patching logic
    patching_results = run_user_activation_pipeline(user_sentence)

    if "error" in patching_results:
        return patching_results

    # Reconstruct the raw output string for the explanation agent
    output_lines = []
    output_lines.append("\n--- Summary ---")
    output_lines.append(f"User sentence: {patching_results['user_sentence']}")
    output_lines.append(f"Prefix used: {repr(patching_results['prefix_used_for_scoring'])}")
    output_lines.append(f"Verb pair (singular/plural): {patching_results['verb_pair']}")
    output_lines.append(f"Actual verb found in sentence: {patching_results['actual_verb_in_sentence']}")
    output_lines.append(f"Constructed bad sentence: {patching_results['bad_sentence']}")
    output_lines.append("")
    output_lines.append("Raw GPT-2 next-token probs for the actual vs wrong token (from prefix):")
    output_lines.append(" p(actual) = {:.6f}".format(patching_results["p_actual_token_raw"]))
    output_lines.append(" p(wrong)  = {:.6f}".format(patching_results["p_wrong_token_raw"])) 
    output_lines.append("")
    output_lines.append("Reference ordering (singular/plural) probs from prefix:")
    output_lines.append(" p(singular) = {:.6f}".format(patching_results["p_singular"])) 
    output_lines.append(" p(plural)   = {:.6f}".format(patching_results["p_plural"])) 
    output_lines.append("")
    output_lines.append("Layer-wise p(correct token) after patching (len = {})".format(
        len(patching_results["layer_probs_correct_after_patch"])
    ))

    for i, v in enumerate(patching_results["layer_probs_correct_after_patch"]):
        output_lines.append(f" layer {i:02d}: p(correct) = {v:.6f}")

    correct_token = patching_results["actual_verb_in_sentence"]
    orig_p_correct = patching_results["p_plural"] if correct_token == patching_results["verb_pair"][1] else patching_results["p_singular"]
    diffs = [v - orig_p_correct for v in patching_results["layer_probs_correct_after_patch"]]
    top_increases = sorted(enumerate(diffs), key=lambda x: x[1], reverse=True)[:6]

    output_lines.append("\nTop layers by increase in p(correct) due to patching (patch_from_good->bad):")
    for layer_idx, diff in top_increases:
        output_lines.append(
            f" layer {layer_idx:02d}: \u0394p = {diff:+.6f} "
            f"(patched p = {patching_results['layer_probs_correct_after_patch'][layer_idx]:.6f})"
        )
    raw_output_string = "\n".join(output_lines)

    # Generate explanation
    explanation_output = generate_explanation(raw_output_string)

    return {
        "patching_results": patching_results,
        "explanation": explanation_output["explanation"]
    }
