import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from datasets import load_dataset

# --- LOAD MODEL (Loaded only once when the module is imported) ---
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

# ============================================================
#               SAFE TRIMMING HELPERS FOR LONG INPUT
# ============================================================

def trim_to_context(text, keep_last_tokens=None):
    """
    Trim `text` so its tokenized length <= model.cfg.n_ctx.
    Keeps the LAST tokens in the sequence (closest to the verb).
    """
    toks = model.tokenizer.encode(text)
    n_ctx = model.cfg.n_ctx
    max_keep = n_ctx if keep_last_tokens is None else min(keep_last_tokens, n_ctx)

    if len(toks) > max_keep:
        toks = toks[-max_keep:]
        return model.tokenizer.decode(toks)
    return text


def trim_pair_for_patching(good_sentence, bad_sentence):
    """
    Trim both good & bad sentences to same-size windows <= n_ctx.
    Ensures cache shapes match when patching.
    """
    good_toks = model.tokenizer.encode(good_sentence)
    bad_toks  = model.tokenizer.encode(bad_sentence)
    n_ctx = model.cfg.n_ctx

    if len(good_toks) > n_ctx or len(bad_toks) > n_ctx:
        good_toks = good_toks[-n_ctx:]
        bad_toks  = bad_toks[-n_ctx:]
        return model.tokenizer.decode(good_toks), model.tokenizer.decode(bad_toks)
    else:
        return good_sentence, bad_sentence


# ============================================================
#                  UPDATED NEXT-TOKEN SCORING
# ============================================================

def score_next_token(sentence_prefix, token1, token2):
    """
    Safely compute p(token1), p(token2) after a prefix.
    Automatically trims prefix to fit GPT-2 context.
    """
    max_prefix_toks = max(1, model.cfg.n_ctx - 1)
    prefix_trimmed = trim_to_context(sentence_prefix, keep_last_tokens=max_prefix_toks)

    tokens = model.to_tokens(prefix_trimmed)
    logits = model(tokens)[0, -1]
    probs = F.softmax(logits, dim=-1)

    t1 = model.tokenizer.encode(" " + token1)[0]
    t2 = model.tokenizer.encode(" " + token2)[0]

    return float(probs[t1].detach().cpu().numpy()), float(probs[t2].detach().cpu().numpy())


# ============================================================
#        DETECT VERB PAIR + BUILD BAD SENTENCE VARIANT
# ============================================================

def detect_and_build_variants(user_sentence):
    verb_pairs = [
        ("has", "have"),
        ("is", "are"),
        ("was", "were"),
        ("does", "do"),
    ]

    s = user_sentence.strip()
    s_lower = s.lower()

    for singular, plural in verb_pairs:
        if singular in s_lower or plural in s_lower:
            # Identify actual & wrong verb
            if singular in s_lower:
                split_token = singular
                actual = singular
                wrong = plural
            else:
                split_token = plural
                actual = plural
                wrong = singular

            idx = s_lower.rfind(split_token)
            prefix = s[:idx] if idx != -1 else s.rsplit(split_token, 1)[0]
            token_len = len(split_token)
            remainder = s[idx + token_len:]

            bad_sentence = (prefix + wrong + remainder).strip()

            return prefix, actual, wrong, bad_sentence, (singular, plural)

    return None, None, None, None, None


# ============================================================
#             UPDATED ACTIVATION PATCHING (SAFE)
# ============================================================

def patch_layer_user(layer, good_sentence, bad_sentence, verb_pair):
    """
    Patch block.layer.attn.hook_z from good->bad.
    Sentences are trimmed to same token length for safe patching.
    """
    # Tokenize both sentences
    good_toks = model.tokenizer.encode(good_sentence)
    bad_toks  = model.tokenizer.encode(bad_sentence)

    # Trim to shortest length
    min_len = min(len(good_toks), len(bad_toks))

    # We now operate on raw tokens, ensuring we don't accidentally introduce BOS
    # or other tokenizer-specific issues that cause length mismatches.
    # The `run_with_cache` and `run_with_hooks` will be called with prepend_bos=False
    # to maintain this consistency.
    good_trimmed_tokens = good_toks[-min_len:]
    bad_trimmed_tokens  = bad_toks[-min_len:]

    good_trimmed_str = model.tokenizer.decode(good_trimmed_tokens)
    bad_trimmed_str  = model.tokenizer.decode(bad_trimmed_tokens)

    # Get cache for good sentence. Explicitly set prepend_bos=False.
    _, cache_good = model.run_with_cache(good_trimmed_str, prepend_bos=False)

    # Patch only matching sequence length
    def patch_hook(value, hook):
        # Since prepend_bos=False was used for run_with_cache,
        # cache_good[hook.name] will have the sequence length of good_trimmed_str (min_len).
        # The 'value' tensor in the hook will also have sequence length min_len because
        # run_with_hooks is also called with prepend_bos=False.
        # Thus, direct replacement is safe.
        return cache_good[hook.name]

    # Run patched logits. Explicitly set prepend_bos=False.
    patched_logits = model.run_with_hooks(
        bad_trimmed_str,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_hook)],
        prepend_bos=False
    )

    probs = F.softmax(patched_logits[0, -1], dim=-1)
    singular, plural = verb_pair
    # Tokenize for next-token prediction, these are usually just single tokens
    t_sing = model.tokenizer.encode(" " + singular)[0]
    t_plur = model.tokenizer.encode(" " + plural)[0]

    return float(probs[t_sing].detach().cpu().numpy()), float(probs[t_plur].detach().cpu().numpy())


# ============================================================
#                  FULL USER PIPELINE
# ============================================================

def run_user_activation_pipeline(user_sentence, n_layers_to_check=None):
    prefix, actual, wrong, bad_sentence, verb_pair = detect_and_build_variants(user_sentence)

    if prefix is None:
        return {"error": "No supported verb pair found (has/have, is/are, was/were, does/do)."}

    p_actual, p_wrong = score_next_token(prefix, actual, wrong)

    singular, plural = verb_pair
    correct_token = actual

    p_sing, p_plur = score_next_token(prefix, singular, plural)

    n_layers = model.cfg.n_layers
    if n_layers_to_check is None:
        n_layers_to_check = n_layers

    layer_probs_correct = []
    for layer in range(min(n_layers_to_check, n_layers)):
        p_sing_patched, p_plur_patched = patch_layer_user(layer, user_sentence, bad_sentence, verb_pair)
        p_correct = p_plur_patched if correct_token == plural else p_sing_patched
        layer_probs_correct.append(p_correct)

    return {
        "user_sentence": user_sentence,
        "prefix_used_for_scoring": prefix,
        "verb_pair": verb_pair,
        "actual_verb_in_sentence": actual,
        "wrong_verb_used_for_bad_sentence": wrong,
        "bad_sentence": bad_sentence,
        "p_actual_token_raw": p_actual,
        "p_wrong_token_raw": p_wrong,
        "p_singular": p_sing,
        "p_plural": p_plur,
        "layer_probs_correct_after_patch": layer_probs_correct,
    }