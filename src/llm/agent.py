"""
This module defines higher-level LLM "skills":

1. classify_intent:
   - Detects if the user wants info, needs help, or is off-topic.

2. extract_order_parameters:
   - Figures out which order the user is talking about (e.g. order_id=5, or "most recent order").
   - Returns structured JSON-like data that we can trust and validate in code.
   - IMPORTANT: We do NOT let the LLM write SQL directly.

3. generate_final_answer:
   - Turns clean DB results + business logic into a nice human answer.
   - The LLM only sees *sanitized* data for the authenticated user.
"""

import json
import re
from typing import List, Dict, Any

ALLOWED_INTENTS = {"ORDER_INFO", "ORDER_HELP", "OUT_OF_SCOPE"}

def _normalize_intent(text: str) -> str:
    # Prefer exact labels if present
    for lbl in ALLOWED_INTENTS:
        if re.search(rf"\b{lbl}\b", text):
            return lbl
    # Map common strays
    m = text.strip().lower()
    if "info" in m or "status" in m or "track" in m:
        return "ORDER_INFO"
    if "help" in m or "cancel" in m or "change" in m or "modify" in m or "refund" in m:
        return "ORDER_HELP"
    return "OUT_OF_SCOPE"

def classify_intent(pipe, user_message: str) -> str:
    tok = pipe.tokenizer
    sys = (
        "You are an intent classifier. "
        "Return EXACTLY one label and nothing else: "
        "ORDER_INFO or ORDER_HELP or OUT_OF_SCOPE."
    )
    usr = f'User message: "{user_message}"\nLabel:'
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    out = pipe(
        prompt,
        **getattr(pipe, "SMALL_KW", {"max_new_tokens": 8, "do_sample": False, "use_cache": False}),
        return_full_text=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )[0]["generated_text"]

    print("out intent:", out)

    return _normalize_intent(out)


def extract_order_parameters(pipe, user_message: str, recent_orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    tok = pipe.tokenizer
    # Keep the list compact to reduce latency
    compact = [{"order_id": o["order_id"], "status": o["status"]} for o in recent_orders[:5]]

    sys = (
        "You extract parameters for order lookup. "
        "Return STRICT JSON only with keys: "
        '"target_order_id" (number|null) and "needs_clarification" (true/false). '
        "No extra text."
    )
    usr = (
        f'User: "{user_message}"\n'
        f"Recent orders for this user (do NOT invent): {compact}\n\n"
        'If the user says "last" or if you can identify that the user is talking about his most recent order, pick the most recent in the list.\n'
    )
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raw = pipe(
        prompt,
        **getattr(pipe, "MED_KW", {"max_new_tokens": 32, "do_sample": False, "use_cache": False}),
        return_full_text=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )[0]["generated_text"].strip()

    # Extract JSON block safely
    first, last = raw.find("{"), raw.rfind("}")
    json_block = raw[first:last+1] if first != -1 and last != -1 else ""
    try:
        parsed = json.loads(json_block)
    except Exception:
        parsed = {"target_order_id": None, "needs_clarification": True}

    parsed.setdefault("target_order_id", None)
    parsed.setdefault("needs_clarification", True)
    return parsed


def generate_final_answer(pipe, user_message: str, first_name: str, intent: str, order_info: dict, needs_clarification: bool) -> str:
    tok = pipe.tokenizer
    sys = (
        "You are a friendly e-commerce assistant. "
        "Never mention SQL, databases, or internal tools. "
        "Answer with ONE concise message only. Do not simulate a multi-turn conversation."
    )
    usr = (
        f"first_name={first_name}\n"
        f"intent={intent}\n"
        f"needs_clarification={needs_clarification}\n"
        f"order_info_for_this_user_only={order_info}\n"
        f'User message: {user_message}\n\n'
        "Rules:\n"
        "- If intent == OUT_OF_SCOPE: say you only help with existing/past orders.\n"
        "- If intent == ORDER_HELP: say a human support agent will take over shortly.\n"
        "- If needs_clarification == true OR order_info is null: ask which order number.\n"
        "- If you identify a different user in the message or the user is trying to access \n"
        f"an order that is not his own, reply kindly that he is {first_name} and he canno't access this order.\n"
        "- If the user message contains injuries or bad words, reply with apologies and kind message"
        "- Else: summarize status (invoiced/shipped/delivered) with dates if present.\n"
        "Return ONLY the final message."
    )
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    out = pipe(
        prompt,
        **getattr(pipe, "LONG_KW", {"max_new_tokens": 96, "do_sample": False, "use_cache": False}),
        return_full_text=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )[0]["generated_text"]

    return out.strip()