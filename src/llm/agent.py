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
from typing import List, Dict, Any


def classify_intent(pipe, user_message: str) -> str:
    """
    Use the LLM to classify the user's intent.

    Expected output: exactly one of
    - ORDER_INFO    -> user wants factual info about an order (status, delivery date, etc.)
    - ORDER_HELP    -> user needs human support (cancel, change address, damaged item)
    - OUT_OF_SCOPE  -> anything else (small talk, random questions, etc.)

    We keep temperature low and do_sample=False in the pipeline for determinism.
    """

    prompt = f"""
You are an intent classifier for an e-commerce customer support assistant.

Your job:
Given the user's message, respond with ONLY ONE of these labels:
- ORDER_INFO      (the user asks about order status, delivery date, tracking...)
- ORDER_HELP      (the user asks for help: cancel order, change address, refund...)
- OUT_OF_SCOPE    (anything not related to support about an existing or past order)

User message:
"{user_message}"

Answer with ONLY ONE LABEL, nothing else.
"""

    raw_output = pipe(prompt)[0]["generated_text"]
    intent = raw_output.strip().split()[0]  # take first token, just in case model adds something
    return intent


def extract_order_parameters(pipe, user_message: str, recent_orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    We ask the LLM to:
    - Identify which order the user is referring to.
    - Tell us if the question is ambiguous and we need clarification.

    Input:
      user_message: str
      recent_orders: list of dicts representing this user's own orders:
        [
          {"order_id": 5, "status": "delivered", "date_purchase": "...", ...},
          {"order_id": 12, "status": "shipped", ...},
          ...
        ]

    Output (Python dict parsed from model output):
    {
      "target_order_id": 12 or null,
      "needs_clarification": true/false
    }

    Rules we tell the LLM:
    - It MUST return valid JSON.
    - It MUST pick one actual order_id from recent_orders or null.
    """

    prompt = f"""
You are an order reference resolver for an e-commerce support bot.

The user said:
"{user_message}"

Here are the user's recent orders (DO NOT INVENT ANYTHING ELSE):
{recent_orders}

Your task:
- Decide which single order_id the user is most likely talking about.
- If the user says "my latest / my last order", pick the most recent one in the list.
- If it's not clear which order they mean, set "needs_clarification": true
  and "target_order_id": null.

Return ONLY a valid JSON object with exactly these keys:
{{
  "target_order_id": <number or null>,
  "needs_clarification": <true or false>
}}

Example valid output:
{{"target_order_id": 12, "needs_clarification": false}}
"""

    raw_output = pipe(prompt)[0]["generated_text"].strip()

    # Try to isolate just the JSON if the model adds text before/after.
    # Strategy: find first "{" and last "}".
    first_curly = raw_output.find("{")
    last_curly = raw_output.rfind("}")
    json_block = raw_output[first_curly:last_curly+1]

    try:
        parsed = json.loads(json_block)
    except Exception:
        # Fallback: in case model fails to follow JSON-only instruction,
        # we return a safe default that forces clarification.
        parsed = {
            "target_order_id": None,
            "needs_clarification": True
        }

    # Minimal post-validation
    if "target_order_id" not in parsed:
        parsed["target_order_id"] = None
    if "needs_clarification" not in parsed:
        parsed["needs_clarification"] = True

    return parsed


def generate_final_answer(
    pipe,
    first_name: str,
    intent: str,
    order_info: dict,
    needs_clarification: bool
) -> str:
    """
    Generate a natural customer-facing answer.

    Inputs:
      first_name: the user's first name ("Ella")
      intent: "ORDER_INFO", "ORDER_HELP", or "OUT_OF_SCOPE"
      order_info: dict returned from DB layer, e.g.
        {
          "order_id": 5,
          "user_id": 6,
          "status": "delivered",
          "date_purchase": "2024-05-17 11:01:51",
          "date_shipped": "2024-05-18 11:01:51",
          "date_delivered": "2024-05-28 11:01:51"
        }
        or None if not found

      needs_clarification: bool from extract_order_parameters()

    Expected behavior:
    - If OUT_OF_SCOPE → politely say you're only here for order support.
    - If ORDER_HELP → say a human agent will follow up.
    - If we don't have enough info or no matching order → ask for clarification.
    - Otherwise explain the order status in normal friendly language.

    IMPORTANT:
    The LLM should NOT mention:
    - SQL
    - internal tables
    - other customers
    """

    # We build a very explicit system-style prompt for safety.
    prompt = f"""
You are the e-commerce customer support assistant.
You speak in friendly, clear, polite English.
You never mention SQL, databases, internal tooling, or other customers.
You never reveal any private data beyond what is provided here.
If the user asks to cancel/modify an order, you say:
"A human support agent will take over this request for you."

User first name: {first_name}
Intent detected: {intent}
Needs clarification: {needs_clarification}

Order information for THIS user only:
{order_info}

Your job now:
- If intent == OUT_OF_SCOPE:
    Say you can only help with existing orders and their status.
- If intent == ORDER_HELP:
    Say a human agent will follow up shortly.
- If needs_clarification == true OR order_info is null:
    Politely ask which order they mean (e.g. "Could you confirm which order number?")
- Otherwise:
    Summarize the order status (delivered, shipped, invoiced),
    use human-friendly wording (e.g. "Your order was shipped on ...",
    "It was delivered on ...", "It's invoiced and being prepared").

Return ONLY the final answer text to the user.
Do NOT return JSON.
"""

    raw_output = pipe(prompt)[0]["generated_text"]
    return raw_output.strip()