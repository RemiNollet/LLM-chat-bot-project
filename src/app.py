"""
Streamlit app that simulates the whole assistant conversation.

High-level flow for each user message:
1. Sanitize input.
2. Classify intent.
3. If intent == ORDER_HELP:
      -> immediately answer "A human agent will contact you".
4. Else if intent == ORDER_INFO:
      -> fetch user's recent orders
      -> ask LLM to pick which order we're talking about
      -> fetch that order status from DB (with ownership check)
      -> generate final answer
5. Else (OUT_OF_SCOPE):
      -> politely say we're only for order support.

For demo purposes, we'll assume the authenticated user is fixed:
user_id=6, first_name="Ella", email="enim.non@outlook.com".
In production this would come from login / session.
"""

import streamlit as st
import logging

from llm.model_loader import load_llm
from llm.agent import (
    classify_intent,
    extract_order_parameters,
    generate_final_answer
)

from db.queries import (
    fetch_orders_for_user,
    fetch_order_status
)

from security.auth import (
    sanitize_user_input,
    verify_order_ownership
)

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

logging.info("Dont forget to install database following instructions in src/db/connection.py !!")
logging.info("Dont forget to create .env file with your HF_TOKEN !!")

# -------------------------
# Simulated authenticated user context
# -------------------------
CURRENT_USER = {
    "user_id": 6,
    "first_name": "Ella",
    "email": "enim.non@outlook.com"
}


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """
    Cache the LLM pipeline so we don't reload the model for every request.
    Streamlit will keep this in memory between interactions.
    """
    pipe = load_llm()
    logger.info("Model loaded successfully.")
    return pipe


def handle_user_message(user_message: str) -> str:
    """
    Core orchestration logic:
    Takes the user's raw message,
    returns the final assistant response string.
    """
    logger.info("------------------------------------------------------------")
    logger.info(f"🧍 User message: {user_message}")

    # Step 1: sanitize input
    clean_message = sanitize_user_input(user_message)
    if clean_message != user_message:
        logger.info(f"Sanitized user input: {clean_message}")

    # Step 2: load model pipeline
    pipe = get_pipeline()

    # Step 3: classify intent
    intent = classify_intent(pipe, clean_message)
    logger.info(f"Detected intent: {intent}")

    # Handle OUT_OF_SCOPE early
    if intent == "OUT_OF_SCOPE":
        logger.info("Intent OUT_OF_SCOPE → returning polite default answer.")
        return (
            "I can help you with information about your existing or past orders "
            "(status, delivery, shipping, etc.). How can I help you today?"
        )

    # Handle ORDER_HELP early
    if intent == "ORDER_HELP":
        logger.info("Intent ORDER_HELP → escalating to human support.")
        return (
            "Thanks for your message! A human support agent will take over this request shortly."
        )

    # At this point → ORDER_INFO
    user_id = CURRENT_USER["user_id"]
    first_name = CURRENT_USER["first_name"]

    # Step 4: fetch recent orders for this user
    recent_orders = fetch_orders_for_user(user_id)
    logger.info(f"Found {len(recent_orders)} recent orders for user {user_id}.")

    # Step 5: extract which order the user is talking about
    params = extract_order_parameters(pipe, clean_message, recent_orders)
    logger.info(f"Parameter extraction output: {params}")

    target_order_id = params.get("target_order_id")
    needs_clarification = params.get("needs_clarification", False)

    # Step 6: fetch that order (if one identified)
    order_info = None
    if target_order_id is not None:
        order_info = fetch_order_status(user_id, target_order_id)
        if not verify_order_ownership(user_id, order_info):
            logger.warning(
                f"Order {target_order_id} does not belong to user {user_id}. Access blocked."
            )
            order_info = None
        else:
            logger.info(
                f"Order {target_order_id} found for user {user_id}, status = {order_info.get('status')}."
            )
    else:
        logger.info("ℹNo specific order ID identified.")

    # Step 7: generate final answer
    logger.info("Generating final LLM answer...")
    final_answer = generate_final_answer(
        pipe=pipe,
        first_name=first_name,
        intent=intent,
        order_info=order_info,
        needs_clarification=needs_clarification or (order_info is None)
    )

    # Log raw LLM output (can be multi-line)
    logger.info("Final LLM answer:")
    for line in final_answer.splitlines():
        logger.info(f"    {line}")

    logger.info("------------------------------------------------------------\n")

    return final_answer

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Customer Support Assistant", page_icon="🛍️")

st.title("Customer Support Assistant")
st.write(
    "Ask about your past or current orders (delivery status, shipping date, etc.)."
)

if "chat_history" in st.session_state:
    pass
else:
    st.session_state["chat_history"] = []

# Display existing conversation
for role, msg in st.session_state["chat_history"]:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")

# User input box
user_input = st.text_input(
    "Your message:",
    placeholder="Where is my last order? Can I get an update on order #5?",
)

if st.button("Send") and user_input:
    # Append user message
    st.session_state["chat_history"].append(("user", user_input))

    # Process the message
    bot_reply = handle_user_message(user_input)

    # Append bot reply
    st.session_state["chat_history"].append(("assistant", bot_reply))

    # Rerender
    st.experimental_rerun()