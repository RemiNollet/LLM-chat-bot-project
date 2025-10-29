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
    return load_llm()


def handle_user_message(user_message: str) -> str:
    """
    Core orchestration logic:
    Takes the user's raw message,
    returns the final assistant response string.
    """

    # 1. Sanitize input (light illustrative step)
    clean_message = sanitize_user_input(user_message)

    pipe = get_pipeline()

    # 2. Detect intent
    intent = classify_intent(pipe, clean_message)

    # Handle OUT_OF_SCOPE early
    if intent == "OUT_OF_SCOPE":
        return (
            "I can help you with information about your existing or past orders "
            "(status, delivery, etc.). How can I help with one of your orders?"
        )

    # Handle ORDER_HELP early (handover to human)
    if intent == "ORDER_HELP":
        return (
            "Thanks for letting me know. A human support agent will take over "
            "this request for you shortly."
        )

    # At this point we assume ORDER_INFO (user wants factual info like 'Where is my order?')
    user_id = CURRENT_USER["user_id"]
    first_name = CURRENT_USER["first_name"]

    # 3. Get recent orders for that user
    recent_orders = fetch_orders_for_user(user_id)

    # 4. Ask LLM which order the user means
    params = extract_order_parameters(pipe, clean_message, recent_orders)

    target_order_id = params.get("target_order_id")
    needs_clarification = params.get("needs_clarification", False)

    # 5. Fetch that specific order from DB
    order_info = None
    if target_order_id is not None:
        order_info = fetch_order_status(user_id, target_order_id)

        # Double-check ownership for safety
        if not verify_order_ownership(user_id, order_info):
            # If it's not theirs, pretend we just didn't find it
            order_info = None

    # 6. Ask the LLM to craft the final user-facing answer
    final_answer = generate_final_answer(
        pipe=pipe,
        first_name=first_name,
        intent=intent,
        order_info=order_info,
        needs_clarification=needs_clarification or (order_info is None)
    )

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