# tests/test_llm_agent.py
from src.llm.agent import (
    classify_intent,
    extract_order_parameters,
    generate_final_answer,
)
from .conftest import FakePipe


def test_classify_intent_order_info():
    """Classifier should detect ORDER_INFO when user asks about order status."""
    pipe = FakePipe(outputs=["ORDER_INFO"])
    intent = classify_intent(pipe, "Where is my last order?")
    assert intent == "ORDER_INFO"


def test_classify_intent_out_of_scope():
    """Should return OUT_OF_SCOPE for unrelated questions."""
    pipe = FakePipe(outputs=["OUT_OF_SCOPE"])
    intent = classify_intent(pipe, "Tell me a joke.")
    assert intent == "OUT_OF_SCOPE"


def test_extract_order_parameters_last_order(sample_orders):
    """
    If user asks for 'my last order', the agent should pick the most recent one.
    Here we assume the agent uses the last element in the list as 'latest'.
    """
    json_output = '{"target_order_id": 7, "needs_clarification": false}'
    pipe = FakePipe(outputs=[json_output])

    params = extract_order_parameters(pipe, "What is the status of my last order?", sample_orders)

    assert params["target_order_id"] == 7
    assert params["needs_clarification"] is False


def test_extract_order_parameters_ambiguous(sample_orders):
    """If the LLM output is not valid JSON, fallback to clarification."""
    pipe = FakePipe(outputs=["I am not sure which order."])

    params = extract_order_parameters(pipe, "What about my order?", sample_orders)

    assert params["target_order_id"] is None
    assert params["needs_clarification"] is True


def test_generate_final_answer_with_order_info():
    """When order_info is available, final answer should mention status."""
    pipe = FakePipe(outputs=["Your order #5 has been delivered."])
    order_info = {
        "order_id": 5,
        "status": "delivered",
        "date_delivered": "2024-05-28 11:01:51",
    }

    answer = generate_final_answer(
        pipe=pipe,
        first_name="Ella",
        intent="ORDER_INFO",
        order_info=order_info,
        needs_clarification=False,
    )

    assert "order #5" in answer or "order 5" in answer
    assert "delivered" in answer.lower()