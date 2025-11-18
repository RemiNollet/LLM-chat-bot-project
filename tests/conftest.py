# tests/conftest.py
import pytest


class FakeTokenizer:
    """
    Minimal tokenizer mock compatible with agent.py.
    It only needs `apply_chat_template` for our tests.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Very simple implementation: join roles and content
        parts = []
        for msg in messages:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        return "\n".join(parts)


class FakePipe:
    """
    Simple fake HF pipeline for unit tests.
    It returns predetermined outputs instead of calling a real LLM.
    """

    def __init__(self, outputs):
        # outputs is a list of strings that will be returned in order
        self.outputs = outputs
        self.call_idx = 0
        self.tokenizer = FakeTokenizer()

    def __call__(self, prompt, **kwargs):
        # Always return one dict with "generated_text"
        if self.call_idx >= len(self.outputs):
            text = ""
        else:
            text = self.outputs[self.call_idx]
            self.call_idx += 1
        return [{"generated_text": text}]


@pytest.fixture
def sample_orders():
    """
    Sample orders for testing parameter extraction.
    """
    return [
        {"order_id": 3, "status": "shipped"},
        {"order_id": 5, "status": "delivered"},
        {"order_id": 7, "status": "invoiced"},
    ]