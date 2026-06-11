from typing import Any


def build_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("Chat template did not return text")
    return rendered
