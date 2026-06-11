from local_inference.chat import build_prompt


class TokenizerStub:
    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        assert messages == [
            {
                "role": "user",
                "content": "hello",
            }
        ]
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "rendered prompt"


def test_build_prompt_disables_thinking() -> None:
    assert build_prompt(TokenizerStub(), "hello") == "rendered prompt"
