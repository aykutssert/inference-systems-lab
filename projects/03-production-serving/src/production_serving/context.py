class ContextWindowExceededError(Exception):
    def __init__(
        self,
        prompt_tokens: int,
        max_tokens: int,
        context_window: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.context_window = context_window
        super().__init__("Context window exceeded")
