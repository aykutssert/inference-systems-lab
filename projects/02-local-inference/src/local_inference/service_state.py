class ServiceState:
    def __init__(self) -> None:
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def mark_ready(self) -> None:
        self._ready = True

    def mark_not_ready(self) -> None:
        self._ready = False
