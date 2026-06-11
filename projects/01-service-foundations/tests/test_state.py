from service_foundations.state import ServiceState


def test_service_state_lifecycle() -> None:
    state = ServiceState()

    assert state.is_ready is False

    state.mark_ready()
    assert state.is_ready is True

    state.mark_not_ready()
    assert state.is_ready is False
