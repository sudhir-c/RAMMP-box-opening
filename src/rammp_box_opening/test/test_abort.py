import pytest

from rammp_box_opening.runtime.abort import AbortFlag, sigint_handler


def test_sigint_with_nothing_in_flight_raises_immediately():
    flag = AbortFlag()
    with pytest.raises(KeyboardInterrupt):
        sigint_handler(flag)(2, None)
    assert flag.requested  # the flag still records the request


def test_sigint_mid_goal_defers_to_the_cancel_path():
    flag = AbortFlag()
    flag.goal_in_flight = True
    sigint_handler(flag)(2, None)  # no raise: execute() delivers the cancel
    assert flag.requested


def test_second_sigint_escalates_even_in_flight():
    flag = AbortFlag()
    flag.goal_in_flight = True
    handler = sigint_handler(flag)
    handler(2, None)
    with pytest.raises(KeyboardInterrupt):
        handler(2, None)
