from sim import env


def _call_transform(monkeypatch, reply):
    async def send_command(_command, _uri=None):
        return reply

    monkeypatch.setattr(env, "SendCommand", send_command)
    return env.TransformAgent((0, 0, 0), (0, 0, 0))


def test_transform_agent_parses_old_three_line_v1_reply(monkeypatch):
    state = _call_transform(
        monkeypatch,
        "Agent position: (1.0, 2.0, 3.0)\n"
        "Agent rotation: (4.0, 5.0, 6.0)\n"
        "Is colliding: False",
    )

    assert state == {
        "translation": (1.0, 2.0, 3.0),
        "rotation": (4.0, 5.0, 6.0),
        "isColliding": False,
        "out_of_bounds_recovery_count": None,
    }


def test_transform_agent_parses_new_four_line_v1_reply_without_extra_tuple(monkeypatch):
    reply = (
        "Agent position: (1.0, 2.0, 3.0)\n"
        "Agent rotation: (4.0, 5.0, 6.0)\n"
        "Is colliding: True\n"
        "Out-of-bounds recovery count: 12"
    )

    state = _call_transform(monkeypatch, reply)

    assert reply.count("(") == 2
    assert state["isColliding"] is True
    assert state["out_of_bounds_recovery_count"] == 12
