from marlenv.utils import Schedule, MultiSchedule


def is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def test_linear_schedule_increasing():
    s = Schedule.linear(0, 1, 10)
    for i in range(10):
        assert is_close(s.value, i / 10)
        s.update()
    for i in range(10):
        assert s.value == 1.0
        s.update()


def test_linear_schedule_decreasing():
    s = Schedule.linear(0, -1, 10)
    for i in range(10):
        assert is_close(s.value, -i / 10)
        s.update()
    for i in range(10):
        assert s.value == -1.0
        s.update()


def test_linear_schedule_set_timestep():
    s = Schedule.linear(0, 1, 10)
    s.update(50)
    assert is_close(s.value, 1)

    s.update(0)
    assert is_close(s.value, 0)

    s.update(5)
    assert is_close(s.value, 0.5)


def test_exp_schedule_increasing():
    s = Schedule.exp(1, 16, 5)
    assert is_close(s.value, 1)
    s.update()
    assert is_close(s.value, 2)
    s.update()
    assert is_close(s.value, 4)
    s.update()
    assert is_close(s.value, 8)
    s.update()
    assert is_close(s.value, 16)
    for _ in range(10):
        s.update()
        assert is_close(s.value, 16)


def test_exp_schedule_set_timestep():
    s = Schedule.exp(1, 16, 5)
    s.update(50)
    assert is_close(s.value, 16)

    s.update(0)
    assert is_close(s.value, 1)

    s.update(5)
    assert is_close(s.value, 16)

    s.update(1)
    assert is_close(s.value, 2)


def test_exp_schedule_decreasing():
    s = Schedule.exp(16, 1, 5)
    assert is_close(s.value, 16)
    s.update()
    assert is_close(s.value, 8)
    s.update()
    assert is_close(s.value, 4)
    s.update()
    assert is_close(s.value, 2)
    s.update()
    assert is_close(s.value, 1)
    for _ in range(10):
        s.update()
        assert is_close(s.value, 1)


def test_const_schedule():
    s = Schedule.constant(50)
    for _ in range(10):
        assert s.value == 50
        s.update()


def test_equality_linear():
    s1 = Schedule.linear(0, 1, 10)
    s2 = Schedule.linear(0, 1, 10)
    s3 = Schedule.linear(0, 1, 5)
    assert s1 == s2
    assert s1 != s3

    s1.update()
    assert s1 != s2
    s2.update()
    assert s1 == s2


def test_equality_exp():
    s1 = Schedule.exp(1, 16, 5)
    s2 = Schedule.exp(1, 16, 5)
    s3 = Schedule.exp(1, 16, 10)
    assert s1 != 5
    assert s1 == 1
    assert s1 == s2
    assert s1 != s3

    s1.update()
    assert s1 != s2
    assert s1 == 2
    assert s2 == 1
    s2.update()
    assert s1 == s2
    assert s2 == 2


def test_multi_schedule():
    s = MultiSchedule(
        {
            0: Schedule.constant(0),
            10: Schedule.linear(0, 1, 10),
            20: Schedule.exp(1, 16, 5),
        }
    )
    expected_values = [0.0] * 10 + [i / 10 for i in range(10)] + [2**i for i in range(5)]
    for i in range(25):
        assert is_close(s.value, expected_values[i])
        s.update()


def test_equality_const():
    s1 = Schedule.constant(50)
    s2 = Schedule.constant(50)
    assert s1 == s2

    s1.update()
    assert s1 == s2
    s2.update()
    assert s1 == s2


def test_inequality_different_schedules():
    s1 = Schedule.linear(1, 2, 10)
    s2 = Schedule.exp(1, 2, 10)
    s3 = Schedule.linear(1, 2, 10)
    assert s1 != s2
    assert not s1 != s3
