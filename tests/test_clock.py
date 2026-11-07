import json

from driver.clock import RunClock


def make(tmp_path, t):
    return RunClock(tmp_path / "s.json", now=lambda: t[0])


def test_clock_excludes_pauses(tmp_path):
    t = [1000.0]
    c = make(tmp_path, t)
    c.start()
    t[0] += 100
    c.pause("usage_limit")
    t[0] += 50
    c.resume()
    t[0] += 10
    assert c.active_seconds() == 110 and c.wall_seconds() == 160


def test_clock_counts_offline_gap_as_paused(tmp_path):
    t = [0.0]
    c = make(tmp_path, t)
    c.start()
    t[0] = 60
    c.heartbeat()
    t[0] = 4000
    c2 = make(tmp_path, t)
    assert c2.active_seconds() == 60 and c2.wall_seconds() == 4000
    st = json.loads((tmp_path / "s.json").read_text())
    assert st["paused_seconds"] == 3940
    assert st["pauses"][-1]["reason"] == "driver_offline"


def test_short_gap_not_offline(tmp_path):
    t = [0.0]
    c = make(tmp_path, t)
    c.start()
    t[0] = 100
    c2 = make(tmp_path, t)
    assert c2.active_seconds() == 100


def test_pause_resume_noops(tmp_path):
    t = [0.0]
    c = make(tmp_path, t)
    c.start()
    c.resume()  # not paused: no-op
    t[0] = 10
    c.pause("a")
    t[0] = 20
    c.pause("b")  # already paused: no-op, keeps reason a
    assert c.is_paused
    assert c.active_seconds() == 10  # open pause excluded
    t[0] = 30
    c.resume()
    assert c.active_seconds() == 10
    st = json.loads((tmp_path / "s.json").read_text())
    assert st["paused_seconds"] == 20 and len(st["pauses"]) == 1
    assert st["pauses"][0]["reason"] == "a"


def test_paused_state_survives_restart(tmp_path):
    t = [0.0]
    c = make(tmp_path, t)
    c.start()
    t[0] = 50
    c.pause("usage_limit")
    t[0] = 5000
    c2 = make(tmp_path, t)
    assert c2.is_paused and c2.active_seconds() == 50
    t[0] = 5100
    c2.resume()
    t[0] = 5110
    assert c2.active_seconds() == 60


def test_start_is_resume_semantics(tmp_path):
    t = [0.0]
    c = make(tmp_path, t)
    assert not c.started
    c.start()
    assert c.started
    t[0] = 30
    c2 = make(tmp_path, t)
    assert c2.started
    c2.start()
    assert c2.wall_seconds() == 30


def test_unstarted_clock_is_zero(tmp_path):
    t = [5.0]
    c = make(tmp_path, t)
    assert c.active_seconds() == 0 and c.wall_seconds() == 0
