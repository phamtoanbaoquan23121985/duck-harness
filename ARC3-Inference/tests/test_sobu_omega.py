from inference.agent.sobu_omega import SobuOmegaController

def test_novel_action_is_not_vetoed():
    c = SobuOmegaController()
    u = c.assess("s0", "LEFT", decision_information=1.0)
    assert not u.hard_veto

def test_repeated_no_progress_becomes_veto_candidate():
    c = SobuOmegaController()
    c.observe("s0", "LEFT", goal_progress=False, board_changed=False)
    c.observe("s0", "LEFT", goal_progress=False, board_changed=False)
    assert c.assess("s0", "LEFT").hard_veto

def test_board_change_alone_is_not_goal_success():
    c = SobuOmegaController()
    c.observe("s0", "LEFT", goal_progress=False, board_changed=True)
    assert c.stats[("s0", "LEFT")].p_success() < 0.5

def test_goal_progress_improves_action_utility():
    c = SobuOmegaController()
    before = c.assess("s0", "RIGHT").value
    c.observe("s0", "RIGHT", goal_progress=True, board_changed=True)
    after = c.assess("s0", "RIGHT").value
    assert after > before

def test_choose_avoids_empirically_dead_action():
    c = SobuOmegaController()
    for _ in range(2):
        c.observe("s0", "LEFT", goal_progress=False, board_changed=False)
    assert c.choose("s0", ["LEFT", "RIGHT"], decision_information={"RIGHT": 1.0}) == "RIGHT"


def test_information_value_decays_after_repeated_probe():
    c = SobuOmegaController()
    first = c.assess("s0", "UP", decision_information=1.0).information
    c.observe("s0", "UP", goal_progress=False, board_changed=False)
    later = c.assess("s0", "UP", decision_information=1.0).information
    assert later < first

def test_fail_open_when_all_candidates_are_vetoed():
    c = SobuOmegaController()
    for action in ("LEFT", "RIGHT"):
        for _ in range(2):
            c.observe("s0", action, goal_progress=False, board_changed=False)
    assert c.choose("s0", ["LEFT", "RIGHT"]) == "LEFT"
