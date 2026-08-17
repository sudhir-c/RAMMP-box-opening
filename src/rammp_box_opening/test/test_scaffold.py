from rammp_box_opening.constants import HOME, JOINTS, WRIST_FLAT_XYZW


def test_constants_shape():
    assert len(HOME) == 7
    assert JOINTS[0] == "joint_1" and JOINTS[-1] == "joint_7"
    assert len(WRIST_FLAT_XYZW) == 4
