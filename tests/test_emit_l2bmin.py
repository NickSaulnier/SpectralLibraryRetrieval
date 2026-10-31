import pytest

from specret.eval.emit_l2bmin import coarse_group


@pytest.mark.parametrize(
    "name, group",
    [
        ("Calcite+.33Muscov AMX5 Ruby W1R1Ba", "carbonate"),
        ("Muscovite GDS113 Ruby W1R1Bb splib06 3348", "white_mica"),
        ("Kaolin_Smect H89-FR-2 .5Kaol W1R1Bb", "kaolin"),
        ("Alunite0.5+Kaol_KGa-1 AMX3 W1R1Bb", "alunite"),
        ("Nacrite GDS88 s06crj3a=c", "kaolin"),
        ("Chert", "silica"),
        ("Montmorillonite SWy-1 W1R1Hb", "smectite"),
        ("Richterite", "other"),
        (None, "other"),
    ],
)
def test_coarse_group(name, group):
    assert coarse_group(name) == group
