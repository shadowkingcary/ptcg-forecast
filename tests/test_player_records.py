# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Pooled player-record fallback checks."""
import sys
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.player_records import fit_player_skill_pooled

def _row(pa, pb, aa, ab, y, month):
    return {"player_A": pa, "player_B": pb, "archetype_A": aa, "archetype_B": ab, "y": y, "month": month}

def _sample_train_rows():
    return (
        [_row("alice", "bob", "x", "y", 1, "2024-01")] * 5 + [_row("alice", "bob", "x", "y", 0, "2024-01")] * 2
        + [_row("alice", "bob", "x", "y", 1, "2024-05")] * 3 + [_row("alice", "bob", "x", "y", 0, "2024-05")] * 1
        + [_row("carol", "dave", "z", "w", 1, "2024-03")] * 4
    )

def test_pooled_zero_history_everywhere_is_zero():
    train = _sample_train_rows()
    pooled = fit_player_skill_pooled(train, k_o=10, k_s=10)
    assert pooled("nobody", "x") == 0.0

def test_pooled_new_archetype_falls_back_to_overall_skill_not_half():
    # dave has only ever played archetype "w" and lost all 4 matches recorded against carol's "z";
    # dave's overall record is 0/4. If dave shows up piloting an entirely new archetype "q" he has
    # never played, the pooled model should predict him weaker than a totally blank-slate player,
    # not reset to exactly 0.5.
    train = _sample_train_rows()
    pooled = fit_player_skill_pooled(train, k_o=10, k_s=10)
    assert pooled("dave", "q") < 0.0
    assert pooled("dave", "q") == pooled("dave", "q")  # deterministic
    # compare directly to a truly blank-slate player's skill on the same new archetype
    assert pooled("dave", "q") < pooled("nobody", "q")
