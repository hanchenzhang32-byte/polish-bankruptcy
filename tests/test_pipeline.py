"""Tests for loading and scoring, run with: python -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import load_data  # noqa: E402
import train  # noqa: E402

ARFF = """@relation test
% comment line
@attribute Attr1 numeric
@attribute Attr2 numeric
@attribute class {0,1}
@data
0.1,?,0
-0.5,2.0,1
0.1,?,0
"""


def test_read_arff_handles_missing_values_and_renames(tmp_path):
    f = tmp_path / "5year.arff"
    f.write_text(ARFF)
    df = load_data.read_arff(f)
    assert list(df.columns) == ["X1", "X2", "bankrupt"]
    assert df.X2.isna().sum() == 2
    assert df.bankrupt.tolist() == [0, 1, 0]
    assert df.X1.dtype == float


def test_main_removes_duplicate_rows(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem in load_data.HORIZON:
        (raw / f"{stem}.arff").write_text(ARFF)
    out = tmp_path / "companies.csv"
    load_data.main(str(raw), str(out))
    df = pd.read_csv(out)
    assert len(df) == 2 * 5                      # one duplicate dropped in each of the 5 files
    assert sorted(df.years_ahead.unique()) == [1, 2, 3, 4, 5]


def test_altman_score_ranks_weaker_company_as_riskier():
    strong = dict(X3=0.4, X6=0.3, X7=0.2, X8=2.0, X9=1.5)
    weak = dict(X3=-0.2, X6=-0.4, X7=-0.1, X8=0.1, X9=0.5)
    risk = train.altman_z_private(pd.DataFrame([strong, weak]))
    assert risk[1] > risk[0]
    expected_z = 0.717 * 0.4 + 0.847 * 0.3 + 3.107 * 0.2 + 0.420 * 2.0 + 0.998 * 1.5
    assert -risk[0] == pytest.approx(expected_z)


def test_altman_score_fills_missing_values():
    df = pd.DataFrame([dict(X3=0.1, X6=0.1, X7=0.1, X8=1.0, X9=1.0),
                       dict(X3=np.nan, X6=0.1, X7=0.1, X8=1.0, X9=1.0)])
    assert not np.isnan(train.altman_z_private(df)).any()


def test_top10_capture():
    y = np.array([1] * 5 + [0] * 45)
    perfect = np.arange(50)[::-1].astype(float)   # positives get the highest scores
    assert train.top10_capture(y, perfect) == 1.0
    assert train.top10_capture(y, -perfect) == 0.0


def test_x24_is_never_used_as_a_feature():
    assert "X24" in train.LEAKY
