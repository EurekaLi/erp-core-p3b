from __future__ import annotations

import numpy as np

from erp_core_p3b.subject_review import temporal_rejection_profile


def test_temporal_rejection_profile_keeps_last_event_in_final_bin() -> None:
    onset = np.array([0.0, 1.0, 2.0, 3.0])
    rejected = np.array([False, True, False, True])

    result = temporal_rejection_profile(onset, rejected, bins=2)

    assert result["epoch_count"].tolist() == [2, 2]
    assert result["rejected_count"].tolist() == [1, 1]
    assert result["rejection_fraction"].tolist() == [0.5, 0.5]
