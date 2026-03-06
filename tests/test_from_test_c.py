import numpy as np

from .libllsm_test_port import build_synthetic_voice, frame_f0_from_instantaneous, run_test_c_pipeline


def test_port_of_libllsm_test_c_pitchshift_pipeline() -> None:
    fs = 16000
    nhop = 256
    x, f0_inst = build_synthetic_voice(fs=fs, seconds=0.8)
    nfrm = x.size // nhop
    f0 = frame_f0_from_instantaneous(f0_inst, nhop=nhop, nfrm=nfrm)

    y_ref = run_test_c_pipeline(x, fs, f0, ratio=1.0)
    y_shift = run_test_c_pipeline(x, fs, f0, ratio=1.15)

    assert y_ref.size > 0
    assert y_shift.size > 0
    assert np.isfinite(y_ref).all()
    assert np.isfinite(y_shift).all()
    assert np.max(np.abs(y_ref)) > 1e-6
    assert np.max(np.abs(y_shift)) > 1e-6
    assert not np.allclose(y_ref[: min(y_ref.size, y_shift.size)], y_shift[: min(y_ref.size, y_shift.size)], atol=1e-4)

