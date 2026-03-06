import numpy as np

import pyllsm as llsm


def test_header_exports_present() -> None:
    required = [
        "llsm_init",
        "llsm_deinit",
        "llsm_create_frame",
        "llsm_create_empty_layer0",
        "llsm_copy_frame",
        "llsm_copy_sinframe",
        "llsm_copy_nosframe",
        "llsm_delete_frame",
        "llsm_layer0_analyze",
        "llsm_layer0_synthesize",
        "llsm_delete_layer0",
        "llsm_delete_output",
        "llsm_uniform_faxis",
        "llsm_liprad",
        "llsm_harmonic_cheaptrick",
        "llsm_harmonic_minphase",
        "llsm_layer1_from_layer0",
        "llsm_delete_layer1",
        "llsm_layer0_phaseshift",
        "llsm_chebyfilt",
        "chebyfilt",
        "llsm_reduce_spectrum_depth",
        "llsm_true_envelope",
        "llsm_warp_freq",
        "llsm_geometric_envelope",
        "llsm_spectrum_from_envelope",
        "llsm_nonuniform_envelope",
        "interp1",
        "interp1u",
        "sincinterp1u",
        "qifft",
        "spec2env",
        "lfmodel_from_rd",
        "lfmodel_spectrum",
        "lfmodel_period",
        "gensins",
    ]
    for name in required:
        assert hasattr(llsm, name), f"missing exported function: {name}"


def test_selected_math_and_envelope_functions() -> None:
    x = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    yi = np.sin(x * np.pi).astype(np.float32)
    xo = np.linspace(0.0, 1.0, 48, dtype=np.float32)

    y_interp = llsm.interp1(x, yi, xo)
    assert y_interp.shape == xo.shape
    assert np.isfinite(y_interp).all()

    w = llsm.llsm_warp_freq(0.0, 8000.0, 48, 1500.0)
    assert w.shape == (48,)
    assert np.all(np.diff(w) >= 0)

    y_filt = llsm.chebyfilt(np.random.default_rng(0).standard_normal(256).astype(np.float32), 0.2, 0.0, "lowpass")
    assert y_filt.shape == (256,)

    model = llsm.lfmodel_from_rd(1.2, 1.0 / 200.0, 1.0)
    freq = np.linspace(100.0, 4000.0, 40, dtype=np.float32)
    g, ph = llsm.lfmodel_spectrum(model, freq, return_phase=True)
    assert g.shape == freq.shape
    assert ph.shape == freq.shape
    assert np.isfinite(g).all()
    assert np.isfinite(ph).all()


