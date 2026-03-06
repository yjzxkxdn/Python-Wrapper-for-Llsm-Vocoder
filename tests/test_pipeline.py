import numpy as np

import pyllsm as llsm


def _build_signal(fs: int, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * seconds)
    t = np.arange(n, dtype=np.float32) / fs
    f0_inst = 190.0 + 25.0 * np.sin(2.0 * np.pi * 1.2 * t)
    phase = np.cumsum(2.0 * np.pi * f0_inst / fs, dtype=np.float32)
    sig = 0.55 * np.sin(phase) + 0.18 * np.sin(2.0 * phase)
    sig += 0.01 * np.random.default_rng(42).standard_normal(n).astype(np.float32)
    return sig.astype(np.float32), f0_inst


def test_python_pitchshift_pipeline_like_libllsm_test() -> None:
    fs = 16000
    nhop = 256
    x, f0_inst = _build_signal(fs, 0.8)

    nfrm = x.size // nhop
    f0 = np.array([f0_inst[i * nhop] for i in range(nfrm)], dtype=np.float32)

    param = llsm.llsm_init(4)
    param.a_f0refine = 0
    param.a_nhop = nhop
    param.a_nhar = 80
    param.a_nhare = 4
    param.a_nnos = 96
    param.a_nosf = fs / 2.0
    param.a_mvf = 6000.0
    param.set_nosbandf([2000.0, 4000.0, 7000.0])

    model = llsm.llsm_layer0_analyze(param, x, fs, f0)

    phase0 = np.zeros(model.nfrm, dtype=np.float32)
    for i in range(model.nfrm):
        frame = model.ptr.frames[i]
        if frame.f0 > 0.0 and frame.sinu.nhar > 0:
            phase0[i] = -frame.sinu.phse[0]
    llsm.llsm_layer0_phaseshift(model, phase0)

    model_lv1 = llsm.llsm_layer1_from_layer0(param, model, 2048, fs)
    faxis = llsm.llsm_uniform_faxis(2048, int(param.a_nosf * 2))
    lip_magn, lip_phse = model_lv1.lip_response()
    ratio = 1.10

    for i in range(model.nfrm):
        frame = model.ptr.frames[i]
        if frame.f0 <= 0.0:
            continue
        orig_f0 = float(frame.f0)
        frame.f0 = frame.f0 * ratio
        nhar = int(frame.sinu.nhar)
        if nhar <= 0:
            continue

        freq = frame.f0 * (np.arange(nhar, dtype=np.float32) + 1.0)
        valid = int(np.count_nonzero(freq <= param.a_mvf))
        frame.sinu.nhar = valid
        nhar = int(frame.sinu.nhar)
        if nhar <= 0:
            continue
        freq = freq[:nhar]

        vt = np.array([model_lv1.ptr.vt_resp_magn[i][j] for j in range(faxis.size)], dtype=np.float32)
        newampl = np.exp(np.interp(freq, faxis, vt)).astype(np.float32)
        newphse = llsm.llsm_harmonic_minphase(newampl)
        lipa = np.interp(freq, faxis, lip_magn).astype(np.float32)
        lipp = np.interp(freq, faxis, lip_phse).astype(np.float32)

        for j in range(nhar):
            frame.sinu.ampl[j] = (
                model_lv1.ptr.vs_har_ampl[i][j] * newampl[j] * lipa[j] * orig_f0 / max(float(frame.f0), 1e-6)
            )
            frame.sinu.phse[j] = model_lv1.ptr.vs_har_phse[i][j] + newphse[j] + lipp[j]

    model_lv1.close()

    phase_recover = np.zeros(model.nfrm, dtype=np.float32)
    for i in range(1, model.nfrm):
        frame = model.ptr.frames[i]
        if frame.f0 > 0.0:
            phase_recover[i] = phase_recover[i - 1] + frame.f0 * nhop / fs * 2.0 * np.pi
    llsm.llsm_layer0_phaseshift(model, phase_recover)

    param.s_fs = fs
    out = llsm.llsm_layer0_synthesize(param, model)
    out_data = out.to_numpy()["y"]

    assert out_data.size > 0
    assert np.isfinite(out_data).all()
    assert float(np.max(np.abs(out_data))) > 1e-6

    out.close()
    model.close()
    param.close()
