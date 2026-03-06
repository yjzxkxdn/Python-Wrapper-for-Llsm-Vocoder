from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import pyllsm as llsm


def build_synthetic_voice(fs: int = 16000, seconds: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * seconds)
    t = np.arange(n, dtype=np.float32) / fs
    f0_inst = 185.0 + 18.0 * np.sin(2.0 * np.pi * 1.6 * t)
    phase = np.cumsum(2.0 * np.pi * f0_inst / fs, dtype=np.float32)
    x = 0.6 * np.sin(phase) + 0.2 * np.sin(2.0 * phase + 0.6) + 0.1 * np.sin(3.0 * phase)
    x += 0.01 * np.random.default_rng(1234).standard_normal(n).astype(np.float32)
    return x.astype(np.float32), f0_inst.astype(np.float32)


def frame_f0_from_instantaneous(f0_inst: np.ndarray, nhop: int, nfrm: int) -> np.ndarray:
    idx = np.clip(np.arange(nfrm, dtype=np.int64) * nhop, 0, f0_inst.size - 1)
    return f0_inst[idx].astype(np.float32)


def default_test_params(fs: int, nhop: int = 256) -> llsm.LlsmParameters:
    param = llsm.llsm_init(4)
    param.a_nhop = nhop
    param.a_nhar = 400
    param.a_nhare = 5
    param.a_nnos = 192
    param.a_nosf = fs / 2.0
    param.a_mvf = min(12000.0, fs / 2.0 - 1.0)
    param.set_nosbandf([2000.0, 5000.0, 9000.0])
    param.a_f0refine = 0
    return param


def _wrap_phase(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def apply_relative_phase_shift(model: llsm.Layer0Model) -> None:
    phase0 = np.zeros(model.nfrm, dtype=np.float32)
    for i in range(model.nfrm):
        frm = model.ptr.frames[i]
        if frm.f0 > 0.0 and frm.sinu.nhar > 0:
            phase0[i] = -frm.sinu.phse[0]
    llsm.llsm_layer0_phaseshift(model, phase0)


def recover_phase_progression(model: llsm.Layer0Model, fs: int, nhop: int) -> None:
    phase0 = np.zeros(model.nfrm, dtype=np.float32)
    for i in range(1, model.nfrm):
        frm = model.ptr.frames[i]
        if frm.f0 > 0.0:
            phase0[i] = phase0[i - 1] + frm.f0 * nhop / fs * 2.0 * np.pi
    llsm.llsm_layer0_phaseshift(model, phase0)


def _pitch_shift_modify_layer0(
    model: llsm.Layer0Model, model_lv1: llsm.Layer1Model, param: llsm.LlsmParameters, ratio: float
) -> None:
    nfft = model_lv1.nfft
    faxis = llsm.llsm_uniform_faxis(nfft, int(param.a_nosf * 2.0))
    lip_magn, lip_phse = model_lv1.lip_response()
    ns = nfft // 2 + 1
    for i in range(model.nfrm):
        frm = model.ptr.frames[i]
        if frm.f0 <= 0.0:
            continue
        nhar = int(frm.sinu.nhar)
        if nhar <= 0:
            continue

        orig_f0 = float(frm.f0)
        frm.f0 = frm.f0 * float(ratio)

        freq = frm.f0 * (np.arange(nhar, dtype=np.float32) + 1.0)
        valid = int(np.count_nonzero(freq <= float(param.a_mvf)))
        frm.sinu.nhar = valid
        nhar = int(frm.sinu.nhar)
        if nhar <= 0:
            continue

        freq = freq[:nhar]
        vt = np.array([model_lv1.ptr.vt_resp_magn[i][j] for j in range(ns)], dtype=np.float32)
        newampl = np.exp(np.interp(freq, faxis, vt)).astype(np.float32)
        newphse = llsm.llsm_harmonic_minphase(newampl)
        lipa = np.interp(freq, faxis, lip_magn).astype(np.float32)
        lipp = np.interp(freq, faxis, lip_phse).astype(np.float32)

        for j in range(nhar):
            frm.sinu.ampl[j] = model_lv1.ptr.vs_har_ampl[i][j] * newampl[j] * lipa[j] * orig_f0 / max(frm.f0, 1e-7)
            frm.sinu.phse[j] = model_lv1.ptr.vs_har_phse[i][j] + newphse[j] + lipp[j]


def run_test_c_pipeline(x: np.ndarray, fs: int, f0: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    nhop = 256
    param = default_test_params(fs, nhop=nhop)
    model = llsm.llsm_layer0_analyze(param, x, fs, f0)
    apply_relative_phase_shift(model)
    model_lv1 = llsm.llsm_layer1_from_layer0(param, model, 2048, fs)
    _pitch_shift_modify_layer0(model, model_lv1, param, ratio)
    model_lv1.close()
    recover_phase_progression(model, fs, nhop)
    param.s_fs = fs
    out = llsm.llsm_layer0_synthesize(param, model)
    y = out.to_numpy()["y"]
    out.close()
    model.close()
    param.close()
    return y


_RD_MIN = 0.02
_RD_MAX = 3.02
_RD_STEP = 0.04
_MT_WARP_AXIS = np.array(
    [
        0.020000000,
        0.177460000,
        0.437530011,
        0.666920006,
        0.847289979,
        0.983470023,
        1.080870032,
        1.173920035,
        1.256489992,
        1.329079986,
        1.401729941,
        1.474879980,
        1.540629983,
        1.597769976,
        1.653419971,
        1.712180018,
        1.771289945,
        1.824579954,
        1.869390011,
        1.909080029,
        1.949130058,
        1.992830038,
        2.039900064,
        2.087340117,
        2.131099939,
        2.168190002,
        2.198319912,
        2.223710060,
        2.247450113,
        2.272160053,
        2.299380064,
        2.329639912,
        2.362709999,
        2.397850037,
        2.433990002,
        2.469929934,
        2.504450083,
        2.536520004,
        2.565459967,
        2.590960026,
        2.613120079,
        2.632359982,
        2.649230003,
        2.664340019,
        2.678220034,
        2.691309929,
        2.703969955,
        2.716409922,
        2.728810072,
        2.741260052,
        2.753809929,
        2.766479969,
        2.779299974,
        2.792259932,
        2.805360079,
        2.818589926,
        2.831929922,
        2.845410109,
        2.858979940,
        2.872669935,
        2.886450052,
        2.900320053,
        2.914249897,
        2.928200006,
        2.942130089,
        2.955970049,
        2.969599962,
        2.981810093,
        2.985869884,
        2.990780115,
        2.996279955,
        3.002049923,
        3.007740021,
        3.012909889,
        3.017149925,
        3.019999981,
    ],
    dtype=np.float32,
)


def _mt_to_rd(rd_in: float, mt: float) -> float:
    rd = float(np.clip(rd_in, _RD_MIN, _RD_MAX))
    warp = np.interp(rd, _RD_MIN + _RD_STEP * np.arange(_MT_WARP_AXIS.size, dtype=np.float32), _MT_WARP_AXIS)
    rd_out = np.interp(warp - 0.01 * float(mt), _MT_WARP_AXIS, _RD_MIN + _RD_STEP * np.arange(_MT_WARP_AXIS.size))
    return float(np.clip(rd_out, _RD_MIN, _RD_MAX))


def _apply_mt_on_layer1(model: llsm.Layer0Model, model_lv1: llsm.Layer1Model, mt: float) -> None:
    for i in range(model.nfrm):
        frm = model.ptr.frames[i]
        if frm.f0 <= 0.0 or frm.sinu.nhar <= 0:
            continue
        nhar = int(frm.sinu.nhar)
        rd_old = float(model_lv1.ptr.vs_rd[i])
        rd_new = _mt_to_rd(rd_old, mt)
        if abs(rd_new - rd_old) < 1e-7:
            continue
        freq = frm.f0 * (np.arange(nhar, dtype=np.float32) + 1.0)
        g_old = llsm.lfmodel_spectrum(llsm.lfmodel_from_rd(rd_old, 1.0 / frm.f0, 1.0), freq)
        g_new = llsm.lfmodel_spectrum(llsm.lfmodel_from_rd(rd_new, 1.0 / frm.f0, 1.0), freq)
        d0_old = max(float(g_old[0]), 1e-12)
        d0_new = max(float(g_new[0]), 1e-12)
        g_old = g_old.copy()
        g_new = g_new.copy()
        g_old[0] = 1.0
        g_new[0] = 1.0
        harm = (np.arange(1, nhar, dtype=np.float32) + 1.0)
        g_old[1:] = g_old[1:] / (harm * d0_old)
        g_new[1:] = g_new[1:] / (harm * d0_new)
        ratio = g_new / np.maximum(g_old, 1e-12)
        for j in range(nhar):
            model_lv1.ptr.vs_har_ampl[i][j] *= float(ratio[j])
        model_lv1.ptr.vs_rd[i] = rd_new


def run_test_mt_pipeline(x: np.ndarray, fs: int, f0: np.ndarray, mt: float) -> np.ndarray:
    nhop = 256
    param = default_test_params(fs, nhop=nhop)
    model = llsm.llsm_layer0_analyze(param, x, fs, f0)
    apply_relative_phase_shift(model)
    model_lv1 = llsm.llsm_layer1_from_layer0(param, model, 2048, fs)
    _apply_mt_on_layer1(model, model_lv1, mt)
    _pitch_shift_modify_layer0(model, model_lv1, param, ratio=1.0)
    model_lv1.close()
    recover_phase_progression(model, fs, nhop)
    param.s_fs = fs
    out = llsm.llsm_layer0_synthesize(param, model)
    y = out.to_numpy()["y"]
    out.close()
    model.close()
    param.close()
    return y



def _reconstruct_harmonic_from_pack_frame(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    f0 = float(frame["f0"])
    nhar = int(frame["nhar"])
    if f0 <= 0.0 or nhar <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    cols = int(frame["cols"])
    arr4 = np.asarray(frame["arr4"], dtype=np.float32).reshape(nhar, cols)
    vt_har_ampl = np.exp(np.clip(arr4[:, cols - 1], -100.0, 100.0)).astype(np.float32)
    freq = f0 * (np.arange(nhar, dtype=np.float32) + 1.0)

    g = llsm.lfmodel_spectrum(llsm.lfmodel_from_rd(float(frame["rd"]), 1.0 / f0, 1.0), freq)
    g = g.astype(np.float32)
    if nhar > 1:
        g[1:] = g[1:] / ((np.arange(1, nhar, dtype=np.float32) + 1.0) * max(float(g[0]), 1e-12))
    g[0] = 1.0

    vt_phse = llsm.llsm_harmonic_minphase(vt_har_ampl)
    lip_magn, lip_phse = llsm.llsm_liprad(freq, radius=1.5, return_phase=True)
    src_phse = np.asarray(frame["vs_har_phse"], dtype=np.float32)
    if src_phse.size != nhar:
        src_phse = np.pad(src_phse, (0, max(0, nhar - src_phse.size)), mode="constant")[:nhar]
    ampl = (vt_har_ampl * g * lip_magn).astype(np.float32)
    phse = (vt_phse + lip_phse + src_phse).astype(np.float32)
    return ampl, phse


def pack_to_layer0(pack: dict[str, Any]) -> llsm.Layer0Model:
    ffi = llsm.ffi
    conf_src = pack["conf"]
    frames = pack["frames"]
    conf = ffi.new("llsm_conf *")
    conf[0].nfrm = int(len(frames))
    conf[0].nhop = int(conf_src["nhop"])
    conf[0].nhar = int(conf_src["nhar"])
    conf[0].nhare = int(conf_src["nhare"])
    conf[0].nnos = int(conf_src["nnos"])
    conf[0].noswarp = float(conf_src["noswarp"])
    conf[0].mvf = float(conf_src["mvf"])
    conf[0].nosf = float(conf_src["nosf"])
    conf[0].thop = float(conf_src["thop"])
    conf[0].nnosband = int(conf_src["nnosband"])
    nosband = np.asarray(conf_src["nosbandf"], dtype=np.float32).reshape(-1)
    c_nosband = ffi.new("FP_TYPE[]", nosband.tolist()) if nosband.size > 0 else ffi.NULL
    conf[0].nosbandf = c_nosband

    model = llsm.llsm_create_empty_layer0(conf[0])
    for i, mf in enumerate(frames):
        noise = mf["noise"]
        fr = llsm.llsm_create_frame(int(mf["nhar"]), conf[0].nhare, int(noise["nnos"]), int(noise["nchannel"]))
        fr.f0 = float(mf["f0"])

        spec = np.asarray(noise["spec"], dtype=np.float32)
        emin = np.asarray(noise["emin"], dtype=np.float32)
        nhar_ch = np.asarray(noise["nhar"], dtype=np.int32)
        for j in range(int(noise["nnos"])):
            fr.noise.spec[j] = float(spec[j])
        for b in range(int(noise["nchannel"])):
            fr.noise.emin[b] = float(emin[b])
            h = int(nhar_ch[b])
            if h > fr.noise.eenv[b].nhar:
                raise ValueError(f"noise harmonic count overflow at frame={i}, band={b}, h={h}")
            for k in range(h):
                fr.noise.eenv[b].ampl[k] = float(noise["ampl"][b][k])
                fr.noise.eenv[b].phse[k] = float(noise["phse"][b][k])

        if fr.f0 > 0.0 and int(mf["nhar"]) > 0:
            ampl, phse = _reconstruct_harmonic_from_pack_frame(mf)
            for k in range(int(mf["nhar"])):
                fr.sinu.ampl[k] = float(ampl[k])
                fr.sinu.phse[k] = float(phse[k])
            ph0 = float(phse[0]) if phse.size > 0 else 0.0
            for b in range(int(noise["nchannel"])):
                h = int(nhar_ch[b])
                for k in range(h):
                    fr.noise.eenv[b].phse[k] = fr.noise.eenv[b].phse[k] + (k + 1.0) * ph0

        model.ptr.frames[i] = fr
    return model


def synthesize_pack(pack: dict[str, Any], out_fs: int = 44100) -> np.ndarray:
    model = pack_to_layer0(pack)
    nfrm = model.nfrm
    nhop = model.nhop
    phase0 = np.zeros(nfrm, dtype=np.float32)
    acc = 0.0
    for i in range(1, nfrm):
        f0_i = float(model.ptr.frames[i].f0)
        f0_ip1 = float(model.ptr.frames[i + 1].f0) if (i + 1 < nfrm) else 0.0
        fmid = 0.0
        if f0_i > 0.0:
            fmid = 0.5 * (f0_i + f0_ip1) if (i + 1 < nfrm and f0_ip1 > 0.0) else f0_i
        elif (i + 1 < nfrm and f0_ip1 > 0.0):
            fmid = f0_ip1
        acc += fmid * nhop / out_fs * 2.0 * np.pi
        phase0[i] = acc
    llsm.llsm_layer0_phaseshift(model, phase0)

    param = llsm.llsm_init(int(pack["conf"]["nnosband"]))
    param.s_fs = int(out_fs)
    out = llsm.llsm_layer0_synthesize(param, model)
    y = out.to_numpy()["y"]
    out.close()
    model.close()
    param.close()
    return y


def analyze_array_to_pack(x: np.ndarray, fs: int, f0: np.ndarray) -> dict[str, Any]:
    nhop = 256
    param = llsm.llsm_init(4)
    param.a_nhop = nhop
    param.a_nhar = 80
    param.a_nhare = 4
    param.a_nnos = 96
    param.a_mvf = min(12000.0, fs / 2.0 - 1.0)
    param.a_nosf = min(15000.0, fs / 2.0 - 1.0)
    param.a_tfft = 0.04
    param.a_f0refine = 0
    param.set_nosbandf([2000.0, 5000.0, 9000.0])

    m0 = llsm.llsm_layer0_analyze(param, x, fs, f0)
    apply_relative_phase_shift(m0)
    m1 = llsm.llsm_layer1_from_layer0(param, m0, 2048, fs)
    faxis = llsm.llsm_uniform_faxis(2048, fs)
    ns = 2048 // 2 + 1

    conf = {
        "nfrm": m0.nfrm,
        "nhop": int(m0.ptr.conf.nhop),
        "nhar": int(m0.ptr.conf.nhar),
        "nhare": int(m0.ptr.conf.nhare),
        "nnos": int(m0.ptr.conf.nnos),
        "noswarp": float(m0.ptr.conf.noswarp),
        "mvf": float(m0.ptr.conf.mvf),
        "nosf": float(m0.ptr.conf.nosf),
        "thop": float(m0.ptr.conf.thop),
        "nnosband": int(m0.ptr.conf.nnosband),
        "nosbandf": np.array(
            [m0.ptr.conf.nosbandf[i] for i in range(max(0, int(m0.ptr.conf.nnosband) - 1))], dtype=np.float32
        ),
    }

    frames: list[dict[str, Any]] = []
    for i in range(m0.nfrm):
        fr = m0.ptr.frames[i]
        nhar = int(fr.sinu.nhar)
        mf: dict[str, Any] = {
            "f0": float(fr.f0),
            "nhar": nhar,
            "cols": 3,
            "rd": float(m1.ptr.vs_rd[i]) if m1.ptr.vs_rd != llsm.ffi.NULL else 0.0,
            "step": float(fr.f0 / 3.0) if fr.f0 > 0.0 else 0.0,
            "vs_har_phse": np.zeros(max(0, nhar), dtype=np.float32),
            "arr4": np.zeros(max(0, nhar * 3), dtype=np.float32),
        }

        if nhar > 0:
            src = np.array([m1.ptr.vs_har_phse[i][j] for j in range(nhar)], dtype=np.float32)
            ph0 = float(src[0])
            rel = _wrap_phase(src - (np.arange(nhar, dtype=np.float32) + 1.0) * ph0)
            mf["vs_har_phse"] = rel
            arr_n = nhar * mf["cols"]
            xq = (np.arange(arr_n, dtype=np.float32) + 1.0) * float(mf["step"])
            vt = np.array([m1.ptr.vt_resp_magn[i][j] for j in range(ns)], dtype=np.float32)
            mf["arr4"] = llsm.interp1(faxis, vt, xq)

        noise = fr.noise
        nchannel = int(noise.nchannel)
        nhar_ch = np.array([int(noise.eenv[b].nhar) for b in range(nchannel)], dtype=np.int32)
        noise_dict: dict[str, Any] = {
            "nchannel": nchannel,
            "nnos": int(noise.nnos),
            "nhar": nhar_ch,
            "spec": np.array([noise.spec[k] for k in range(int(noise.nnos))], dtype=np.float32),
            "emin": np.array([noise.emin[b] for b in range(nchannel)], dtype=np.float32),
            "ampl": [],
            "phse": [],
        }
        ph0_model = 0.0
        if nhar > 0:
            _, phse = _reconstruct_harmonic_from_pack_frame(mf)
            ph0_model = float(phse[0]) if phse.size > 0 else 0.0
        for b in range(nchannel):
            h = int(nhar_ch[b])
            ampl = np.array([noise.eenv[b].ampl[k] for k in range(h)], dtype=np.float32)
            phse = np.array([noise.eenv[b].phse[k] - (k + 1.0) * ph0_model for k in range(h)], dtype=np.float32)
            noise_dict["ampl"].append(ampl)
            noise_dict["phse"].append(phse)
        mf["noise"] = noise_dict
        frames.append(mf)

    pack = {"version": 524, "duration": float(m0.ptr.conf.nfrm * m0.ptr.conf.thop), "conf": conf, "frames": frames}

    m1.close()
    m0.close()
    param.close()
    return pack


def default_llsm_fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "templlsm.llsm"
