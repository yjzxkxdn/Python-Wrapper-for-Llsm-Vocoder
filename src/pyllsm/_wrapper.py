from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ._pyllsm_cffi import ffi, lib


def _as_f32_array(data: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return np.ascontiguousarray(arr)


def _as_i32_array(data: Iterable[int], name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.int32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return np.ascontiguousarray(arr)


def _char_array(value: str | bytes) -> Any:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = str(value).encode("ascii")
    return ffi.new("char[]", raw)


def _copy_f32_ptr(ptr: Any, n: int, free_after: bool = True) -> np.ndarray:
    if ptr == ffi.NULL:
        raise MemoryError("native call returned NULL")
    out = np.frombuffer(ffi.buffer(ptr, n * ffi.sizeof("FP_TYPE")), dtype=np.float32).copy()
    if free_after:
        lib.llsm_py_free(ptr)
    return out


def _unwrap_param(param: "LlsmParameters | Any") -> Any:
    if isinstance(param, LlsmParameters):
        return param.as_c()
    return param


def _unwrap_layer0(model: "Layer0Model | Any") -> Any:
    if isinstance(model, Layer0Model):
        return model.ptr
    return model


def _unwrap_layer1(model: "Layer1Model | Any") -> Any:
    if isinstance(model, Layer1Model):
        return model.ptr
    return model


def _unwrap_output(out: "LlsmOutput | Any") -> Any:
    if isinstance(out, LlsmOutput):
        return out.ptr
    return out


class LlsmParameters:
    _SCALAR_FIELDS = {
        "a_nhop",
        "a_nhar",
        "a_nhare",
        "a_nnos",
        "a_f0refine",
        "a_wsize",
        "a_noswarp",
        "a_tfft",
        "a_mvf",
        "a_nosf",
        "a_nnosband",
        "s_fs",
        "s_noiseonly",
        "s_n0",
        "s_n1",
    }

    def __init__(self, nnosband: int = 4):
        self._ptr = ffi.new("llsm_parameters *")
        self._ptr[0] = lib.llsm_init(int(nnosband))
        self._closed = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        if name in self._SCALAR_FIELDS:
            self._ensure_open()
            return getattr(self._ptr[0], name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_ptr", "_closed"}:
            object.__setattr__(self, name, value)
            return
        if name in self._SCALAR_FIELDS:
            self._ensure_open()
            setattr(self._ptr[0], name, value)
            return
        object.__setattr__(self, name, value)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LlsmParameters is closed")

    def as_c(self) -> Any:
        self._ensure_open()
        return self._ptr[0]

    def close(self) -> None:
        if not self._closed:
            lib.llsm_deinit(self._ptr[0])
            self._closed = True

    def get_nosbandf(self) -> np.ndarray:
        self._ensure_open()
        n = int(self._ptr[0].a_nnosband) - 1
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.array([self._ptr[0].a_nosbandf[i] for i in range(n)], dtype=np.float32)

    def set_nosbandf(self, values: Iterable[float]) -> None:
        self._ensure_open()
        arr = _as_f32_array(values, "values")
        n = int(self._ptr[0].a_nnosband) - 1
        if arr.size != n:
            raise ValueError(f"nosbandf size mismatch: expected {n}, got {arr.size}")
        for i in range(n):
            self._ptr[0].a_nosbandf[i] = float(arr[i])


class Layer0Model:
    def __init__(self, ptr: Any, owns_memory: bool = True):
        if ptr == ffi.NULL:
            raise MemoryError("llsm_layer0 pointer is NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if not self._closed and self._owns_memory:
            lib.llsm_delete_layer0(self.ptr)
            self._closed = True

    @property
    def nfrm(self) -> int:
        return int(self.ptr.conf.nfrm)

    @property
    def nhop(self) -> int:
        return int(self.ptr.conf.nhop)

    def frame_f0(self) -> np.ndarray:
        return np.array([self.ptr.frames[i].f0 for i in range(self.nfrm)], dtype=np.float32)

    def frame_nhar(self, frame_idx: int) -> int:
        return int(self.ptr.frames[frame_idx].sinu.nhar)

    def set_frame_nhar(self, frame_idx: int, nhar: int) -> None:
        cur = int(self.ptr.frames[frame_idx].sinu.nhar)
        if nhar > cur:
            raise ValueError(f"cannot increase nhar in-place: current={cur}, requested={nhar}")
        self.ptr.frames[frame_idx].sinu.nhar = int(nhar)

    def phaseshift(self, phaseshift: Iterable[float]) -> None:
        shift = _as_f32_array(phaseshift, "phaseshift")
        if shift.size != self.nfrm:
            raise ValueError(f"phaseshift size mismatch: expected {self.nfrm}, got {shift.size}")
        shift_ptr = ffi.from_buffer("FP_TYPE[]", shift)
        lib.llsm_layer0_phaseshift(self.ptr, shift_ptr)


class Layer1Model:
    def __init__(self, ptr: Any, owns_memory: bool = True):
        if ptr == ffi.NULL:
            raise MemoryError("llsm_layer1 pointer is NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if not self._closed and self._owns_memory:
            lib.llsm_delete_layer1(self.ptr)
            self._closed = True

    @property
    def nfrm(self) -> int:
        return int(self.ptr.nfrm)

    @property
    def nfft(self) -> int:
        return int(self.ptr.nfft)

    @property
    def ns(self) -> int:
        return self.nfft // 2 + 1

    def lip_response(self) -> tuple[np.ndarray, np.ndarray]:
        ns = self.ns
        magn = np.array([self.ptr.lip_resp_magn[i] for i in range(ns)], dtype=np.float32)
        phse = np.array([self.ptr.lip_resp_phse[i] for i in range(ns)], dtype=np.float32)
        return magn, phse


class LlsmOutput:
    def __init__(self, ptr: Any, owns_memory: bool = True):
        if ptr == ffi.NULL:
            raise MemoryError("llsm_output pointer is NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if not self._closed and self._owns_memory:
            lib.llsm_delete_output(self.ptr)
            self._closed = True

    def to_numpy(self) -> dict[str, np.ndarray | float | int]:
        ny = int(self.ptr.ny)
        y = np.array([self.ptr.y[i] for i in range(ny)], dtype=np.float32)
        y_sin = np.array([self.ptr.y_sin[i] for i in range(ny)], dtype=np.float32)
        y_nos = np.array([self.ptr.y_nos[i] for i in range(ny)], dtype=np.float32)
        return {"y": y, "y_sin": y_sin, "y_nos": y_nos, "ny": ny, "fs": float(self.ptr.fs)}


def llsm_init(nnosband: int = 4) -> LlsmParameters:
    return LlsmParameters(nnosband=nnosband)


def llsm_deinit(param: LlsmParameters | Any) -> None:
    if isinstance(param, LlsmParameters):
        param.close()
    else:
        lib.llsm_deinit(param)


def llsm_create_frame(nhar: int, nhare: int, nnos: int, nchannel: int) -> Any:
    return lib.llsm_create_frame(nhar, nhare, nnos, nchannel)


def llsm_create_empty_layer0(conf: Any) -> Layer0Model:
    return Layer0Model(lib.llsm_create_empty_layer0(conf))


def llsm_copy_frame(dst: Any, src: Any) -> None:
    lib.llsm_copy_frame(dst, src)


def llsm_copy_sinframe(dst: Any, src: Any) -> None:
    lib.llsm_copy_sinframe(dst, src)


def llsm_copy_nosframe(dst: Any, src: Any) -> None:
    lib.llsm_copy_nosframe(dst, src)


def llsm_delete_frame(frame: Any) -> None:
    lib.llsm_delete_frame(frame)


def llsm_delete_layer0(model: Layer0Model | Any) -> None:
    if isinstance(model, Layer0Model):
        model.close()
    else:
        lib.llsm_delete_layer0(model)


def llsm_delete_layer1(model: Layer1Model | Any) -> None:
    if isinstance(model, Layer1Model):
        model.close()
    else:
        lib.llsm_delete_layer1(model)


def llsm_delete_output(out: LlsmOutput | Any) -> None:
    if isinstance(out, LlsmOutput):
        out.close()
    else:
        lib.llsm_delete_output(out)


def llsm_layer0_analyze(
    param: LlsmParameters | Any,
    x: Iterable[float],
    fs: int,
    f0: Iterable[float],
    *,
    return_xap: bool = False,
) -> Layer0Model | tuple[Layer0Model, np.ndarray]:
    cparam = _unwrap_param(param)
    x_arr = _as_f32_array(x, "x")
    f0_arr = _as_f32_array(f0, "f0")

    x_ptr = ffi.from_buffer("FP_TYPE[]", x_arr)
    f0_ptr = ffi.from_buffer("FP_TYPE[]", f0_arr)
    xap_ptr = ffi.new("FP_TYPE **") if return_xap else ffi.NULL

    model_ptr = lib.llsm_layer0_analyze(cparam, x_ptr, x_arr.size, int(fs), f0_ptr, f0_arr.size, xap_ptr)
    model = Layer0Model(model_ptr)
    if not return_xap:
        return model
    xap = _copy_f32_ptr(xap_ptr[0], x_arr.size, free_after=True)
    return model, xap


def llsm_layer0_synthesize(param: LlsmParameters | Any, model: Layer0Model | Any) -> LlsmOutput:
    cparam = _unwrap_param(param)
    model_ptr = _unwrap_layer0(model)
    return LlsmOutput(lib.llsm_layer0_synthesize(cparam, model_ptr))


def llsm_layer1_from_layer0(
    param: LlsmParameters | Any, model: Layer0Model | Any, nfft: int, fs: int
) -> Layer1Model:
    cparam = _unwrap_param(param)
    model_ptr = _unwrap_layer0(model)
    return Layer1Model(lib.llsm_layer1_from_layer0(cparam, model_ptr, int(nfft), int(fs)))


def llsm_layer0_phaseshift(model: Layer0Model | Any, phaseshift: Iterable[float]) -> None:
    model_ptr = _unwrap_layer0(model)
    shift = _as_f32_array(phaseshift, "phaseshift")
    shift_ptr = ffi.from_buffer("FP_TYPE[]", shift)
    lib.llsm_layer0_phaseshift(model_ptr, shift_ptr)


def llsm_uniform_faxis(nfft: int, fs: int) -> np.ndarray:
    n = int(nfft) // 2 + 1
    out_ptr = lib.llsm_uniform_faxis(int(nfft), int(fs))
    return _copy_f32_ptr(out_ptr, n)


def llsm_liprad(
    freq: Iterable[float], radius: float = 1.5, *, return_phase: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    freq_arr = _as_f32_array(freq, "freq")
    freq_ptr = ffi.from_buffer("FP_TYPE[]", freq_arr)
    ph_ptr = ffi.new("FP_TYPE[]", freq_arr.size) if return_phase else ffi.NULL
    magn_ptr = lib.llsm_liprad(freq_ptr, freq_arr.size, float(radius), ph_ptr)
    magn = _copy_f32_ptr(magn_ptr, freq_arr.size)
    if not return_phase:
        return magn
    phse = np.frombuffer(ffi.buffer(ph_ptr, freq_arr.size * ffi.sizeof("FP_TYPE")), dtype=np.float32).copy()
    return magn, phse


def llsm_harmonic_cheaptrick(ampl: Iterable[float], f0: float, nfft: int, fs: float) -> np.ndarray:
    ampl_arr = _as_f32_array(ampl, "ampl")
    ampl_ptr = ffi.from_buffer("FP_TYPE[]", ampl_arr)
    out_ptr = lib.llsm_harmonic_cheaptrick(ampl_ptr, ampl_arr.size, float(f0), int(nfft), float(fs))
    return _copy_f32_ptr(out_ptr, int(nfft) // 2 + 1)


def llsm_harmonic_minphase(ampl: Iterable[float]) -> np.ndarray:
    ampl_arr = _as_f32_array(ampl, "ampl")
    ampl_ptr = ffi.from_buffer("FP_TYPE[]", ampl_arr)
    out_ptr = lib.llsm_harmonic_minphase(ampl_ptr, ampl_arr.size)
    return _copy_f32_ptr(out_ptr, ampl_arr.size)


def llsm_chebyfilt(
    x: Iterable[float], cutoff1: float, cutoff2: float, filter_type: str = "lowpass"
) -> np.ndarray:
    x_arr = _as_f32_array(x, "x")
    x_ptr = ffi.from_buffer("FP_TYPE[]", x_arr)
    out_ptr = lib.llsm_chebyfilt(x_ptr, x_arr.size, float(cutoff1), float(cutoff2), _char_array(filter_type))
    return _copy_f32_ptr(out_ptr, x_arr.size)


def chebyfilt(
    x: Iterable[float], cutoff1: float, cutoff2: float, filter_type: str = "lowpass"
) -> np.ndarray:
    x_arr = _as_f32_array(x, "x")
    x_ptr = ffi.from_buffer("FP_TYPE[]", x_arr)
    out_ptr = lib.llsm_py_chebyfilt(x_ptr, x_arr.size, float(cutoff1), float(cutoff2), _char_array(filter_type))
    return _copy_f32_ptr(out_ptr, x_arr.size)


def llsm_warp_freq(fmin: float, fmax: float, n: int, warp_const: float) -> np.ndarray:
    out_ptr = lib.llsm_warp_freq(float(fmin), float(fmax), int(n), float(warp_const))
    return _copy_f32_ptr(out_ptr, int(n))


def llsm_geometric_envelope(spectrum: Iterable[float], nfft: int, fs: int, freq: Iterable[float]) -> np.ndarray:
    spec_arr = _as_f32_array(spectrum, "spectrum")
    freq_arr = _as_f32_array(freq, "freq")
    spec_ptr = ffi.from_buffer("FP_TYPE[]", spec_arr)
    freq_ptr = ffi.from_buffer("FP_TYPE[]", freq_arr)
    out_ptr = lib.llsm_geometric_envelope(spec_ptr, int(nfft), int(fs), freq_ptr, freq_arr.size)
    return _copy_f32_ptr(out_ptr, freq_arr.size)


def llsm_spectrum_from_envelope(
    freq: Iterable[float], magn: Iterable[float], nfft: int, fs: int
) -> np.ndarray:
    freq_arr = _as_f32_array(freq, "freq")
    magn_arr = _as_f32_array(magn, "magn")
    if freq_arr.size != magn_arr.size:
        raise ValueError("freq and magn size mismatch")
    freq_ptr = ffi.from_buffer("FP_TYPE[]", freq_arr)
    magn_ptr = ffi.from_buffer("FP_TYPE[]", magn_arr)
    out_ptr = lib.llsm_spectrum_from_envelope(freq_ptr, magn_ptr, freq_arr.size, int(nfft), int(fs))
    return _copy_f32_ptr(out_ptr, int(nfft) // 2 + 1)


def llsm_nonuniform_envelope(
    x: Iterable[float], instant: Iterable[int], winlen: Iterable[int], mode: int
) -> np.ndarray:
    x_arr = _as_f32_array(x, "x")
    instant_arr = _as_i32_array(instant, "instant")
    winlen_arr = _as_i32_array(winlen, "winlen")
    if instant_arr.size != winlen_arr.size:
        raise ValueError("instant and winlen size mismatch")
    x_ptr = ffi.from_buffer("FP_TYPE[]", x_arr)
    i_ptr = ffi.from_buffer("int[]", instant_arr)
    w_ptr = ffi.from_buffer("int[]", winlen_arr)
    out_ptr = lib.llsm_nonuniform_envelope(x_ptr, x_arr.size, i_ptr, w_ptr, instant_arr.size, int(mode))
    return _copy_f32_ptr(out_ptr, instant_arr.size)


def llsm_true_envelope(spectrum: Iterable[float], nfft: int, order: int, niter: int) -> np.ndarray:
    spec_arr = _as_f32_array(spectrum, "spectrum")
    spec_ptr = ffi.from_buffer("FP_TYPE[]", spec_arr)
    out_ptr = lib.llsm_true_envelope(spec_ptr, int(nfft), int(order), int(niter))
    return _copy_f32_ptr(out_ptr, int(order))


def llsm_reduce_spectrum_depth(
    spectrum: Iterable[float], nhop: int, minimum: float, depth: float
) -> np.ndarray:
    spec_arr = _as_f32_array(spectrum, "spectrum").copy()
    spec_ptr = ffi.from_buffer("FP_TYPE[]", spec_arr)
    lib.llsm_reduce_spectrum_depth(spec_ptr, spec_arr.size, int(nhop), float(minimum), float(depth))
    return spec_arr


def interp1(xi: Iterable[float], yi: Iterable[float], x: Iterable[float]) -> np.ndarray:
    xi_arr = _as_f32_array(xi, "xi")
    yi_arr = _as_f32_array(yi, "yi")
    x_arr = _as_f32_array(x, "x")
    if xi_arr.size != yi_arr.size:
        raise ValueError("xi and yi size mismatch")
    out_ptr = lib.cig_interp(
        ffi.from_buffer("FP_TYPE[]", xi_arr),
        ffi.from_buffer("FP_TYPE[]", yi_arr),
        xi_arr.size,
        ffi.from_buffer("FP_TYPE[]", x_arr),
        x_arr.size,
    )
    return _copy_f32_ptr(out_ptr, x_arr.size)


def interp1u(xi0: float, xi1: float, yi: Iterable[float], x: Iterable[float]) -> np.ndarray:
    yi_arr = _as_f32_array(yi, "yi")
    x_arr = _as_f32_array(x, "x")
    out_ptr = lib.cig_interpu(
        float(xi0),
        float(xi1),
        ffi.from_buffer("FP_TYPE[]", yi_arr),
        yi_arr.size,
        ffi.from_buffer("FP_TYPE[]", x_arr),
        x_arr.size,
    )
    return _copy_f32_ptr(out_ptr, x_arr.size)


def sincinterp1u(xi0: float, xi1: float, yi: Iterable[float], x: Iterable[float]) -> np.ndarray:
    yi_arr = _as_f32_array(yi, "yi")
    x_arr = _as_f32_array(x, "x")
    out_ptr = lib.cig_sincinterpu(
        float(xi0),
        float(xi1),
        ffi.from_buffer("FP_TYPE[]", yi_arr),
        yi_arr.size,
        ffi.from_buffer("FP_TYPE[]", x_arr),
        x_arr.size,
    )
    return _copy_f32_ptr(out_ptr, x_arr.size)


def qifft(magn: Iterable[float], k: int) -> tuple[float, float]:
    magn_arr = _as_f32_array(magn, "magn")
    dst_freq = ffi.new("FP_TYPE *")
    val = lib.cig_qifft(ffi.from_buffer("FP_TYPE[]", magn_arr), int(k), dst_freq)
    return float(val), float(dst_freq[0])


def spec2env(S: Iterable[float], nfft: int, f0_ratio: float, nhar: int, Cout: Iterable[float] | None = None) -> np.ndarray:
    s_arr = _as_f32_array(S, "S")
    cout_ptr = ffi.NULL
    if Cout is not None:
        cout_arr = _as_f32_array(Cout, "Cout")
        cout_ptr = ffi.from_buffer("FP_TYPE[]", cout_arr)
    out_ptr = lib.cig_spec2env(
        ffi.from_buffer("FP_TYPE[]", s_arr), int(nfft), float(f0_ratio), int(nhar), cout_ptr
    )
    return _copy_f32_ptr(out_ptr, int(nfft) // 2 + 1)


def lfmodel_from_rd(rd: float, T0: float, Ee: float) -> Any:
    return lib.cig_lfmodel_from_rd(float(rd), float(T0), float(Ee))


def lfmodel_spectrum(model: Any, freq: Iterable[float], return_phase: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    freq_arr = _as_f32_array(freq, "freq")
    ph_ptr = ffi.new("FP_TYPE[]", freq_arr.size) if return_phase else ffi.NULL
    out_ptr = lib.cig_lfmodel_spectrum(model, ffi.from_buffer("FP_TYPE[]", freq_arr), freq_arr.size, ph_ptr)
    ampl = _copy_f32_ptr(out_ptr, freq_arr.size)
    if not return_phase:
        return ampl
    phse = np.frombuffer(ffi.buffer(ph_ptr, freq_arr.size * ffi.sizeof("FP_TYPE")), dtype=np.float32).copy()
    return ampl, phse


def lfmodel_period(model: Any, fs: int, n: int) -> np.ndarray:
    out_ptr = lib.cig_lfmodel_period(model, int(fs), int(n))
    return _copy_f32_ptr(out_ptr, int(n))


def gensins(freq: Iterable[float], ampl: Iterable[float], phse: Iterable[float], fs: int, n: int) -> np.ndarray:
    f_arr = _as_f32_array(freq, "freq")
    a_arr = _as_f32_array(ampl, "ampl")
    p_arr = _as_f32_array(phse, "phse")
    if not (f_arr.size == a_arr.size == p_arr.size):
        raise ValueError("freq/ampl/phse size mismatch")
    out_ptr = lib.cig_gensins(
        ffi.from_buffer("FP_TYPE[]", f_arr),
        ffi.from_buffer("FP_TYPE[]", a_arr),
        ffi.from_buffer("FP_TYPE[]", p_arr),
        f_arr.size,
        int(fs),
        int(n),
    )
    return _copy_f32_ptr(out_ptr, int(n))


def find_peak(x: Iterable[float], lidx: int, uidx: int, orient: int = 1) -> int:
    arr = _as_f32_array(x, "x")
    return int(lib.cig_find_peak(ffi.from_buffer("FP_TYPE[]", arr), int(lidx), int(uidx), int(orient)))


def medfilt1(x: Iterable[float], order: int) -> np.ndarray:
    arr = _as_f32_array(x, "x")
    out_ptr = lib.cig_medfilt(ffi.from_buffer("FP_TYPE[]", arr), arr.size, int(order))
    return _copy_f32_ptr(out_ptr, arr.size)


def moving_avg(x: Iterable[float], halford: float) -> np.ndarray:
    arr = _as_f32_array(x, "x")
    out_ptr = lib.cig_moving_avg(ffi.from_buffer("FP_TYPE[]", arr), arr.size, float(halford))
    return _copy_f32_ptr(out_ptr, arr.size)


def rresample(x: Iterable[float], ratio: float) -> np.ndarray:
    arr = _as_f32_array(x, "x")
    ny_ptr = ffi.new("int *")
    out_ptr = lib.cig_rresample(ffi.from_buffer("FP_TYPE[]", arr), arr.size, float(ratio), ny_ptr)
    return _copy_f32_ptr(out_ptr, int(ny_ptr[0]))


init = llsm_init
analyze_layer0 = llsm_layer0_analyze
synthesize_layer0 = llsm_layer0_synthesize
layer1_from_layer0 = llsm_layer1_from_layer0
