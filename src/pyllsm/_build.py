from pathlib import Path
import platform

from cffi import FFI


ffibuilder = FFI()

CDEF = r"""
typedef float FP_TYPE;

typedef struct {
  FP_TYPE real;
  FP_TYPE imag;
} cplx;

typedef struct {
  FP_TYPE* ampl;
  FP_TYPE* phse;
  int nhar;
} llsm_sinframe;

typedef struct {
  llsm_sinframe** eenv;
  FP_TYPE* emin;
  FP_TYPE* spec;
  int nchannel;
  int nnos;
} llsm_nosframe;

typedef struct {
  llsm_sinframe* sinu;
  llsm_nosframe* noise;
  FP_TYPE f0;
} llsm_frame;

typedef struct {
  int nfrm;
  int nhop;
  int nhar;
  int nhare;
  int nnos;
  FP_TYPE noswarp;
  FP_TYPE mvf;
  FP_TYPE nosf;
  FP_TYPE thop;
  FP_TYPE* nosbandf;
  int nnosband;
} llsm_conf;

typedef struct {
  llsm_conf conf;
  llsm_frame** frames;
} llsm_layer0;

typedef struct {
  FP_TYPE** vt_resp_magn;
  FP_TYPE** vs_har_ampl;
  FP_TYPE** vs_har_phse;
  FP_TYPE* vs_rd;
  FP_TYPE* lip_resp_magn;
  FP_TYPE* lip_resp_phse;
  int nfrm;
  int nfft;
  int fnyquist;
} llsm_layer1;

typedef struct {
  int a_nhop;
  int a_nhar;
  int a_nhare;
  int a_nnos;
  int a_f0refine;
  FP_TYPE a_wsize;
  FP_TYPE a_noswarp;
  FP_TYPE a_tfft;
  FP_TYPE a_mvf;
  FP_TYPE a_nosf;
  FP_TYPE* a_nosbandf;
  int a_nnosband;
  int s_fs;
  int s_noiseonly;
  int s_n0;
  int s_n1;
} llsm_parameters;

typedef struct {
  FP_TYPE* y;
  FP_TYPE* y_sin;
  FP_TYPE* y_nos;
  int ny;
  FP_TYPE fs;
} llsm_output;

typedef struct {
  FP_TYPE T0;
  FP_TYPE te;
  FP_TYPE tp;
  FP_TYPE ta;
  FP_TYPE Ee;
} lfmodel;

llsm_frame* llsm_create_frame(int nhar, int nhare, int nnos, int nchannel);
llsm_layer0* llsm_create_empty_layer0(llsm_conf conf);
void llsm_copy_frame(llsm_frame* dst, llsm_frame* src);
void llsm_copy_sinframe(llsm_sinframe* dst, llsm_sinframe* src);
void llsm_copy_nosframe(llsm_nosframe* dst, llsm_nosframe* src);
void llsm_delete_frame(llsm_frame* dst);

llsm_parameters llsm_init(int nnosband);
void llsm_deinit(llsm_parameters dst);
llsm_layer0* llsm_layer0_analyze(llsm_parameters param, FP_TYPE* x, int nx, int fs,
  FP_TYPE* f0, int nf0, FP_TYPE** xap);
llsm_output* llsm_layer0_synthesize(llsm_parameters param, llsm_layer0* model);
void llsm_delete_layer0(llsm_layer0* dst);
void llsm_delete_output(llsm_output* dst);

FP_TYPE* llsm_uniform_faxis(int nfft, int fs);
FP_TYPE* llsm_liprad(FP_TYPE* freq, int nf, FP_TYPE radius, FP_TYPE* dst_phaseresp);
FP_TYPE* llsm_harmonic_cheaptrick(FP_TYPE* ampl, int nhar, FP_TYPE f0, int nfft, FP_TYPE fs);
FP_TYPE* llsm_harmonic_minphase(FP_TYPE* ampl, int nhar);
llsm_layer1* llsm_layer1_from_layer0(llsm_parameters param, llsm_layer0* model, int nfft, int fs);
void llsm_delete_layer1(llsm_layer1* dst);
void llsm_layer0_phaseshift(llsm_layer0* dst, FP_TYPE* phaseshift);

FP_TYPE* llsm_chebyfilt(FP_TYPE* x, int nx, FP_TYPE cutoff1, FP_TYPE cutoff2, char* type);
FP_TYPE* llsm_py_chebyfilt(FP_TYPE* x, int nx, FP_TYPE cutoff1, FP_TYPE cutoff2, char* type);

void llsm_reduce_spectrum_depth(FP_TYPE* spectrum, int ns, int nhop, FP_TYPE minimum, FP_TYPE depth);
FP_TYPE* llsm_true_envelope(FP_TYPE* spectrum, int nfft, int order, int niter);
FP_TYPE* llsm_warp_freq(FP_TYPE fmin, FP_TYPE fmax, int n, FP_TYPE warp_const);
FP_TYPE* llsm_geometric_envelope(FP_TYPE* spectrum, int nfft, int fs, FP_TYPE* freq, int nf);
FP_TYPE* llsm_spectrum_from_envelope(FP_TYPE* freq, FP_TYPE* magn, int nf, int nfft, int fs);
FP_TYPE* llsm_nonuniform_envelope(FP_TYPE* x, int nx, int* instant, int* winlen, int ni, int mode);

FP_TYPE* cig_gensins(FP_TYPE* freq, FP_TYPE* ampl, FP_TYPE* phse, int nsin, int fs, int n);
FP_TYPE* cig_interp(FP_TYPE* xi, FP_TYPE* yi, int ni, FP_TYPE* x, int nx);
FP_TYPE* cig_interpu(FP_TYPE xi0, FP_TYPE xi1, FP_TYPE* yi, int ni, FP_TYPE* x, int nx);
FP_TYPE* cig_sincinterpu(FP_TYPE xi0, FP_TYPE xi1, FP_TYPE* yi, int ni, FP_TYPE* x, int nx);
FP_TYPE cig_qifft(FP_TYPE* magn, int k, FP_TYPE* dst_freq);
FP_TYPE* cig_spec2env(FP_TYPE* S, int nfft, FP_TYPE f0, int nhar, FP_TYPE* Cout);
lfmodel cig_lfmodel_from_rd(FP_TYPE rd, FP_TYPE T0, FP_TYPE Ee);
FP_TYPE* cig_lfmodel_spectrum(lfmodel model, FP_TYPE* freq, int nf, FP_TYPE* dst_phase);
FP_TYPE* cig_lfmodel_period(lfmodel model, int fs, int n);
int cig_find_peak(FP_TYPE* x, int lidx, int uidx, int orient);
FP_TYPE* cig_medfilt(FP_TYPE* x, int nx, int order);
FP_TYPE* cig_moving_avg(FP_TYPE* x, int nx, FP_TYPE halford);
FP_TYPE* cig_rresample(FP_TYPE* x, int nx, FP_TYPE ratio, int* ny);

void llsm_py_free(void* p);
"""

ffibuilder.cdef(CDEF)

_libllsm = Path("vendor/libllsm")

SOURCES = [
    _libllsm / "math-funcs.c",
    _libllsm / "llsm-layer0.c",
    _libllsm / "llsm-layer1.c",
    _libllsm / "envelope.c",
    _libllsm / "external" / "ciglet" / "ciglet.c",
    _libllsm / "external" / "ciglet" / "external" / "fftsg_h.c",
    _libllsm / "external" / "ciglet" / "external" / "fast_median.c",
]

INCLUDE_DIRS = [
    _libllsm,
    _libllsm / "external" / "ciglet",
]

libraries = []
if platform.system() != "Windows":
    libraries.append("m")

ffibuilder.set_source(
    "pyllsm._pyllsm_cffi",
    r"""
    #include <stdlib.h>
    #define FP_TYPE float
    #include "llsm.h"
    #include "math-funcs.h"
    #include "envelope.h"
    #include "external/ciglet/ciglet.h"

    FP_TYPE* llsm_py_chebyfilt(FP_TYPE* x, int nx, FP_TYPE cutoff1, FP_TYPE cutoff2, char* type) {
      return llsm_chebyfilt(x, nx, cutoff1 / 2.0f, cutoff2 / 2.0f, type);
    }

    void llsm_py_free(void* p) {
      free(p);
    }
    """,
    sources=[str(p).replace("\\", "/") for p in SOURCES],
    include_dirs=[str(p).replace("\\", "/") for p in INCLUDE_DIRS],
    define_macros=[("FP_TYPE", "float")],
    libraries=libraries,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
