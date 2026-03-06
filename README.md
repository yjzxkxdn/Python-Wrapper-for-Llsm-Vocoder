# pyllsm Python Wrapper

This directory contains a standalone Python package that wraps `libllsm`.

## Goals

- `import pyllsm` directly after installation
- expose `llsm.h`, `math-funcs.h`, `envelope.h` APIs
- expose selected LLSM-relevant high-level APIs from `ciglet.h`
- include Python tests that mirror `libllsm/test` flow

## Install (editable)

```bash
pip install -e ./pyllsm
```

## Run tests

```bash
pytest pyllsm/tests -q
```

## C test ports

- `libllsm/test/test.c` -> `pyllsm/tests/test_from_test_c.py`
- `libllsm/test/test_mt.c` -> `pyllsm/tests/test_from_test_mt.py`
- `libllsm/test/moresampler_llsm_tool.c` -> `pyllsm/tests/test_from_moresampler_llsm_tool.py`
