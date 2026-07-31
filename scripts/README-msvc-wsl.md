# Testing MSVC from WSL

YARPGen can drive Microsoft's `cl.exe` (installed on the Windows host, outside
WSL) as just another compiler in the differential-testing pool, alongside the
Linux gcc/clang. This is done through [`cl-proxy`](cl-proxy), a shim that
presents a GCC/Clang-style command line and forwards to MSVC — the same idea as
[`ispc-proxy`](ispc-proxy).

## Requirements

- **Visual Studio (Build Tools) 2019/2022/2026** installed on the Windows host.
  `cl-proxy` auto-detects `vcvars64.bat` in the common install locations; set
  `YARPGEN_VCVARS` to its full Windows path if detection fails.
- **`scripts/` on your `PATH`** so `make` can find `cl-proxy` (same as
  `ispc-proxy`).
- **Run everything under `/mnt/<drive>/`** (e.g. `/mnt/c/...`). `cl.exe` cannot
  see WSL paths like `/home/...`, and the working directory must be a real
  Windows path. Point `run_gen.py -o` at a `/mnt/c` location.
- C++ only (`--std c++`). C / ISPC / SYCL are not handled by the proxy.

## Running

For a 1h run:
```sh
export PATH="/path/to/yarpgen/scripts:$PATH"
cd /mnt/c/some/scratch/dir
run_gen.py --std c++ -o /mnt/c/some/scratch/dir/testing --target "gcc clang msvc" -t 60
```

`run_gen.py` automatically passes `--max-array-dims=3` to the generator whenever
an MSVC target is selected (see below). The MSVC compiler spec and the
`msvc_no_opt` / `msvc_opt` testing sets live in [`test_sets.txt`](test_sets.txt).

## Why the array-dimension cap

The loop generator produces arrays with up to 7 dimensions of size 10–25, i.e.
multi-gigabyte globals. gcc/clang handle these with `-mcmodel=large`; **MSVC has
no large-memory model and a hard 2 GB image limit**, so the default programs often
won't link. The generator therefore gained `--max-array-dims=<n>` (default `0` =
no explicit cap), and `run_gen.py` sets it to `3` when MSVC is in the pool so the
shared test fits every compiler. Tune the value if you want larger arrays.

## Why `/permissive-`

`cl.exe` keeps alive some language extensions and known bugs to allow for backwards
compatibility. yarpgen's intended usage is to find unknown compiler bugs, so we wish 
to filter those out. To that end, `cl-proxy` hardcodes `/permissive-` on 
every `cl.exe` invocation ([cl-proxy:102](cl-proxy#L102)). This turns off MSVC's 
non-conforming language extensions.

MSVC does enable `/permissive-` implicitly under `/std:c++20` and `/std:c++latest`, 
but *not* under `/std:c++14` / `/std:c++17`, where the default is `/permissive`. 
Passing `/permissive-` explicitly keeps conformance behavior uniform across the 
whole `--std` range.

If you're chasing a suspected MSVC-extension issue and *want* the permissive
front end, drop `/permissive-` from the list in `translate()` — there's no
environment knob for it.

## How cl-proxy works

1. **Flag translation** — `-c`→`/c`, `-o X`→`/Fo:X` (compile) or `/Fe:X.exe`
   (link), `-O0`→`/Od`, `-O2/3`→`/O2`, `-std=c++NN`→`/std:...`, and it drops
   flags MSVC has no use for (`-fPIC`, `-mcmodel=large`, `-march=...`, ...).
   Every invocation also gets `/nologo /EHsc /w /permissive-` (see below).
2. **Environment caching** — `vcvars64.bat` takes ~25 s, so running it per
   compile is unworkable. On first use `cl-proxy` captures the MSVC environment
   once into `~/.cache/yarpgen_msvc_env.json` (override with
   `YARPGEN_MSVC_ENV_CACHE`), then invokes `cl.exe` directly (~0.5 s/call),
   forwarding `INCLUDE`/`LIB`/`LIBPATH` to the Win32 process via `WSLENV`. Delete
   the cache file to refresh it (e.g. after a VS update). If `cl.exe` can't be
   located it falls back to a slower `cmd.exe` + `vcvars` path.
3. **Executable launcher** — MSVC produces a Windows `.exe`; `cl-proxy` writes a
   small shell launcher (the extension-less name the harness expects) that runs
   the `.exe` and strips CRLF from its output, so its checksum matches the
   LF-terminated output of the other compilers.
