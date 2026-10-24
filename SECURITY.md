# Security policy

DAMASK-Metal is a local simulation code. It has no network service, and it trusts the input files it is given in the same way DAMASK does.

## Reporting a vulnerability

Report it privately through GitHub's **Security → Report a vulnerability** (private security advisory) on this repository. Please do not open a public issue. Problems in DAMASK itself (code outside the Metal backend and the `tdt`/`twintrip` models) belong upstream: https://damask-multiphysics.org.

## Audits

- **2026-10-24, run 1:** quick, scoped audit with Cloudflare's security-audit skill.
  - Scope: all changes since DAMASK v3.1.0; source review only.
  - Results: no confirmed findings and three leads. All three leads were in test tools that read capture or golden files: an unchecked state stride, an `int` allocation overflow, and an unchecked parameter-blob length.
  - All three were fixed at the interface (lengths and strides are checked and the tools validate file sizes). Crafted files are now rejected.
  - Not yet audited: the CPU code of the `tdt`/`twintrip` models and the size arithmetic of the GPU spectral path.
