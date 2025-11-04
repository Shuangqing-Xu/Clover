# MaliciousSecurity — Distributed Discrete DP & Maliciously Secure Shuffle

This directory contains code and research prototypes related to:
- Google's distributed discrete-noise generation and compression utilities (under the `distributed_dp` subdirectory).
- A proof-of-concept maliciously secure secret-shared shuffle (single-file prototype `rss_shuffle_mali.py`).

The README below summarizes the components, important modules, and quick usage.

## Overview

- Distributed discrete differential privacy and compression utilities implement the Distributed Discrete Gaussian and Skellam mechanisms for federated learning and related utilities. See the submodule documentation: [MaliciousSecurity/distributed_dp/README.md](MaliciousSecurity/distributed_dp/README.md).
- The malicious shuffle prototype demonstrates a secret-shared shuffle with MAC checks. See [`MaliciousSecurity/rss_shuffle_mali.py`](MaliciousSecurity/rss_shuffle_mali.py).

## Key files and symbols

Distributed DP and compression:
- [MaliciousSecurity/distributed_dp/README.md](MaliciousSecurity/distributed_dp/README.md)
- [MaliciousSecurity/distributed_dp/discrete_gaussian_utils.py](MaliciousSecurity/distributed_dp/discrete_gaussian_utils.py)
  - function: [`distributed_dp.discrete_gaussian_utils.sample_discrete_gaussian`](MaliciousSecurity/distributed_dp/discrete_gaussian_utils.py)
- [MaliciousSecurity/distributed_dp/distributed_discrete_gaussian_query.py](MaliciousSecurity/distributed_dp/distributed_discrete_gaussian_query.py)
  - class: [`distributed_dp.distributed_discrete_gaussian_query.DistributedDiscreteGaussianSumQuery`](MaliciousSecurity/distributed_dp/distributed_discrete_gaussian_query.py)
- [MaliciousSecurity/distributed_dp/accounting_utils.py](MaliciousSecurity/distributed_dp/accounting_utils.py)
  - function: [`distributed_dp.accounting_utils.skellam_local_stddev`](MaliciousSecurity/distributed_dp/accounting_utils.py)
- [MaliciousSecurity/distributed_dp/compression_utils.py](MaliciousSecurity/distributed_dp/compression_utils.py)
  - function: [`distributed_dp.compression_utils.sample_rademacher`](MaliciousSecurity/distributed_dp/compression_utils.py)
- Example/run scripts:
  - [MaliciousSecurity/distributed_dp/dme_run.py](MaliciousSecurity/distributed_dp/dme_run.py)
  - [MaliciousSecurity/distributed_dp/fl_run.py](MaliciousSecurity/distributed_dp/fl_run.py)

Malicious shuffle prototype:
- [MaliciousSecurity/rss_shuffle_mali.py](MaliciousSecurity/rss_shuffle_mali.py)
  - class: [`rss_shuffle_mali.SecureRationalVector`](MaliciousSecurity/rss_shuffle_mali.py)
  - function: [`rss_shuffle_mali.combine_and_shuffle`](MaliciousSecurity/rss_shuffle_mali.py)
  - entry/demo: [`rss_shuffle_mali.shuffle_mali`](MaliciousSecurity/rss_shuffle_mali.py)

Notes:
- The distributed DP code is from Google's research code; see the `distributed_dp` README for citations and implementation details.
- The shuffle implementation in `rss_shuffle_mali.py` is a research prototype (not production hardened). It contains secret sharing, MAC operations, PRG (AES-CTR) usage, resharing, and modular arithmetic helpers. See symbol references above for entry points.

