# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash on Intel XPU.

The model definition is shared (``common/``); this package holds the XPU
sparse-attention indexer and XPU copies of the model's Triton kernels, selected
by the platform dispatch in ``sparse_indexer.py`` and ``common/``.
"""
