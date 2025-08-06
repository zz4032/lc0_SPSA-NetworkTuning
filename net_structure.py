#!/usr/bin/env python3

# Copyright (c) 2025 zz4032
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
# See LICENSE file for details.

"""
Network Layer Structure Definitions for Neural Network Training

This module defines the layer structures for various neural network configurations used in the
SPSA-based training script (`tune.py`). Each structure is a tuple of tuples, where each inner tuple
contains a layer ID and its corresponding `c_end` value for Simultaneous Perturbation Stochastic
Approximation (SPSA) optimization.

Structures:
    - T74: Residual network with policy and value heads
    - T74_policy: Policy-layers of T74 (for policy-tournament mode)
    - T79: Residual network with squeeze-and-excitation (SE) layers
"""

# Network layer structures: List[Tuple[int, float]]
# Format: (layer_id, c_end)
# - layer_id: Unique identifier for the layer in the network
# - c_end: Final perturbation size for SPSA optimization

T74 = (
    # Block 1: Residual conv1, conv2
    (5, 0.450), (10, 0.255),
    # Block 2
    (19, 0.348), (24, 0.459),
    # Block 3
    (33, 0.292), (38, 0.516),
    # Block 4
    (47, 0.344), (52, 0.475),
    # Block 5
    (61, 0.337), (66, 0.905),
    # Block 6
    (75, 0.305), (80, 0.331),
    # Block 7
    (89, 0.364), (94, 0.517),
    # Block 8
    (103, 0.307), (108, 0.542),
    # Block 9
    (117, 0.434), (122, 0.779),
    # Block 10
    (131, 0.419), (136, 0.751),
    # Policy and value heads
    (145, 1.793),  # policy, weights
    (156, 1.033),  # policy1, weights
    (152, 2.792),  # value, ip1_val_w
)

T74_policy = (
    # Policy layers only
    (145, 1.812),  # policy, weights
    (156, 0.834),  # policy1, weights
)

T79 = (
    # Block 1: Residual conv1, conv2
    (5, 0.78), (10, 0.52),
    # Block 2
    (19, 0.50), (24, 0.38),
    # Block 3
    (33, 1.07), (38, 0.79),
    # Block 4
    (47, 0.98), (52, 0.94),
    # Block 5
    (61, 0.94), (66, 0.60),
    # Block 6
    (75, 0.76), (80, 0.53),
    # Block 7
    (89, 0.63), (94, 0.72),
    # Block 8
    (103, 0.76), (108, 0.69),
    # Block 9
    (117, 0.78), (122, 0.71),
    # Block 10
    (131, 0.77), (136, 0.73),
    # Block 11
    (145, 0.68), (150, 0.53),
    # Block 12
    (159, 0.72), (164, 0.69),
    # Block 13
    (173, 0.99), (178, 0.96),
    # Block 14
    (187, 0.82), (192, 0.70),
    # Block 15
    (201, 0.81), (206, 0.75),
    # Additional layers
    (215, 1.05),  # se, ip_pol_w
    (217, 0.58),  # value, weights
    (222, 4.00),  # value, ip1_val_w
)
