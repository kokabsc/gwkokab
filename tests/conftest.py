# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import numpyro


# NumPyro 0.22 turned argument validation on by default (pyro-ppl/numpyro#2201). With
# validation on, every distribution built without an explicit ``validate_args`` checks its
# parameters in ``__init__`` and warns on out-of-support values in ``log_prob``; because
# ``filterwarnings = ["error"]`` that warning is a test failure. The suite deliberately
# evaluates densities outside their support (normalisation grids, mixtures of components
# with different supports, samplers proposing anywhere) and deliberately builds
# distributions with switched-off parameters, so it runs with the pre-0.22 default. Tests
# that check validation pass ``validate_args=True`` explicitly, which still takes effect.
# This runs at import so it also covers distributions built at collection time.
numpyro.enable_validation(False)
