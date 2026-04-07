# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

import spu.intrinsic as si
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim


def _run_gelu(label, intrinsic_fn):
    np.random.seed(1234)
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = False
    config.cheetah_2pc_config.enable_mul_lsb_error = True
    config.cheetah_2pc_config.approx_less_precision = 4

    sim = ppsim.Simulator(2, config)

    x = np.random.randn(1 << 20) * 100.0
    spu_fn = ppsim.sim_jax(sim, intrinsic_fn)
    z = spu_fn(x)
    g = jnn.gelu(x)
    diff = z - g

    print("{} max abs diff = {}".format(label, np.max(np.abs(diff))))
    print("{} mean abs diff = {}".format(label, np.mean(np.abs(diff))))


def gelu():
    _run_gelu("gelu_hybrid", si.spu_gelu_hybrid)
    _run_gelu("gelu_fm32_baseline", si.spu_gelu_fm32_baseline)


def silu():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = False
    config.cheetah_2pc_config.enable_mul_lsb_error = True
    config.cheetah_2pc_config.approx_less_precision = 4

    sim = ppsim.Simulator(2, config)

    x = np.random.randn(1 << 10) * 8.0
    spu_fn = ppsim.sim_jax(sim, si.spu_silu)
    z = spu_fn(x)
    g = jnn.silu(x)
    diff = z - g

    # print(f"silu spu out = {z[:10]}")
    # print(f"silu cpu out = {g[:10]}")
    print("silu max diff = {}".format(np.max(diff)))


if __name__ == "__main__":
    gelu()
    #silu()
