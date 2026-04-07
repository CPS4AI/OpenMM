# Copyright 2024 Ant Group Co., Ltd.
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

__all__ = ["spu_gelu", "spu_gelu_fm32_baseline", "spu_gelu_hybrid"]

from functools import partial

from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


def _make_spu_gelu(call_target_name, primitive_name):
    def _spu_gelu(input):
        return _prim.bind(input)

    def _spu_gelu_abstract(input):
        shape = input.shape
        dtype = dtypes.canonicalize_dtype(input.dtype)
        return ShapedArray(shape, dtype)

    def _spu_gelu_lowering(ctx, input):
        dtype = mlir.ir.RankedTensorType(input.type)
        call = custom_call(
            call_target_name,
            result_types=[dtype],
            operands=[input],
        )
        return call.results

    def _spu_gelu_jvp(args, tangents):
        raise NotImplementedError()

    def _spu_gelu_batch(args, axes):
        assert axes[0] == axes[1]
        return _spu_gelu(*args), axes

    _prim = core.Primitive(primitive_name)
    _prim.multiple_results = False
    _prim.def_impl(partial(xla.apply_primitive, _prim))
    _prim.def_abstract_eval(_spu_gelu_abstract)

    mlir.register_lowering(_prim, _spu_gelu_lowering)
    ad.primitive_jvps[_prim] = _spu_gelu_jvp
    batching.primitive_batchers[_prim] = _spu_gelu_batch
    return _spu_gelu


# Public facing interface
spu_gelu = _make_spu_gelu("spu.gelu", "spu_gelu")
spu_gelu_fm32_baseline = _make_spu_gelu(
    "spu.gelu_fm32_baseline", "spu_gelu_fm32_baseline"
)
spu_gelu_hybrid = _make_spu_gelu("spu.gelu_hybrid", "spu_gelu_hybrid")


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

