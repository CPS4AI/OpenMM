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

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/conf/2pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_bert -- --config `pwd`/examples/python/conf/2pc.json

import argparse
import json
import time
from contextlib import contextmanager

import flax.linen as fnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
from transformers import (
    BertTokenizerFast,
    FlaxBertForSequenceClassification,
)

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

FIXED_SEQ_LEN = 128


def _gelu(x):
    return intrinsic.spu_gelu_hybrid(x)


def _softmax(x, axis=-1, where=None, initial=None):
    x_max = jax.numpy.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max
    # spu.neg_exp will clip values that too large.
    # nexp = jax.numpy.exp(x) * (x > -14.0)
    nexp = intrinsic.spu_neg_exp(x)
    divisor = jax.numpy.sum(nexp, axis, where=where, keepdims=True)
    return nexp / divisor


@contextmanager
def hijack(enabled=True):
    if not enabled:
        yield
        return
    # hijack some target functions
    jnn_gelu = jnn.gelu
    fnn_gelu = fnn.gelu
    jnn_sm = jnn.softmax
    fnn_sm = fnn.softmax

    jnn.gelu = _gelu
    fnn.gelu = _gelu
    jnn.softmax = _softmax
    fnn.softmax = _softmax
    yield
    # recover back
    jnn.gelu = jnn_gelu
    fnn.gelu = fnn_gelu
    jnn.softmax = jnn_sm
    fnn.softmax = fnn_sm


def run_on_cpu(model, input_ids, attention_masks):
    print(f"Running on CPU ...")
    params = model.params

    def eval(params, input_ids, attention_masks):
        with hijack(enabled=False):
            logits = model(input_ids, attention_masks, params=params)[0]
        return logits

    start = time.time()
    logits = eval(params, input_ids, attention_masks)
    end = time.time()
    print(f"CPU runtime: {(end - start)}s\noutput logits: {logits}")


def run_on_spu(model, input_ids, attention_masks):
    print(f"Running on SPU ...")
    params = model.params

    def eval(params, input_ids, attention_masks):
        with hijack(enabled=True):
            logits = model(input_ids, attention_masks, params=params)[0]
        return logits

    spu_input_ids = ppd.device("P1")(lambda x: x)(input_ids)
    spu_attention_masks = ppd.device("P1")(lambda x: x)(attention_masks)
    spu_params = ppd.device("P2")(lambda x: x)(params)
    start = time.time()
    logits_spu = ppd.device("SPU")(eval, copts=copts)(
        spu_params, spu_input_ids, spu_attention_masks
    )
    end = time.time()
    logits_spu = ppd.get(logits_spu)
    print(f"SPU runtime: {(end - start)}s\noutput logits: {logits_spu}")


def make_fixed_input(tokenizer):
    # Build a deterministic single-sentence input with exactly 128 valid tokens.
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    filler_tokens = tokenizer.encode("hello", add_special_tokens=False)
    filler_id = filler_tokens[0] if filler_tokens else tokenizer.unk_token_id

    input_ids = [cls_id] + [filler_id] * (FIXED_SEQ_LEN - 2) + [sep_id]
    attention_mask = [1] * FIXED_SEQ_LEN

    return (
        jnp.asarray([input_ids], dtype=jnp.int32),
        jnp.asarray([attention_mask], dtype=jnp.int32),
    )


def main(tokenizer_func, model_func, checkpoint):
    model = model_func.from_pretrained(checkpoint)
    tokenizer = tokenizer_func.from_pretrained(checkpoint)
    input_ids, attention_masks = make_fixed_input(tokenizer)

    print(
        f"input_ids shape: {input_ids.shape}, "
        f"attention_mask sum: {attention_masks.sum()}"
    )
    run_on_cpu(model, input_ids, attention_masks)
    run_on_spu(model, input_ids, attention_masks)


if __name__ == "__main__":
    tokenizer = BertTokenizerFast
    model = FlaxBertForSequenceClassification
    checkpoint = "bert-base-cased"
    main(tokenizer, model, checkpoint)
