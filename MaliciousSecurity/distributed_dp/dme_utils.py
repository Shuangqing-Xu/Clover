# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for distributed mean estimation."""
import time

import numpy as np
import tensorflow as tf

from distributed_dp.modular_clipping_factory import modular_clip_by_value


def generate_client_data(d, n, l2_norm=1):
  """Sample `n` of `d`-dim vectors on the l2 ball with radius `l2_norm`.

  Args:
    d: The dimension of the client vector.
    n: The number of clients.
    l2_norm: The L2 norm of the sampled vector.

  Returns:
    A list of `n` np.array each with shape (d,).
  """
  vectors = np.random.normal(size=(n, d))
  unit_vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
  scaled_vectors = unit_vectors * l2_norm
  # Cast to float32 as TF implementations use float32.
  return list(scaled_vectors.astype(np.float32))


def compute_dp_average(client_data, dp_query, is_compressed, bits):
  """Aggregate client data with DPQuery's interface and take average."""
  global_state = dp_query.initial_global_state()
  sample_params = dp_query.derive_sample_params(global_state)

  client_template = tf.zeros_like(client_data[0])
  sample_state = dp_query.initial_sample_state(client_template)

  if is_compressed:
    # Achieve compression via modular clipping. Upper bound is exclusive.
    clip_lo, clip_hi = -(2**(bits - 1)), 2**(bits - 1)

    # 1. Client pre-processing stage.
    Total_client_time = 0
    for x in client_data:
      Per_Client_time_start = time.time()
      record = tf.convert_to_tensor(x)
      prep_record = dp_query.preprocess_record(sample_params, record)
      # print("prep_record")
      # Client applies modular clip on the preprocessed record.
      prep_record = modular_clip_by_value(prep_record, clip_lo, clip_hi)
      Per_Client_time_end = time.time()
      Total_client_time += (Per_Client_time_end-Per_Client_time_start)
      sample_state = dp_query.accumulate_preprocessed_record(
          sample_state, prep_record)

    # 2. Server applies modular clip on the aggregate.
    Server_time_start = time.time()
    sample_state = modular_clip_by_value(sample_state, clip_lo, clip_hi)
    

  else:
    for x in client_data:
      record = tf.convert_to_tensor(x)
      sample_state = dp_query.accumulate_record(
          sample_params, sample_state, record=record)

  # Apply server post-processing.
  agg_result, _, _ = dp_query.get_noised_result(sample_state, global_state)

  # The agg_result should have the same input type as client_data.
  assert agg_result.shape == client_data[0].shape
  assert agg_result.dtype == client_data[0].dtype
  Server_time = time.time() - Server_time_start

  # Take the average on the aggregate.
  return agg_result / len(client_data), Total_client_time, Server_time
