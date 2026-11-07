# Copyright 2025 The JAX Authors.
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

"""Mamba2 JAX/Flax NNX Implementation.

A pure JAX/Flax implementation of the Mamba2 architecture using the State Space Duality (SSD) mechanism.
Reference: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
Paper: https://arxiv.org/abs/2405.21060
"""

# Implementation based on repo https://github.com/jax-state-spaces/mamba2-jax/blob/main/mamba2_jax/modeling.py
# This version is truncated for regressor tasks and does not consider language modelling in as much detail.

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from mamba2.nn.linear import DepthwiseConv1d, ACT2FN
from mamba2.nn.ssd import ssd_forward, ACT2FN
from mamba2.nn.norm import RMSNorm

@dataclasses.dataclass(frozen=True)
class Mamba2Config:
    hidden_size: int = 768
    state_size: int = 128
    head_dim: int = 64
    chunk_size: int = 256
    expand: int = 2
    conv_kernel: int = 4
    num_hidden_layers: int = 24
    layer_norm_epsilon: float = 1e-5
    
    use_bias: bool = True
    use_conv_bias: bool = True
    hidden_act: Literal["silu", "gelu", "relu", "tanh"] = "silu"
    
    A_initializer_range: tuple[float, float] = (1.0, 16.0)
    
    time_step_min: float = 0.001
    time_step_max: float = 1.0
    time_step_floor: float = 1e-4
    time_step_limit: tuple[float, float] = (0.0, float("inf"))
    
    residual_in_fp32: bool = True
    
    @property
    def intermediate_size(self) -> int:
        return int(self.hidden_size * self.expand)
    
    @property
    def num_heads(self) -> int:
        return self.intermediate_size // self.head_dim
    
@jax.tree_util.register_pytree_node_class
class Mamba2Cache:
    "Cache for mamba2 SSM and convolution states."
    
    ssm_states: list[jnp.ndarray]
    conv_states: list[jnp.ndarray]

    def tree_flatten(self):
        return (self.ssm_states, self.conv_states), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(ssm_states=list(children[0]), conv_states=list(children[1]))

def create_empty_cache(
    cfg: Mamba2Config,
    batch_size: int,
    dtype: jnp.dtype = jnp.float32,
) -> Mamba2Cache:
    """Create an empty cache for Mamba2 model.

    Args:
        cfg: Mamba2Config for the model.
        batch_size: Batch size for the cache.
        dtype: Data type for cache arrays.

    Returns:
        Empty Mamba2Cache with zero-initialized states.
    """
    conv_dim = cfg.intermediate_size + 2 * cfg.state_size
    cache_len = cfg.conv_kernel - 1

    conv_states = [jnp.zeros((batch_size, conv_dim, cache_len), dtype=dtype) for _ in range(cfg.num_hidden_layers)]
    ssm_states = [
        jnp.zeros((batch_size, cfg.num_heads, cfg.head_dim, cfg.state_size), dtype=dtype)
        for _ in range(cfg.num_hidden_layers)
    ]

    return Mamba2Cache(ssm_states=ssm_states, conv_states=conv_states)

class Mamba2Mixer(nnx.Module):
    """Mamba2 mixer block using the SSD algorithm."""

    def __init__(self, cfg: Mamba2Config, layer_idx: int, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.ssm_state_size = cfg.state_size
        self.intermediate_size = cfg.intermediate_size
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.chunk_size = cfg.chunk_size
        self.dt_min, self.dt_max = cfg.time_step_limit
        self.act = ACT2FN[cfg.hidden_act]

        # Input projection
        proj_size = 2 * (self.intermediate_size + self.ssm_state_size) + self.num_heads
        self.in_proj = nnx.Linear(cfg.hidden_size, proj_size, use_bias=cfg.use_bias, rngs=rngs)

        # Depthwise conv
        conv1d_dim = self.intermediate_size + 2 * self.ssm_state_size
        self.conv1d = DepthwiseConv1d(conv1d_dim, cfg.conv_kernel, use_bias=cfg.use_conv_bias, rngs=rngs)

        # SSM parameters
        key = rngs.params()
        low, high = cfg.time_step_min, cfg.time_step_max
        floor = cfg.time_step_floor
        dt_init = jnp.exp(jax.random.uniform(key, (cfg.num_heads,)) * (jnp.log(high) - jnp.log(low)) + jnp.log(low))
        dt_init = jnp.maximum(dt_init, floor)
        self.dt_bias = nnx.Param(dt_init + jnp.log(-jnp.expm1(-dt_init)))  # inverse softplus

        key = rngs.params()
        A_low, A_high = cfg.A_initializer_range
        A_init = jax.random.uniform(key, (cfg.num_heads,), minval=A_low, maxval=A_high)
        self.A_log = nnx.Param(jnp.log(A_init))

        self.D = nnx.Param(jnp.ones((cfg.num_heads,)))

        # Internal norm and output projection
        self.norm = RMSNorm(self.intermediate_size, eps=1e-5, gate_residual=True, rngs=rngs)
        self.out_proj = nnx.Linear(self.intermediate_size, cfg.hidden_size, use_bias=cfg.use_bias, rngs=rngs)

    @jax.named_scope("mamba2_mixer")
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        conv_state: jnp.ndarray | None = None,
        ssm_state: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B_size, L, _ = hidden_states.shape

        # 1) Parallel projection
        zxbcdt = self.in_proj(hidden_states)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.intermediate_size - 2 * self.ssm_state_size - self.num_heads) // 2

        z0, x0, z, xBC, dt = jnp.split(
            zxbcdt,
            [
                d_mlp,
                2 * d_mlp,
                2 * d_mlp + self.intermediate_size,
                2 * d_mlp + self.intermediate_size + self.intermediate_size + 2 * self.ssm_state_size,
            ],
            axis=-1,
        )

        # 2) Depthwise causal convolution with state caching
        xBC, new_conv_state = self.conv1d(xBC, conv_state=conv_state)
        xBC = self.act(xBC)
        x, B_t, C_t = jnp.split(xBC, [self.intermediate_size, self.intermediate_size + self.ssm_state_size], axis=-1)

        # 3) SSD forward with state caching
        init_state = ssm_state[:, None, ...] if ssm_state is not None else None
        A = -jnp.exp(self.A_log[:].astype(jnp.float32))

        B_exp = jnp.broadcast_to(jnp.expand_dims(B_t, 2), (B_size, L, self.num_heads, self.ssm_state_size))
        C_exp = jnp.broadcast_to(jnp.expand_dims(C_t, 2), (B_size, L, self.num_heads, self.ssm_state_size))

        y, new_ssm_state = ssd_forward(
            x=x.reshape(B_size, L, -1, self.head_dim),
            dt=dt,
            A=A,
            B_mat=B_exp,
            C_mat=C_exp,
            chunk_size=self.chunk_size,
            D=self.D[:],
            dt_bias=self.dt_bias[:],
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            initial_states=init_state,
            return_final_states=True,
        )
        y = y.reshape(B_size, L, -1)

        # 4) Residual gate normalization
        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = jnp.concatenate([self.act(z0) * x0, y], axis=-1)

        # 5) Output projection
        return self.out_proj(y), new_conv_state, new_ssm_state

class Mamba2Block(nnx.Module):
    """Single Mamba2 block with pre-norm and residual connection."""

    def __init__(self, cfg: Mamba2Config, layer_idx: int, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.residual_in_fp32 = cfg.residual_in_fp32
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon, rngs=rngs)
        self.mixer = Mamba2Mixer(cfg, layer_idx=layer_idx, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        conv_state: jnp.ndarray | None = None,
        ssm_state: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        residual = hidden_states
        hs = self.norm(hidden_states.astype(jnp.float32))
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hs_out, new_conv_state, new_ssm_state = self.mixer(hs, conv_state=conv_state, ssm_state=ssm_state)
        return residual + hs_out, new_conv_state, new_ssm_state

class Mamba2Model(nnx.Module):
    """Mamba2 backbone model (no task-specific head)."""

    def __init__(self, cfg: Mamba2Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embedder = nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.hidden_size, rngs=rngs)
        self.layers = nnx.List([Mamba2Block(cfg, layer_idx=i, rngs=rngs) for i in range(cfg.num_hidden_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon, rngs=rngs)

    @jax.named_scope("mamba2_backbone")
    def __call__(
        self,
        input_ids: jax.Array | None = None,
        inputs_embeds: jax.Array | None = None,
        cache: Mamba2Cache | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, jax.Array | Mamba2Cache | list[jnp.ndarray] | None]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        hidden_states = self.embedder(input_ids) if inputs_embeds is None else inputs_embeds

        # Extract per-layer states from cache or initialize
        if cache is None:
            conv_states = [None] * self.cfg.num_hidden_layers
            ssm_states = [None] * self.cfg.num_hidden_layers
        else:
            if len(cache.conv_states) != self.cfg.num_hidden_layers:
                raise ValueError("cache.conv_states length must equal num_hidden_layers")
            if len(cache.ssm_states) != self.cfg.num_hidden_layers:
                raise ValueError("cache.ssm_states length must equal num_hidden_layers")
            conv_states = cache.conv_states
            ssm_states = cache.ssm_states

        all_hidden_states = [] if output_hidden_states else None
        new_conv_states = []
        new_ssm_states = []

        for layer, conv_state, ssm_state in zip(self.layers, conv_states, ssm_states):
            hidden_states, new_conv_state, new_ssm_state = layer(
                hidden_states, conv_state=conv_state, ssm_state=ssm_state
            )
            new_conv_states.append(new_conv_state)
            new_ssm_states.append(new_ssm_state)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Build updated cache
        updated_cache = Mamba2Cache(ssm_states=new_ssm_states, conv_states=new_conv_states)

        return {
            "last_hidden_state": hidden_states,
            "cache": updated_cache,
            "hidden_states": all_hidden_states,
        }

class Mamba2Forecaster(nnx.Module):
    """Mamba2-based time series forecaster."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 768,
        n_layers: int = 4,
        output_dim: int = 1,
        forecast_horizon: int = 24,
        d_state: int = 128,
        headdim: int = 64,
        d_conv: int = 4,
        chunk_size: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

        self.input_proj = nnx.Linear(input_dim, d_model, rngs=rngs)
        cfg = Mamba2Config(
            vocab_size=1,
            hidden_size=d_model,
            state_size=d_state,
            head_dim=headdim,
            conv_kernel=d_conv,
            chunk_size=chunk_size,
            num_hidden_layers=n_layers,
        )
        self.mamba2 = Mamba2Model(cfg, rngs=rngs)
        self.output_proj = nnx.Linear(d_model, output_dim * forecast_horizon, rngs=rngs)

    @jax.named_scope("mamba2_forecaster")
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. Input: (batch, seq_len, input_dim) -> Output: (batch, forecast_horizon, output_dim)."""
        x_proj = self.input_proj(x)
        outputs = self.mamba2(input_ids=None, inputs_embeds=x_proj)
        last_hidden = outputs["last_hidden_state"][:, -1, :]
        out = self.output_proj(last_hidden)
        return out.reshape(x.shape[0], self.forecast_horizon, self.output_dim)