from typing import Callable, Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

class Linear(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_initializer = nnx.initializers.truncated_normal(stddev=0.02),
        bias_initializer = nnx.initializers.zeros,        
        use_bias: bool = False,
        *,
        rngs: nnx.Rngs,
        param_dtype = jnp.float32,
        dtype = jnp.float32
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_initialzer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.param_dtype = param_dtype
        self.dtype = dtype

        self._define_weights(rngs)

    def _define_weights(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            self.kernel_initialzer(
                rngs(), (self.in_features, self.out_features), self.param_dtype
            )
        )

        if self.use_bias:
            self.bias: jax.Array = nnx.Param(
                self.bias_initializer(
                    rngs(), (self.out_features,), self.param_dtype
                )
            )
                
    def __call__(self, x: jax.Array):
        x = jnp.matmul(x, self.weight.value)
        if self.use_bias:
            x = x + self.bias.value


class MLP(nnx.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            activation_fn: Callable[..., jax.Array] = nnx.gelu,
            drop: float = 0.0,
            bias: bool = True,
            *,
            rngs: nnx.Rngs
        ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, use_bias=bias, rngs=rngs)
        self.activation_fn = activation_fn
        self.fc2 = Linear(hidden_features, out_features, use_bias=bias, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool, *, rngs: nnx.Rngs):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.drop(x, deterministic=(None if training == None else not training), rngs=rngs)
        x = self.fc2(x)
        x = self.drop(x, deterministic=(None if training == None else not training), rngs=rngs)
        return x


class SwiGLUFFN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        align_to: int = 8,
        *,
        rngs: nnx.Rngs,
        **args
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = Linear(in_features, swiglu_hidden_features, use_bias=bias, rngs=rngs)
        self.w2 = Linear(in_features, swiglu_hidden_features, use_bias=bias, rngs=rngs)
        self.w3 = Linear(swiglu_hidden_features, out_features, use_bias=bias, rngs=rngs)
    
    def __call__(self, x: jax.Array, **args):
        x1 = self.w1(x),
        x2 = self.w2(x)
        hidden = nnx.silu(x1) * x2
        return self.w3(hidden)

class DepthwiseConv1d(nnx.Module):
    def __init__(
        self,
        features: int,
        kernel_size: int,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs
    ):
        self.features = features
        self.kernel_size = kernel_size
        self.conv = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(kernel_size,),
            padding="SAME",
            feature_group_count=features,
            use_bias=use_bias,
            rngs=rngs
        )
    
    @jax.named_scope("depthwise_conv1d")
    def __call__(self, x: jax.Array, conv_state: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        cache_len = self.kernel_size - 1
        seq_len = x.shape[1]
        
        if conv_state is None:
            x_padded = jnp.pad(x, ((0, 0), (cache_len, 0), (0, 0)), mode="constant", constant_values=0.0)
        else:
            x_padded = jnp.concatenate([jnp.transpose(conv_state, (0, 2, 1)), x], axis=1)
        
        w = self.conv.kernel[:][:, 0, :]
        output = w[0] * x_padded[:, 0:seq_len, :]
        for j in range(1, self.kernel_size):
            output = output + w[j] * x_padded[:, j:j + seq_len, :]
        if self.conv.bias is not None:
            output = output + self.conv.bias[:]

        new_conv_state = jnp.transpose(x_padded[:, -cache_len:, :], (0, 2, 1))

        return output, new_conv_state        
            