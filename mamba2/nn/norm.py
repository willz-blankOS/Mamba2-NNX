import jax
import jax.numpy as jnp
import flax.nnx as nnx

class LayerNorm(nnx.Module):
    def __init__(self):
        super().__init__()
        

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-5, *, rngs: nnx.Rngs):
        super().__init__()
        self.weight = nnx.Param(
            nnx.initializers.ones(rngs(), (dim,))
        )
        self.eps = eps

    def _norm(self, x: jax.Array):
        return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), -1, keepdims=True) + self.eps)

    def __call__(self, x: jax.Array):
        output = self._norm(x)
        return output * self.weight.value