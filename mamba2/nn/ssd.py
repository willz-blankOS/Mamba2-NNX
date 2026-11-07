import jax
import jax.numpy as jnp
import flax.nnx as nnx

def _pad_seq_dim(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Pad zeros at the end of the sequence dimension (axis=1)."""
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)

def segsum(x: jax.Array) -> jax.Array:
    """
    
    """
    T = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=x.dtype), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum

def _ssd_single_chunk(
    x: jax.Array, dt: jax.Array, A: jax.Array, B_mat: jax.Array, C_mat: jax.Array, D: jax.Array, return_final_states: bool):
    num_heads = x.shape[2]
    x_disc = x * dt[..., None]
    A_disc = A.astype(x_disc.dtype) * dt
    D_residual = D.reshape(1, 1, num_heads, 1) * x
    A_t = jnp.transpose(A_disc, (0, 2, 1))
    L_mat = jnp.exp(segsum(A_t))
    y = jnp.einsum("blhn,bshn,bhls,bshp->blhp", C_mat, B_mat, L_mat, x_disc) + D_residual
    final_state = None
    if return_final_states:
        A_cumsum = jnp.cumsum(A_t, axis=-1)
        decay = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
        final_state = jnp.einsum("blhn,bhl,blhp->bhpn", B_mat, decay, x_disc)
    return y, final_state

def ssd_forward(
    x: jnp.ndarray,  # (B, L, H, P)
    dt: jnp.ndarray,  # (B, L, H)
    A: jnp.ndarray,  # (H,)
    B_mat: jnp.ndarray,  # (B, L, H, N)
    C_mat: jnp.ndarray,  # (B, L, H, N)
    chunk_size: int,
    D: jnp.ndarray,  # (H,)
    dt_bias: jnp.ndarray,  # (H,)
    dt_min: float,
    dt_max: float,
    initial_states: jnp.ndarray | None = None,
    return_final_states: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """SSD (State Space Duality) forward pass with chunked computation.

    Args:
        x: Input tensor (batch_size, seq_len, num_heads, head_dim)
        dt: Time deltas (batch_size, seq_len, num_heads)
        A: State transition scalar per head (num_heads,)
        B_mat: Input-to-state matrix (batch_size, seq_len, num_heads, state_size)
        C_mat: State-to-output matrix (batch_size, seq_len, num_heads, state_size)
        chunk_size: Size of chunks for efficient computation
        D: Skip connection weights (num_heads,)
        dt_bias: Bias for time deltas (num_heads,)
        dt_min: Minimum time delta after clamping
        dt_max: Maximum time delta after clamping
        initial_states: Optional initial SSM states (batch, 1, heads, head_dim, state_size)
        return_final_states: Whether to return final SSM states

    Returns:
        y: Output tensor (batch_size, seq_len, num_heads, head_dim)
        final_state: Optional final states (batch_size, num_heads, head_dim, state_size)
    """
    _B_size, seq_len, num_heads, _head_dim = x.shape

    # Apply dt bias with softplus and clamp
    dt = jax.nn.softplus(dt + dt_bias)
    dt = jnp.clip(dt, dt_min, dt_max)

    if initial_states is None and seq_len <= chunk_size:
        return _ssd_single_chunk(x, dt, A, B_mat, C_mat, D, return_final_states)

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    # Pad tensors along sequence dimension
    x_padded = _pad_seq_dim(x, pad_size)
    dt_padded = _pad_seq_dim(dt, pad_size)
    B_padded = _pad_seq_dim(B_mat, pad_size)
    C_padded = _pad_seq_dim(C_mat, pad_size)

    # D residual connection
    D_residual = D.reshape(1, 1, num_heads, 1) * x_padded

    # Discretize x and A
    x_disc = x_padded * dt_padded[..., None]
    A_disc = A.astype(x_disc.dtype) * dt_padded

    # Chunk everything
    def chunk_tensor(t):
        b, cl, *remaining = t.shape
        return t.reshape(b, cl // chunk_size, chunk_size, *remaining)

    x_blk = chunk_tensor(x_disc)
    A_blk = chunk_tensor(A_disc)
    B_blk = chunk_tensor(B_padded)
    C_blk = chunk_tensor(C_padded)

    # A cumsum over intra-chunk time dimension
    A_blk2 = jnp.transpose(A_blk, (0, 3, 1, 2))
    A_cumsum = jnp.cumsum(A_blk2, axis=-1)

    # 1. Intra-chunk (diagonal blocks)
    L_mat = jnp.exp(segsum(A_blk2))
    Y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blk, B_blk, L_mat, x_blk)

    # 2. States within each chunk
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B_blk, decay_states, x_blk)

    # 3. Inter-chunk recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1, ...])
    states = jnp.concatenate([initial_states, states], axis=1)

    A_end = A_cumsum[..., -1]
    A_end_padded = jnp.pad(A_end, ((0, 0), (0, 0), (1, 0)))
    decay_chunk = jnp.exp(segsum(A_end_padded))
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1, ...], new_states[:, -1, ...]

    # 4. Convert states -> outputs
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum("bclhn,bchpn,bhcl->bclhp", C_blk, states, state_decay_out)

    y = Y_diag + Y_off
    b, c, l, h, p = y.shape
    y = y.reshape(b, c * l, h, p)
    y = y + D_residual

    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    return (y, final_state) if return_final_states else (y, None)

ACT2FN = {
    "silu": nnx.silu,
    "gelu": nnx.gelu,
    "relu": nnx.relu,
    "tanh": jnp.tanh,
}