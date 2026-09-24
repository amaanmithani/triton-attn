"""FlashAttention-2 style fused attention in Triton."""

from tattn.api import attention, triton_available

__all__ = ["attention", "triton_available"]
