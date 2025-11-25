"""Sparse linear layers for vLLM using 2:4 structured sparsity."""

from typing import Any, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod, VocabParallelEmbedding
from vllm.utils.torch_utils import direct_register_custom_op


logger = init_logger(__name__)

try:
    import hipsparse_py
    HAS_HIPSPARSE = True
except ImportError:
    HAS_HIPSPARSE = False
    logger.warning(
        "hipsparse_py not found. Sparse layers will fall back to dense computation. "
        "To use 2:4 structured sparsity, install hipsparse_py."
    )


def hipsarselt_spmm_impl(M: int, K: int, N: int, dtype: torch.dtype,
                          compressed_data: torch.Tensor, dense_b: torch.Tensor,
                          output: torch.Tensor, workspace: torch.Tensor,
                          alpha: float, beta: float, stream: int) -> None:
    hipsparse_py.spmm_run(M, K, N, dtype, compressed_data, dense_b, output, workspace, alpha, beta, stream)

def hipsarselt_spmm_fake(M: int, K: int, N: int, dtype: torch.dtype,
                          compressed_data: torch.Tensor, dense_b: torch.Tensor,
                          output: torch.Tensor, workspace: torch.Tensor,
                          alpha: float, beta: float, stream: int) -> None:
    pass

direct_register_custom_op(
    op_name="hipsarselt_spmm",
    op_func=hipsarselt_spmm_impl,
    fake_impl=hipsarselt_spmm_fake,
    mutates_args=["output"]
)

class SparseConfig:
    """Configuration for sparse linear layers.

    Args:
        sparsity_type: Type of sparsity pattern. Currently only "2:4" is supported.
        enable_sparse: Whether to enable sparse computation. If False, uses dense fallback.
    """

    def __init__(
        self,
        sparsity_type: str = "2:4",
        enable_sparse: bool = True,
    ):
        self.sparsity_type = sparsity_type
        self.enable_sparse = enable_sparse and HAS_HIPSPARSE

        if self.sparsity_type != "2:4":
            raise ValueError(f"Unsupported sparsity type: {self.sparsity_type}. "
                           f"Only '2:4' is currently supported.")

        if enable_sparse and not HAS_HIPSPARSE:
            logger.warning(
                "Sparse computation requested but hipsparse_py not available. "
                "Falling back to dense computation."
            )

    def __repr__(self) -> str:
        return (f"SparseConfig(sparsity_type={self.sparsity_type}, "
                f"enable_sparse={self.enable_sparse})")


class SparseLinearMethod(LinearMethodBase):
    """Base sparse linear method for vLLM.

    This method handles weight creation and sparse matrix multiplication
    using 2:4 structured sparsity with hipSPARSELt.
    """

    def __init__(self, sparse_config: SparseConfig):
        self.sparse_config = sparse_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for the sparse linear layer.

        For sparse layers, we store both the original weight and
        the compressed sparse representation.
        """
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Store layer metadata
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.sparse_config = self.sparse_config

        # Create the weight parameter (will be pruned and compressed after loading)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

        # Initialize placeholders for sparse representation
        # These will be populated in process_weights_after_loading
        layer.compressed_weight = None
        layer.spmm_workspace = None
        layer.spmm_M = input_size
        layer.spmm_K = input_size_per_partition
        layer.spmm_N = output_size_per_partition


    def process_weights_after_loading(self, layer: Module) -> None:
        """Process weights after loading from checkpoint.

        This method prunes the weights to 2:4 sparsity and compresses them
        for efficient sparse computation.
        """
        if not self.sparse_config.enable_sparse:
            logger.info(f"Sparse computation disabled for layer. Using dense fallback.")
            return

        if not HAS_HIPSPARSE:
            logger.warning("hipsparse_py not available. Skipping sparse compression.")
            return

        weight = layer.weight.t().contiguous()
        assert weight.shape == (layer.spmm_K, layer.spmm_N)

        logger.info(f"Compressing weight with shape {weight.shape} to 2:4 sparsity")

        try:
            # Get current CUDA stream
            stream = torch.cuda.current_stream().cuda_stream

            # Prune to 2:4 sparsity pattern
            weight_t = weight
            pruned_weight = hipsparse_py.spmm_prune(layer.spmm_M , layer.spmm_K, layer.spmm_N, weight.dtype, weight_t, stream)

            # Compress the pruned weight
            compressed_weight = hipsparse_py.spmm_compress(layer.spmm_M , layer.spmm_K, layer.spmm_N, weight.dtype, pruned_weight, stream)

            # Get workspace size and allocate workspace
            workspace_size = hipsparse_py.spmm_get_workspace_size(layer.spmm_M , layer.spmm_K, layer.spmm_N, weight.dtype)
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device=weight.device)

            # Store compressed representation and dimensions
            layer.compressed_weight = compressed_weight
            layer.spmm_workspace = workspace

            # Allocate output tensor
            layer.output = torch.zeros(
                layer.spmm_M,
                layer.spmm_N,
                dtype=weight.dtype,
                device=weight.dtype
            )
            # Calculate sparsity statistics
            total_elements = weight.numel()
            nonzero_elements = pruned_weight.count_nonzero().item()
            actual_sparsity = 1.0 - (nonzero_elements / total_elements)

            logger.info(
                f"Weight compressed: shape={weight.shape}, "
                f"sparsity={actual_sparsity:.2%}, "
                f"compression_ratio={total_elements / nonzero_elements:.2f}x"
            )

        except Exception as e:
            logger.error(f"Failed to compress weight: {e}")
            logger.warning("Falling back to dense computation for this layer")
            layer.compressed_weight = None
            layer.spmm_workspace = None
            layer.spmm_M = None
            layer.spmm_K = None
            layer.spmm_N = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the sparse linear transformation.

        Computes: output = x @ weight.T + bias

        If sparse computation is available, uses the compressed weight
        and hipSPARSELt. Otherwise, falls back to dense computation.
        """
        # Check if we can use sparse computation
        use_sparse = (
            self.sparse_config.enable_sparse
            and layer.compressed_weight is not None
            and layer.spmm_workspace is not None
            and layer.spmm_M is not None
        )

        if use_sparse:


            # Get current CUDA stream
            stream = torch.cuda.current_stream().cuda_stream
            # Run sparse matmul: compressed_weight @ x_i
            torch.ops.vllm.hipsarselt_spmm(
                layer.spmm_M,
                layer.spmm_K,
                layer.spmm_N,
                x.dtype,
                layer.compressed_weight,
                x,
                layer.output,
                layer.spmm_workspace,
                1.0,
                0.0,
                stream
            )

            if bias is not None:
                output = output + bias

            return output

        # Dense fallback path
        output = torch.matmul(x, layer.weight.t())
        if bias is not None:
            output = output + bias
        return output


class Sparse24LinearMethod(SparseLinearMethod):
    """2:4 structured sparse linear method.

    This is a convenience class that explicitly uses 2:4 sparsity.
    """

    def __init__(self, enable_sparse: bool = True):
        config = SparseConfig(sparsity_type="2:4", enable_sparse=enable_sparse)
        super().__init__(config)


class SparseQuantizationConfig(QuantizationConfig):
    """Quantization config wrapper for sparsity.

    This allows sparsity to be used in the same way as quantization
    in vLLM's architecture.
    """

    def __init__(self, sparse_config: Optional[SparseConfig] = None):
        super().__init__()
        if sparse_config is None:
            sparse_config = SparseConfig(sparsity_type="2:4", enable_sparse=True)
        self.sparse_config = sparse_config

    @classmethod
    def get_name(cls) -> str:
        return "sparse"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Requires hipSPARSELt support
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SparseQuantizationConfig":
        sparse_config = SparseConfig(
            sparsity_type=config.get("sparsity_type", "2:4"),
            enable_sparse=config.get("enable_sparse", True),
        )
        return cls(sparse_config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str = "",
    ) -> LinearMethodBase:
        """Get the quantization method for a layer.

        Returns UnquantizedLinearMethod for embedding layers to skip sparse quantization.
        Only applies sparsity to linear layers (attention, MLP).
        """
        # Check if this is an embedding layer - use unquantized method
        layer_class_name = layer.__class__.__name__
        if isinstance(layer, LinearBase):
            return SparseLinearMethod(self.sparse_config)
        
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        return None
    
    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: Optional[str],
    ) -> Optional[str]:
        # Don't override quantization method
        return None
