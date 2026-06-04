import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quantization: str | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quantization = quantization
        if self.quantization == "w8a16":
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.int8),
                requires_grad=False,
            )
            self.register_buffer("weight_scale", torch.empty(output_size, dtype=torch.float32))
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def load_weight(self, loaded_weight: torch.Tensor, output_offset: int = 0):
        if self.quantization == "w8a16":
            loaded_weight = loaded_weight.float()
            scales = loaded_weight.abs().amax(dim=1).clamp_min(1e-6) / 127.0
            qweight = torch.round(loaded_weight / scales[:, None]).clamp(-127, 127).to(torch.int8)
            output_size = qweight.size(0)
            self.weight.data.narrow(0, output_offset, output_size).copy_(qweight)
            self.weight_scale.data.narrow(0, output_offset, output_size).copy_(scales)
        else:
            self.weight.data.narrow(0, output_offset, loaded_weight.size(0)).copy_(loaded_weight)

    def apply_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantization == "w8a16":
            weight = self.weight.float().mul_(self.weight_scale[:, None]).to(x.dtype)
            return F.linear(x, weight, self.bias)
        return F.linear(x, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization: str | None = None,
    ):
        super().__init__(input_size, output_size, bias, quantization=quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if param.ndim == 1:
            param.data.copy_(loaded_weight)
        else:
            self.load_weight(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_weight(x)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization: str | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if param.ndim == 1:
            param_data = param.data
            shard_size = param_data.size(self.tp_dim)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
            param_data.copy_(loaded_weight)
            return
        shard_size = param.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        self.load_weight(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_weight(x)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quantization: str | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        self.load_weight(loaded_weight, shard_offset)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quantization: str | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        if param.ndim == 1:
            param.data.narrow(self.tp_dim, shard_offset, shard_size).copy_(loaded_weight)
        else:
            self.load_weight(loaded_weight, shard_offset)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization: str | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        self.load_weight(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantization == "w8a16":
            weight = self.weight.float().mul_(self.weight_scale[:, None]).to(x.dtype)
            y = F.linear(x, weight, self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
