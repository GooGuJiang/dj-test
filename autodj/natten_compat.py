from __future__ import annotations

import importlib
import importlib.metadata
import importlib.machinery
import sys
import types
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NattenCompatStatus:
    backend: str
    version: str
    native: bool
    message: str


def _window_indices_1d(length: int, kernel_size: int, dilation: int, device: torch.device) -> torch.Tensor:
    kernel_size = int(kernel_size)
    dilation = int(dilation)
    span = (kernel_size - 1) * dilation + 1
    if length < span:
        raise ValueError(
            f"NATTEN fallback requires length >= effective window: {length} < {span}"
        )
    positions = torch.arange(length, device=device)
    start = positions - (kernel_size // 2) * dilation
    start = torch.clamp(start, min=0, max=length - span)
    offsets = torch.arange(kernel_size, device=device) * dilation
    return start[:, None] + offsets[None, :]


def _rpb_indices_1d(indices: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    positions = torch.arange(indices.shape[0], device=indices.device)[:, None]
    relative = torch.div(indices - positions, max(1, int(dilation)), rounding_mode="trunc")
    return torch.clamp(relative + int(kernel_size) - 1, 0, 2 * int(kernel_size) - 2).long()


def natten1dqkrpb(
    query: torch.Tensor,
    key: torch.Tensor,
    rpb: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """Pure-PyTorch compatibility implementation of the legacy NATTEN 1-D QK op."""
    length = int(query.shape[-2])
    indices = _window_indices_1d(length, kernel_size, dilation, query.device)
    key_windows = key[:, :, indices, :]
    scores = torch.sum(query.unsqueeze(-2) * key_windows, dim=-1)
    if rpb is not None:
        bias_indices = _rpb_indices_1d(indices, kernel_size, dilation)
        scores = scores + rpb[:, bias_indices].unsqueeze(0)
    return scores


def natten1dav(
    attention: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """Pure-PyTorch compatibility implementation of the legacy NATTEN 1-D AV op."""
    length = int(value.shape[-2])
    indices = _window_indices_1d(length, kernel_size, dilation, value.device)
    windows = value[:, :, indices, :]
    return torch.sum(attention.unsqueeze(-1) * windows, dim=-2)


def _window_indices_axis(length: int, kernel_size: int, dilation: int, device: torch.device) -> torch.Tensor:
    return _window_indices_1d(length, kernel_size, dilation, device)


def natten2dqkrpb(
    query: torch.Tensor,
    key: torch.Tensor,
    rpb: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """Pure-PyTorch compatibility implementation of the legacy NATTEN 2-D QK op."""
    height = int(query.shape[-3])
    width = int(query.shape[-2])
    y_indices = _window_indices_axis(height, kernel_size, dilation, query.device)
    x_indices = _window_indices_axis(width, kernel_size, dilation, query.device)
    y_rpb = _rpb_indices_1d(y_indices, kernel_size, dilation)
    x_rpb = _rpb_indices_1d(x_indices, kernel_size, dilation)

    parts: list[torch.Tensor] = []
    for ky in range(int(kernel_size)):
        y = y_indices[:, ky]
        for kx in range(int(kernel_size)):
            x = x_indices[:, kx]
            patch = key[:, :, y[:, None], x[None, :], :]
            score = torch.sum(query * patch, dim=-1)
            if rpb is not None:
                bias = rpb[:, y_rpb[:, ky][:, None], x_rpb[:, kx][None, :]]
                score = score + bias.unsqueeze(0)
            parts.append(score)
    return torch.stack(parts, dim=-1)


def natten2dav(
    attention: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """Pure-PyTorch compatibility implementation of the legacy NATTEN 2-D AV op."""
    height = int(value.shape[-3])
    width = int(value.shape[-2])
    y_indices = _window_indices_axis(height, kernel_size, dilation, value.device)
    x_indices = _window_indices_axis(width, kernel_size, dilation, value.device)
    output = torch.zeros_like(value)
    index = 0
    for ky in range(int(kernel_size)):
        y = y_indices[:, ky]
        for kx in range(int(kernel_size)):
            x = x_indices[:, kx]
            patch = value[:, :, y[:, None], x[None, :], :]
            output = output + attention[..., index].unsqueeze(-1) * patch
            index += 1
    return output


def _install_torch_module() -> NattenCompatStatus:
    functional = types.ModuleType("natten.functional")
    functional.natten1dqkrpb = natten1dqkrpb
    functional.natten1dav = natten1dav
    functional.natten2dqkrpb = natten2dqkrpb
    functional.natten2dav = natten2dav
    # Short aliases used by some intermediate NATTEN releases.
    functional.na1d_qk = natten1dqkrpb
    functional.na1d_av = natten1dav
    functional.na2d_qk = natten2dqkrpb
    functional.na2d_av = natten2dav

    # ``transformers`` and other libraries use ``importlib.util.find_spec`` to
    # inspect optional packages. A synthetic module with ``__spec__ = None``
    # makes find_spec raise ``ValueError: natten.__spec__ is None``. Give both
    # injected modules proper specs so All-In-One and CUE-DETR can coexist in
    # the same Python process.
    functional.__package__ = "natten"
    functional.__spec__ = importlib.machinery.ModuleSpec(
        name="natten.functional",
        loader=None,
        is_package=False,
    )

    package = types.ModuleType("natten")
    package.functional = functional
    package.__version__ = "autodj-torch-fallback"
    package.__package__ = "natten"
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec(
        name="natten",
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = []
    sys.modules["natten"] = package
    sys.modules["natten.functional"] = functional
    return NattenCompatStatus(
        backend="torch-fallback",
        version="autodj",
        native=False,
        message="使用纯 PyTorch NATTEN 兼容层；可运行但比原生 CUDA 内核慢。",
    )


def ensure_natten_compat(force_torch: bool = False) -> NattenCompatStatus:
    """Ensure All-In-One's legacy NATTEN symbols are importable.

    All-In-One 1.1.0 imports the pre-0.20 function names. New NATTEN releases
    removed those symbols, while very old releases used incompatible signatures.
    For known-compatible 0.15-0.19 releases we keep the native kernels; otherwise
    a deterministic pure-PyTorch implementation is injected before importing
    ``allin1``.
    """
    if force_torch:
        return _install_torch_module()

    try:
        version = importlib.metadata.version("natten")
    except importlib.metadata.PackageNotFoundError:
        return _install_torch_module()

    try:
        major_minor = tuple(int(part) for part in version.split("+")[0].split(".")[:2])
    except ValueError:
        major_minor = (0, 0)

    try:
        functional = importlib.import_module("natten.functional")
    except Exception:
        return _install_torch_module()

    required = ("natten1dqkrpb", "natten1dav", "natten2dqkrpb", "natten2dav")
    has_legacy = all(hasattr(functional, name) for name in required)
    if has_legacy and (0, 15) <= major_minor < (0, 20):
        return NattenCompatStatus(
            backend="native",
            version=version,
            native=True,
            message=f"使用原生 NATTEN {version}。",
        )

    # Preserve the imported package but replace/add exactly the legacy functions
    # that All-In-One 1.1.0 expects.
    functional.natten1dqkrpb = natten1dqkrpb
    functional.natten1dav = natten1dav
    functional.natten2dqkrpb = natten2dqkrpb
    functional.natten2dav = natten2dav
    functional.na1d_qk = natten1dqkrpb
    functional.na1d_av = natten1dav
    functional.na2d_qk = natten2dqkrpb
    functional.na2d_av = natten2dav
    return NattenCompatStatus(
        backend="torch-fallback",
        version=version,
        native=False,
        message=f"NATTEN {version} 与 All-In-One 旧接口不兼容，已使用纯 PyTorch 兼容层。",
    )
