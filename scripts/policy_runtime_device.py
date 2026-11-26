"""Verify the explicitly selected Torch device inside an admitted policy worker."""

import torch


def verify_runtime(device):
    if device == "cpu":
        return {"device": "cpu", "torch_version": torch.__version__}
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"The requested {device} device is unavailable in the installed Torch runtime")
    probe = torch.ones((2, 2), device=device, dtype=torch.float32)
    if (probe @ probe)[0, 0].item() != 2.:
        raise RuntimeError("The installed CUDA runtime did not execute the policy precision probe")
    free, total = torch.cuda.mem_get_info()
    return {"device": str(probe.device), "device_name": torch.cuda.get_device_name(probe.device),
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "cuda_free_before_model_bytes": free, "cuda_total_bytes": total}
