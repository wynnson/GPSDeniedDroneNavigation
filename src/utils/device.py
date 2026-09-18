def get_device() -> str:
    # Hide this so we don't compile with torch for ONNX builds
    # Note: ONNX builds don't need get_device
    import torch

    """Get device"""
    if torch.cuda.is_available():
        device = "cuda"

    elif torch.backends.mps.is_available():
        device = "mps"

    else:
        device = "cpu"

    print(f"Using device: {device}")
    return device
