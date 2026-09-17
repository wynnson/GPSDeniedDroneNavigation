import torch

from pathlib import Path
from omegaconf.dictconfig import DictConfig

from src.utils.config import load_config
from src.utils.decorators import performance


@performance
def export_model_onnx(config: DictConfig):
    if config.model.source == "torch_hub":
        if not config.model.repo or not config.model.name:
            raise ValueError("Missing torch hub configuration")

        output_path = Path("models") / f"{config.model.name}.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = torch.hub.load(config.model.repo, config.model.name)

        # sample "image" to create computation graph
        dummy_img = torch.randn(1, 3, 518, 518)

        # dynamic dimension variable
        batch = torch.export.Dim("batch")

        torch.onnx.export(
            model,
            (dummy_img,),
            str(output_path),
            input_names=["image"],
            output_names=["embedding"],

            # accept dynamic batches
            dynamic_shapes=(
                ({0: batch},),
            ),
            dynamo=True
        )


if __name__ == "__main__":
    config_path = Path("src/config/default.yaml")
    config = load_config(config_path)
    export_model_onnx(config)