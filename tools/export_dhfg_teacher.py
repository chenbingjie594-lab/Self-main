"""Export an Ultralytics detector as a dependency-free DHFG TorchScript teacher."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


class DetectorFeatureExport(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.layers = detector.model
        self.save = set(int(index) for index in detector.save)

    def forward(self, x):
        saved = []
        for layer_index, layer in enumerate(self.layers):
            source = layer.f
            if source != -1:
                if isinstance(source, int):
                    x = saved[source]
                else:
                    x = [x if index == -1 else saved[index] for index in source]
            if layer_index == len(self.layers) - 1:
                feature_maps = x if isinstance(x, (tuple, list)) else [x]
                class_logits = [
                    layer.cv3[index](feature)
                    for index, feature in enumerate(feature_maps)
                ]
                return tuple(feature_maps + class_logits)
            x = layer(x)
            saved.append(x if layer_index in self.save else None)
        raise RuntimeError("Detector has no final prediction head")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Real-only YOLO best.pt")
    parser.add_argument("--output", required=True, help="Output .torchscript file")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights).model.to(args.device).float().eval()
    wrapper = DetectorFeatureExport(model).to(args.device).eval()
    example = torch.zeros(1, 3, args.input_size, args.input_size, device=args.device)
    with torch.no_grad():
        outputs = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced_outputs = traced(example)
    if len(outputs) % 2 or len(outputs) != len(traced_outputs):
        raise RuntimeError("Unexpected detector feature output layout")

    names = model.names
    if isinstance(names, dict):
        class_names = [str(names[index]) for index in sorted(names)]
    else:
        class_names = [str(name) for name in names]
    metadata = {
        "format_version": 1,
        "input_size": args.input_size,
        "num_scales": len(outputs) // 2,
        "class_names": class_names,
        "source_weights": str(Path(args.weights).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        traced.cpu(),
        str(output),
        _extra_files={"dhfg.json": json.dumps(metadata)},
    )
    print(f"Exported DHFG teacher: {output.resolve()}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

