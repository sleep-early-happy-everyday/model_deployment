"""Summarize the local DiT transformer with a single forward pass."""

import argparse
from pathlib import Path

import torch
from diffusers import DiTTransformer2DModel
from torchinfo import summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # The current DiT model has at most 6 levels of nested modules.
    parser.add_argument(
        "--depth", type=int, choices=range(1, 7), default=6,
        help="Module nesting depth to display (1–6; default: 6)",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = DiTTransformer2DModel.from_pretrained(
        project_dir / "weights" / "transformer",
        torch_dtype=dtype,
        local_files_only=True,
        use_safetensors=False,
    ).to(device).eval()
    config = model.config
    inputs = {
        "hidden_states": torch.zeros(
            1, config.in_channels, config.sample_size, config.sample_size,
            device=device, dtype=dtype,
        ),
        "timestep": torch.tensor([500], device=device, dtype=torch.long),
        "class_labels": torch.tensor([207], device=device, dtype=torch.long),
    }
    with torch.inference_mode():
        stats = summary(
            model,
            input_data=inputs,
            return_dict=False,
            depth=args.depth,
            col_names=("input_size", "output_size", "num_params"),
            row_settings=("var_names",),
            device=device,
            mode="eval",
            verbose=0,
        )

    header = (
        f"DiT transformer | device={device} | dtype={dtype} | batch_size=1\n"
        f"Blocks: {config.num_layers}; heads: {config.num_attention_heads}; "
        f"hidden size: {config.num_attention_heads * config.attention_head_dim}; "
        f"tokens: {(config.sample_size // config.patch_size) ** 2}\n"
        "Scope: one transformer forward pass; excludes VAE, scheduler and CFG batch duplication.\n"
        "Note: torchinfo Mult-Adds and memory estimates are not complete FLOPs or measured peak VRAM.\n"
    )
    report = header + "\n" + str(stats) + "\n"
    output = Path(__file__).resolve().parent / "dit_summary.txt"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
