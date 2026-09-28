"""Generate one ImageNet class-conditioned image using local DiT weights."""

import argparse
from pathlib import Path

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-id", type=int, default=207, help="ImageNet class ID (0–999); default: golden retriever")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps (1–1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/golden_retriever.png"))
    args = parser.parse_args()
    if not 0 <= args.class_id <= 999:
        parser.error("--class-id must be between 0 and 999")
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be between 1 and 1000")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the NVIDIA driver and PyTorch installation")

    weights = Path(__file__).resolve().parent / "weights"
    pipe = DiTPipeline.from_pretrained(
        weights,
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        class_labels=[args.class_id],
        num_inference_steps=args.steps,
        generator=generator,
    ).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved {image.width}x{image.height} image to {args.output.resolve()}")


if __name__ == "__main__":
    main()
