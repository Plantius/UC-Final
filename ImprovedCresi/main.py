import argparse
import os

import torch

from diffusionsat import DiffusionSatPipeline, SatUNet
from diffusionsat.pipeline import StableDiffusionPipeline


def argparser():
    parser = argparse.ArgumentParser(description="InpaintCresi Command Line Interface")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inpaint", "cresi", "improved_cresi"],
        default="improved_cresi",
        help="Choose the operation mode: inpaint, cresi, or improved_cresi",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Input data for processing",
    )
    return parser.parse_args()


class InpaintCresi:
    def __init__(self) -> None:
        self.checkpoint_path = "finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def SatUNet_pipeline(self) -> StableDiffusionPipeline:
        unet = SatUNet.from_pretrained(
            self.checkpoint_path + "checkpoint-100000",
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        pipe: StableDiffusionPipeline = DiffusionSatPipeline.from_pretrained(
            self.checkpoint_path, unet=unet, torch_dtype=torch.float16
        )
        pipe = pipe.to(self.device)
        return pipe

    def inpaint(self, data):
        pipe = self.SatUNet_pipeline()

        caption = "a satellite image of a amusement park in Australia"
        metadata = [925.8798, 345.2111, 411.4541, 0.0000, 308.3333, 166.6667, 354.8387]

        image = pipe(
            caption,
            metadata=metadata,
            num_inference_steps=2,
            guidance_scale=7.5,
            height=128,
            width=128,
        ).images[0]

        image.save("inpainted_image.png")

    def cresi(self, data):
        pass


def main(args: argparse.Namespace):
    processor = InpaintCresi()
    if args.mode == "inpaint":
        output = processor.inpaint(args.data)
    elif args.mode == "cresi":
        output = processor.cresi(args.data)
    elif args.mode == "improved_cresi":
        output = processor.inpaint(args.data)
        output = processor.cresi(output)


if __name__ == "__main__":
    os.environ["HF_HOME"] = "~/.cache/"

    args = argparser()
    main(args)
