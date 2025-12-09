import argparse
import os
from tkinter import Image

import torch

from diffusionsat.data_util import metadata_normalize
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
        "--unet-model-path",
        type=str,
        default="finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64/",
        help="Path to the UNet model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Input data for processing",
    )
    parser.add_argument(
        "--img-size-x",
        type=int,
        default=512,
        help="Width of the output image",
    )
    parser.add_argument(
        "--img-size-y",
        type=int,
        default=512,
        help="Height of the output image",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="example_data/satellite_image.png",
        help="Path to the input image for inpainting",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="example_data/mask.png",
        help="Path to the input mask for inpainting",
    )
    return parser.parse_args()


class InpaintCresi:
    def __init__(self, checkpoint_path: str, img_size_x: int, img_size_y: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y    
        self.num_inference_steps = 10
        self.guidance_scale = 7.5

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

    def inpaint(self, image_path: str, mask_path: str):
        pipe = self.SatUNet_pipeline()

        init_image = Image.open(image_path).convert("RGB").resize(
            (self.img_size_x, self.img_size_y)
        )
        mask_image = Image.open(mask_path).convert("L").resize(
            (self.img_size_x, self.img_size_y)
        )
        
        caption = "a cloudy satellite image of a city with buildings and roads in Paris, France"
        # metadata: [longitude, latitude, gsd, cloud cover, year, month, day]
        metadata = metadata_normalize([76.5712666476, 28.6965307997, 0.929417550564, 0.0765712666476, 2015, 2, 27]).tolist()

        
        image = pipe(
            caption,
            image=init_image,
            mask_image=mask_image,
            metadata=metadata,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images[0]

        image.save("inpainted_image.png")

    def cresi(self, data):
        pass


def main(args: argparse.Namespace):
    processor = InpaintCresi(args.unet_model_path, args.img_size_x, args.img_size_y)
    output = processor.inpaint(args.image, args.mask)
    output = processor.cresi(output)


if __name__ == "__main__":
    os.environ["HF_HOME"] = "~/.cache/"

    args = argparser()
    main(args)
