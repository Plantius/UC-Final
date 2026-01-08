import argparse

import numpy as np
import s3fs
import torch
import torch.nn.functional as F
from PIL import Image
from src.diffusers.pipelines import StableDiffusionInpaintPipeline
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


def argparser():
    parser = argparse.ArgumentParser(description="InpaintCresi Command Line Interface")
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
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps for the diffusion process",
    )
    parser.add_argument(
        "--image-s3",
        type=str,
        default="example_data/satellite_image.png",
        help="Path to the input image for inpainting",
    )
    return parser.parse_args()


class InpaintCresi:
    def __init__(
        self,
        fs: s3fs.S3FileSystem,
        img_size_x: int,
        img_size_y: int,
        num_inference_steps: int,
        uername: str = "s3322637",
    ) -> None:
        self.mask_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.inpaint_model_name = "stable-diffusion-v1-5/stable-diffusion-inpainting"
        self.username = uername

        self.fs = fs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = 7.5

        self.init_models()

    def init_models(self):
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.mask_model_name,
                cache_dir="/local/{self.username}/.cache/",
            )
            self.model = (
                AutoModelForSemanticSegmentation.from_pretrained(
                    self.mask_model_name,
                    cache_dir="/local/{self.username}/.cache/",
                )
                .to(self.device)
                .eval()
            )

            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.inpaint_model_name,
                cache_dir=f"/local/{self.username}/.cache/",
            ).to(self.device)

            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def load_s3_image(self, s3_path: str) -> Image.Image:
        with self.fs.open(s3_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = img.resize((self.img_size_x, self.img_size_y))
        return img

    def detect_cloud_mask(self, image: Image.Image) -> Image.Image:
        # Preprocess
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        upsampled_logits = F.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy()
        cloud_mask = ((pred_seg == 1) | (pred_seg == 4)).astype(np.uint8)

        return Image.fromarray(cloud_mask).resize((self.img_size_x, self.img_size_y))

    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, output_path: str
    ):
        result = self.inpaint_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.img_size_y,
            width=self.img_size_x,
        ).images[0]
        result.save(output_path)
        print(f"Inpainted image saved to {output_path}")

    def cresi(self, data):
        pass


def main(args: argparse.Namespace):
    fs = s3fs.S3FileSystem(anon=True)
    processor = InpaintCresi(
        fs,
        args.img_size_x,
        args.img_size_y,
        args.num_inference_steps,
    )
    img = processor.load_s3_image(args.image_s3)
    mask = processor.detect_cloud_mask(img)
    img.save("original.png")
    mask.save("mask.png")

    prompt = "A satellite image of a city with buildings and roads, clouds removed realistically"
    output = processor.inpaint(img, mask, prompt, "2inpainted_image.png")
    output = processor.cresi(output)


if __name__ == "__main__":
    args = argparser()
    main(args)
