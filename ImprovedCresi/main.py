import argparse

import numpy as np
import s3fs
import torch
import torch.nn.functional as F
import tqdm
from diffusers import AutoPipelineForInpainting
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


def argparser():
    parser = argparse.ArgumentParser(description="InpaintCresi Command Line Interface")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing image tiles",
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
        help="Path to the remote input image for inpainting",
    )
    parser.add_argument(
        "--image-local",
        type=str,
        help="Path to the local input image for inpainting",
    )
    return parser.parse_args()


def has_cloud(mask_tile, threshold=0.01):
    mask_np = np.array(mask_tile)
    cloud_fraction = mask_np.mean() / 255.0
    return cloud_fraction > threshold


def tile_image_and_mask(image, mask, tile_size=512):
    tiles = []

    w, h = image.size

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            box = (x, y, x + tile_size, y + tile_size)

            if x + tile_size > w or y + tile_size > h:
                continue

            img_tile = image.crop(box)
            mask_tile = mask.crop(box)

            tiles.append((x, y, img_tile, mask_tile))

    return tiles


def pad_to_multiple(image: Image.Image, multiple: int, fill=0):
    w, h = image.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple

    if image.mode == "RGB":
        padded = Image.new("RGB", (new_w, new_h), (fill, fill, fill))
    else:
        padded = Image.new("L", (new_w, new_h), fill)

    padded.paste(image, (0, 0))
    return padded, (w, h)


class InpaintCresi:
    def __init__(
        self,
        fs: s3fs.S3FileSystem,
        num_inference_steps: int,
        batch_size: int,
        username: str = "s3322637",
    ) -> None:
        self.mask_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.inpaint_model_name = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
        self.username = username
        self.batch_size = batch_size

        self.fs = fs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = 7.5

        self.init_models()

    def init_models(self):
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.mask_model_name,
                cache_dir=f"/local/{self.username}/.cache/",
            )
            self.model = (
                AutoModelForSemanticSegmentation.from_pretrained(
                    self.mask_model_name,
                    cache_dir=f"/local/{self.username}/.cache/",
                )
                .to(self.device)
                .eval()
            )

            self.inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                self.inpaint_model_name,
                cache_dir=f"/local/{self.username}/.cache/",
                torch_dtype=torch.float16,
            ).to(self.device)

            self.inpaint_pipe.enable_model_cpu_offload()
            # self.inpaint_pipe.set_progress_bar_config(disable=True)

            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def load_s3_image(self, s3_path: str) -> Image.Image:
        with self.fs.open(s3_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return img

    def load_image(self, file_path: str) -> Image.Image:
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image from {file_path}: {e}")
            raise
        return img

    def detect_cloud_mask(self, image: Image.Image) -> Image.Image:
        # Preprocess
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        upsampled_logits = F.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy()
        cloud_mask = (
            ~(
                (pred_seg == 1)
                | (pred_seg == 4)
                | (pred_seg == 16)
                | (pred_seg == 9)
                | (pred_seg == 10)
                | (pred_seg == 11)
                | (pred_seg == 26)
            )
        ).astype(np.uint8)
        cloud_mask = cloud_mask * 255
        return Image.fromarray(cloud_mask).convert("L")

    def inpaint_tile(self, img_tile, mask_tile, prompt):
        return self.inpaint_pipe(
            prompt=prompt,
            image=img_tile,
            mask_image=mask_tile,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images[0]

    def inpaint_full_image(self, image, mask, prompt):
        result = image.copy()

        tiles = tile_image_and_mask(image, mask)

        jobs = []
        for i, tile in enumerate(tiles):
            x, y, img_tile, mask_tile = tile
            if has_cloud(mask_tile):
                img_tile.save(f"img_tile_{i}.png")
                mask_tile.save(f"mask_tile_{i}.png")
                jobs.append((x, y, img_tile, mask_tile))

        print(f"Total tiles to inpaint: {len(jobs)}")

        if len(jobs) == 0:
            return result

        pbar = tqdm.tqdm(
            total=int(np.ceil(len(jobs) / self.batch_size)),
            desc="Inpainting tiles",
            unit="batch",
        )

        for i in range(0, len(jobs), self.batch_size):
            batch = jobs[i : i + self.batch_size]

            images = [j[2] for j in batch]

            masks = [j[3] for j in batch]
            prompts = [prompt] * len(images)

            outputs = self.inpaint_pipe(
                prompt=prompts,
                image=images,
                mask_image=masks,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            ).images

            for (x, y, _, _), out_img in zip(batch, outputs):
                result.paste(out_img, (x, y))
                pbar.update(1)

        pbar.close()
        return result

    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, output_path: str
    ):
        padded_image, original_size = pad_to_multiple(image, 512, fill=0)
        padded_mask, _ = pad_to_multiple(mask, 512, fill=0)

        padded_result = self.inpaint_full_image(
            image=padded_image,
            mask=padded_mask,
            prompt=prompt,
        )

        w, h = original_size
        result = padded_result.crop((0, 0, w, h))

        result.save(output_path)
        print(f"Inpainted image saved to {output_path}")

    def cresi(self, data):
        pass


def main(args: argparse.Namespace):
    fs = s3fs.S3FileSystem(anon=True)
    processor = InpaintCresi(
        fs,
        args.num_inference_steps,
        args.batch_size,
    )

    if args.image_local:
        img = processor.load_image(args.image_local)
    elif args.image_s3:
        img = processor.load_s3_image(args.image_s3)
    else:
        raise ValueError("Either --image-local or --image-s3 must be provided.")
    mask = processor.detect_cloud_mask(img)
    img.save("original.png")
    mask.save("mask.png", mode="L")
    print("Original image and mask saved.")

    prompt = "a clear sky satellite image"
    output = processor.inpaint(img, mask, prompt, "inpainted_image.png")
    output = processor.cresi(output)


if __name__ == "__main__":
    args = argparser()
    main(args)
