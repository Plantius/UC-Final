import argparse

import numpy as np
import s3fs
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.training.trainers import load_checkpoint
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

            self.lama = load_checkpoint(
                "/local/s3322637/data/big-lama",
                map_location=self.device,
                strict=False,
            ).eval()

            # self.inpaint_pipe.enable_model_cpu_offload()
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

    def lama_inpaint_batch(self, images, masks):
        imgs, msks = [], []

        for img, msk in zip(images, masks):
            img = np.array(img).astype(np.float32) / 255.0
            msk = (np.array(msk) > 0).astype(np.float32)

            imgs.append(torch.from_numpy(img).permute(2, 0, 1))
            msks.append(torch.from_numpy(msk).unsqueeze(0))

        imgs = torch.stack(imgs).to(self.device)
        msks = torch.stack(msks).to(self.device)

        imgs = pad_tensor_to_modulo(imgs, 8)
        msks = pad_tensor_to_modulo(msks, 8)

        with torch.no_grad():
            out = self.lama(imgs, msks)

        results = []
        for o in out:
            o = (o.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            results.append(Image.fromarray(o))

        return results

    def inpaint(self, image: Image.Image, mask: Image.Image, output_path: str):
        image, orig_size = pad_to_multiple_pil(image, 512)
        mask, _ = pad_to_multiple_pil(mask, 512)

        result = image.copy()
        tiles = tile_image_and_mask(image, mask)

        jobs = [(x, y, i, m) for x, y, i, m in tiles if has_cloud(m)]

        print(f"Tiles to inpaint: {len(jobs)}")

        pbar = tqdm.tqdm(total=len(jobs), desc="Inpainting tiles")

        for i in range(0, len(jobs), self.batch_size):
            batch = jobs[i : i + self.batch_size]
            imgs = [b[2] for b in batch]
            msks = [b[3] for b in batch]

            outs = self.lama_inpaint_batch(imgs, msks)

            for (x, y, _, _), out in zip(batch, outs):
                result.paste(out, (x, y))
                pbar.update(1)

        pbar.close()

        w, h = orig_size
        result.crop((0, 0, w, h)).save(output_path)
        print(f"Saved: {output_path}")

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
