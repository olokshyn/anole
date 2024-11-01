import os
import io
import pickle
import random
import logging

import asyncio
import aiofiles
from PIL import Image, ImageFile
from chameleon.inference.chameleon import TokenManager
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
from tqdm import tqdm

from training.constants_training import (
    ANOLE_PATH_TORCH,
    WIKIHOW_DATASET_PATH,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PID %(process)d] %(levelname)s: %(message)s",
)


def set_seed(seed: int):
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for CUDA (if using)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make PyTorch deterministic (this can slow down the computation)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_token_manager(device: str | None = None) -> TokenManager:
    return TokenManager(
        tokenizer_path=(
            ANOLE_PATH_TORCH / "tokenizer" / "text_tokenizer.json"
        ).as_posix(),
        vqgan_cfg_path=(ANOLE_PATH_TORCH / "tokenizer" / "vqgan.yaml").as_posix(),
        vqgan_ckpt_path=(ANOLE_PATH_TORCH / "tokenizer" / "vqgan.ckpt").as_posix(),
        device=device,
    )


class WikiHowDataset:

    def __init__(
        self,
        wikihow_root_path: str,
        metadata_path: str | None = None,
        train_images_path: str | None = None,
        test_images_path: str | None = None,
        train_split: float = 0.8,
        limit: int | None = None,
    ) -> None:
        self.wikihow_root_path = wikihow_root_path
        self.metadata_path = metadata_path or os.path.join(
            wikihow_root_path, "WikihowText_data.json"
        )
        self.train_images_path = train_images_path or os.path.join(
            wikihow_root_path, "wiki_images", "train"
        )
        self.test_images_path = test_images_path or os.path.join(
            wikihow_root_path, "wiki_images", "test"
        )
        self.train_split = train_split
        self.limit = limit

    def load_and_split_metadata(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        meta = pd.read_json(self.metadata_path, lines=True)
        if self.limit is not None:
            meta = meta.head(self.limit)
        train_size = int(len(meta) * self.train_split)
        train_meta = meta.sample(train_size)
        test_meta = meta.drop(train_meta.index)
        return train_meta, test_meta

    def _find_image_path(self, step_id: str) -> str:
        image_path = os.path.join(self.train_images_path, step_id) + ".png"
        if not os.path.exists(image_path):
            image_path = os.path.join(self.test_images_path, step_id) + ".png"
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {step_id}.png not found")
        return image_path

    def load_image(self, step_id: str) -> ImageFile.ImageFile:
        image_path = self._find_image_path(step_id)
        return Image.open(image_path)

    async def async_load_image(self, step_id: str) -> ImageFile.ImageFile:
        image_path = self._find_image_path(step_id)
        async with aiofiles.open(image_path, "rb") as infile:
            image_bytes = await infile.read()
        return Image.open(io.BytesIO(image_bytes))


class WikiHowTokenizer:

    def __init__(self, dataset: WikiHowDataset, device: str | None = None) -> None:
        self.dataset = dataset
        self.token_manager = build_token_manager(device=device)

    def tokenize_text_batch(self, inputs: list[str]) -> list[list[int]]:
        tokenized = self.token_manager.tokenizer.encode_batch(inputs)
        return [x.ids for x in tokenized]

    def tokenize_image_batch(
        self, inputs: list[ImageFile.ImageFile]
    ) -> list[list[int]]:
        tokenized = [self.token_manager.tokenize_image(x) for x in inputs]
        return tokenized

    async def _process_method(
        self, output_dir: str, goal: str, goal_desc: str, method: dict
    ) -> str:
        file_path = os.path.join(output_dir, f"{method['method_id']}.pickle")
        if os.path.exists(file_path):
            return file_path
        header = f"""{goal}?
{goal_desc + os.linesep if goal_desc else ''}
Method: {method['name']}
"""
        all_text = [header]
        all_images = []
        for step_idx, step in enumerate(method["steps"]):
            step_text = f"""
Step {step_idx + 1}: {step['headline']}
{step['description'] + os.linesep if step['description'] else ''}"""
            all_text.append(step_text)
            image = await self.dataset.async_load_image(step["step_id"])
            all_images.append(image)

        all_tokenized_text = self.tokenize_text_batch(all_text)
        all_tokenized_images = self.tokenize_image_batch(all_images)

        all_tokens = list(all_tokenized_text[0])
        for text_tokens, image_tokens in zip(
            all_tokenized_text[1:], all_tokenized_images
        ):
            all_tokens.extend(text_tokens)
            all_tokens.extend(image_tokens)

        async with aiofiles.open(file_path, "wb") as outfile:
            await outfile.write(pickle.dumps(all_tokens))
        async with aiofiles.open(
            os.path.splitext(file_path)[0] + ".txt", "w"
        ) as outfile:
            await outfile.write("".join(all_text))
        return file_path

    async def _process_goal(
        self,
        output_dir: str,
        file_id: int,
        goal: str,
        goal_desc: str,
        methods: list[dict],
    ) -> list[str]:
        goal_dir = os.path.join(output_dir, str(file_id))
        os.makedirs(goal_dir, exist_ok=True)
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._process_method(goal_dir, goal, goal_desc, method))
                for method in methods
            ]
            all_paths = await asyncio.gather(*tasks)

        return all_paths

    async def async_process_batch(
        self, batch: pd.DataFrame, output_dir: str
    ) -> list[str]:
        pid = os.getpid()
        all_paths = []
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Worker {pid}"):
            all_paths.extend(
                await self._process_goal(
                    output_dir,
                    row["file_id"],
                    row["goal"],
                    row["goal_description"],
                    row["methods"],
                )
            )
        return all_paths


def process_batch(
    dataset: WikiHowDataset,
    batch: pd.DataFrame,
    output_dir: str,
    device: str | None = None,
) -> list[str]:
    if device:
        torch.cuda.set_device(device)
    tokenizer = WikiHowTokenizer(dataset, device=device)
    return asyncio.run(tokenizer.async_process_batch(batch, output_dir))


def process_wikihow_split(
    dataset: WikiHowDataset,
    meta: pd.DataFrame,
    split: str,
    output_dir: str,
    n_workers: int,
    use_gpus: tuple[int, ...],
) -> list[str]:
    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    batch_size = int(np.ceil(len(meta) / n_workers))
    logging.info(
        f"Splitting {len(meta)} records between {n_workers} workers, {batch_size} records per worker"
    )
    batches = [
        meta[i : i + batch_size] for i in range(0, len(meta), batch_size)  # noqa: E203
    ]

    # results = map(lambda x: process_batch(dataset, x, output_dir), batches)

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.starmap(
            process_batch,
            [
                (dataset, batch, output_dir, f"cuda:{index % len(use_gpus)}")
                for index, batch in enumerate(batches)
            ],
        )
    all_paths = [path for paths in results for path in paths]
    return all_paths


def process_wikihow(
    dataset: WikiHowDataset,
    *,
    output_dir: str,
    n_workers: int = -1,
    max_workers: int = 10,
    use_gpus: tuple[int, ...] = (0,),
):
    if n_workers == -1:
        n_workers = max((os.cpu_count() or 1) - 1, 1)
        n_workers = min(n_workers, max_workers)

    train_meta, test_meta = dataset.load_and_split_metadata()
    process_wikihow_split(dataset, train_meta, "train", output_dir, n_workers, use_gpus)
    process_wikihow_split(dataset, test_meta, "test", output_dir, n_workers, use_gpus)


if __name__ == "__main__":
    set_seed(42)
    multiprocessing.set_start_method("spawn")

    dataset = WikiHowDataset(wikihow_root_path=str(WIKIHOW_DATASET_PATH))

    process_wikihow(
        dataset,
        output_dir="wikihow_tokenized",
        max_workers=30,
        use_gpus=(0, 1),
    )
