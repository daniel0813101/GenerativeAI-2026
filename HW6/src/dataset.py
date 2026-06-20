import csv
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import ANIMALS, IMAGE_SIZE, OBJECTS


ANIMAL_TO_ID = {name: idx for idx, name in enumerate(ANIMALS)}
OBJECT_TO_ID = {name: idx for idx, name in enumerate(OBJECTS)}


@dataclass(frozen=True)
class Condition:
    image_name: str
    animal: str
    object: str
    prompt: str

    @property
    def animal_id(self) -> int:
        return ANIMAL_TO_ID[self.animal]

    @property
    def object_id(self) -> int:
        return OBJECT_TO_ID[self.object]


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def prompt_from_labels(animal: str, obj: str) -> str:
    return f"a {animal} and a {obj}"


def read_conditions(path: str | Path) -> list[Condition]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for key, item in data.items():
            image_name = item.get("image_name", f"{key}.png")
            prompt = item.get("text_prompt", item.get("prompt", ""))
            animal = normalize_label(item.get("animal", _infer_from_prompt(prompt, ANIMALS)))
            obj = normalize_label(item.get("object", _infer_from_prompt(prompt, OBJECTS)))
            rows.append(Condition(image_name=image_name, animal=animal, object=obj, prompt=prompt or prompt_from_labels(animal, obj)))
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            image_name = row.get("id") or row.get("image_name") or row.get("filename") or row.get("file_name")
            if image_name is None:
                raise ValueError(f"{path} must contain an id/image_name/filename column")
            animal = normalize_label(row.get("animal", ""))
            obj = normalize_label(row.get("object", ""))
            prompt = row.get("prompt") or row.get("text_prompt") or ""
            if not animal:
                animal = normalize_label(_infer_from_prompt(prompt, ANIMALS))
            if not obj:
                obj = normalize_label(_infer_from_prompt(prompt, OBJECTS))
            if animal not in ANIMAL_TO_ID or obj not in OBJECT_TO_ID:
                raise ValueError(f"Unknown labels in {path}: animal={animal!r}, object={obj!r}")
            rows.append(Condition(image_name=image_name, animal=animal, object=obj, prompt=prompt or prompt_from_labels(animal, obj)))
        return rows


def _infer_from_prompt(prompt: str, choices: list[str]) -> str:
    prompt = normalize_label(prompt)
    for choice in choices:
        if choice in prompt:
            return choice
    raise ValueError(f"Could not infer label from prompt: {prompt!r}")


class BrainrotDataset(Dataset):
    def __init__(self, image_dir: str | Path, metadata_path: str | Path, image_size: int = IMAGE_SIZE, augment: bool = True):
        self.image_dir = Path(image_dir)
        self.conditions = read_conditions(metadata_path)
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.01),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.96, 1.04), fill=128),
        ] if augment else []
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        cond = self.conditions[index]
        path = self.image_dir / cond.image_name
        if not path.exists() and path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        image = Image.open(path).convert("RGB")
        return {
            "image": self.transform(image),
            "animal_id": torch.tensor(cond.animal_id, dtype=torch.long),
            "object_id": torch.tensor(cond.object_id, dtype=torch.long),
            "image_name": cond.image_name,
            "prompt": cond.prompt,
        }


def collate_batch(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "animal_id": torch.stack([item["animal_id"] for item in batch]),
        "object_id": torch.stack([item["object_id"] for item in batch]),
        "image_name": [item["image_name"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
    }

