import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import read_conditions


def validate(image_dir: Path, conditions_path: Path, image_size: int) -> None:
    conditions = read_conditions(conditions_path)
    expected = [item.image_name for item in conditions]
    files = sorted(p.name for p in image_dir.glob("*.png"))
    errors = []
    if len(files) != len(expected):
        errors.append(f"Expected {len(expected)} PNGs, found {len(files)}")
    missing = sorted(set(expected) - set(files))
    extra = sorted(set(files) - set(expected))
    if missing:
        errors.append(f"Missing files: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if extra:
        errors.append(f"Unexpected files: {extra[:10]}{' ...' if len(extra) > 10 else ''}")
    for name in expected:
        path = image_dir / name
        if not path.exists():
            continue
        with Image.open(path) as image:
            if image.mode != "RGB":
                errors.append(f"{name} is {image.mode}, expected RGB")
            if image.size != (image_size, image_size):
                errors.append(f"{name} is {image.size}, expected {(image_size, image_size)}")
        if len(errors) > 20:
            break
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(expected)} PNG files in {image_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HW6 generated image folder.")
    parser.add_argument("--image_dir", default="scoring_program/input/res")
    parser.add_argument("--conditions", default="data/generate.csv")
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    validate(root / args.image_dir, root / args.conditions, args.image_size)


if __name__ == "__main__":
    main()

