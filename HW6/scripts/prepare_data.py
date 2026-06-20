import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HW6 dataset.zip into data/ without running training.")
    parser.add_argument("--dataset_zip", default="dataset.zip")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / args.dataset_zip) as zf:
        zf.extractall(output_dir)
    if (output_dir / "train.csv").exists():
        print(f"Prepared {output_dir / 'train.csv'}")
    if (output_dir / "generate.csv").exists():
        print(f"Prepared {output_dir / 'generate.csv'}")
    print(f"Prepared {output_dir}")


if __name__ == "__main__":
    main()
