import argparse
import zipfile
from pathlib import Path


def unzip(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the HW6 local scoring directory.")
    parser.add_argument("--scoring_zip", default="scoring_program.zip")
    parser.add_argument("--reference_zip", default="hw6_reference.zip")
    parser.add_argument("--output_dir", default="scoring_program")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    unzip(root / args.scoring_zip, output_dir)
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    unzip(root / args.reference_zip, input_dir / "ref")
    (input_dir / "res").mkdir(parents=True, exist_ok=True)
    print(f"Prepared {output_dir}")


if __name__ == "__main__":
    main()

