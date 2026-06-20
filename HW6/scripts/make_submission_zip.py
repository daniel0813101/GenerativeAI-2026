import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.validate_submission import validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HW6_{student_id}.zip for E3 upload.")
    parser.add_argument("--student_id", required=True)
    parser.add_argument("--generated_dir", default="scoring_program/input/res")
    parser.add_argument("--conditions", default="data/generate.csv")
    parser.add_argument("--checkpoint", default="checkpoints/model_ema.pth")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    package_dir = root / "submission" / f"HW6_{args.student_id}"
    images_dir = package_dir / "generated_images"
    scripts_dir = package_dir / "scripts"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    images_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    for path in sorted((root / args.generated_dir).glob("*.png")):
        shutil.copy2(path, images_dir / path.name)
    validate(images_dir, root / args.conditions, 64)
    shutil.copytree(root / "scripts", scripts_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "src", package_dir / "src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(root / args.checkpoint, package_dir / "model.pth")
    shutil.copy2(root / "requirements.txt", package_dir / "requirements.txt")
    shutil.copy2(root / "README.md", package_dir / "README.md")

    zip_path = root / "submission" / f"HW6_{args.student_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package_dir.parent))
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()

