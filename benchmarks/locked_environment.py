"""Export this repository's Pixi Linux lock without introducing a new solve."""
import argparse
import re
from pathlib import Path


def export(lock_path, output_dir):
    text = Path(lock_path).read_text()
    environment = text.split("\npackages:\n", 1)[0]
    conda = re.findall(r"^      - conda: (https://\S+)$", environment, re.M)
    pip = re.findall(r"^      - pypi: (https://\S+)$", environment, re.M)
    if not conda or not pip:
        raise ValueError("No locked Linux packages found")
    if any("/win-" in url or "/osx-" in url for url in conda):
        raise ValueError("Expected only the Linux lock environment")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Preserve lock checksums; explicit URLs alone do not supply libmamba with
    # the metadata needed to verify cached package integrity.
    checksums = {}
    for match in re.finditer(
        r"^- conda: (https://\S+)\n(.*?)(?=^- (?:conda|pypi):|\Z)",
        text, re.M | re.S,
    ):
        digest = re.search(r"^  md5: ([0-9a-f]{32})$", match[2], re.M)
        if digest:
            checksums[match[1]] = digest[1]
    missing = set(conda) - checksums.keys()
    if missing:
        raise ValueError(f"Missing locked package checksums: {sorted(missing)}")
    explicit = [url + "#" + checksums[url] for url in conda]
    (output / "conda-explicit.txt").write_text("@EXPLICIT\n" + "\n".join(explicit) + "\n")
    (output / "pip-locked.txt").write_text("\n".join(pip) + "\n")
    print(f"Exported {len(conda)} Conda packages and {len(pip)} PyPI artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lock")
    parser.add_argument("output")
    args = parser.parse_args()
    export(args.lock, args.output)
