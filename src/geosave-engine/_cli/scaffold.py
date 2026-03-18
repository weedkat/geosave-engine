import argparse
from pathlib import Path
from importlib.resources import files


def scaffold(root: Path) -> None:
	"""Create the minimal folder structure only."""
	(root / "config").mkdir(parents=True, exist_ok=True)
	(root / "dataset").mkdir(parents=True, exist_ok=True)
	(root / "script").mkdir(parents=True, exist_ok=True)
	(root / "test").mkdir(parents=True, exist_ok=True)

	for dir in ["script", "test"]:
		src = files("geomoka").joinpath("template").joinpath(dir)
		for file in src.iterdir():
			if file.is_file():
				dest = root / dir / file.name
				if not dest.exists():
					dest.write_bytes(file.read_bytes())

def main():
    parser = argparse.ArgumentParser(description="Scaffold minimal project folders")
    parser.add_argument("--root", type=str, default=None, help="Project root directory (default: current directory)")
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path.cwd()
    scaffold(root)
    print("Scaffold created under:", root)


if __name__ == "__main__":
    main()
