from importlib.resources import files
from pathlib import Path
import argparse

def main():
    
    parser = argparse.ArgumentParser(description="Scaffold minimal project folders")
    parser.add_argument("--root", type=str, default=None, help="Project root directory (default: current directory)")
    
    # Discover available config templates
    config_dir = files("geomoka").joinpath('template').joinpath('config')
    available_configs = sorted([item.name for item in config_dir.iterdir() if item.is_dir()])
    available_configs_str = ", ".join(available_configs)
    
    parser.add_argument("--config", type=str, default='generic', help=f"Template config name options: {available_configs_str}")
    parser.add_argument("--force", action='store_true', help="Overwrite existing config files")
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path.cwd()

    dest_dir = root / 'config' / args.config
    dest_dir.mkdir(parents=True, exist_ok=True)

    src = files("geomoka").joinpath('template').joinpath('config').joinpath(args.config)
    for file in src.iterdir():
        if file.is_file():
            dest = dest_dir / file.name
            if dest.exists() and not args.force:
                print(f"[skip] {dest} already exists (use --force to overwrite)")
                continue
            dest.write_bytes(file.read_bytes())
            print(f"[write] {dest}")

if __name__ == "__main__":
    main()