from pathlib import Path
from dotenv import load_dotenv


def pytest_configure(config):
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
