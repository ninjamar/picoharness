import subprocess
from pathlib import Path


def services_main(args: list[str]) -> None:
    services_dir = Path(__file__).parent / "services"
    compose_file = services_dir / "docker-compose.yml"
    cmd = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "--project-directory",
        str(services_dir),
        *args,
    ]
    subprocess.run(cmd)
