from pathlib import Path

from two_d_screw_hole_od.config import download_target_image
from two_d_screw_hole_od.experiment import (
    run_zero_shot_experiment,
    save_report,
)


def main() -> None:
    image_path = download_target_image(Path("artifacts/printables_screw_hole.webp"))
    report = run_zero_shot_experiment(image_path)
    save_report(report, Path("artifacts/model_experiment_report.json"))
    print("Wrote artifacts/model_experiment_report.json")
