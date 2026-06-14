"""Mark unreliable person attributes as unknown (-1).

Dataset-specific rules:
  - helmet: keep helmet, mark smoking as no-smoking
  - smoking: ignore helmet, force smoking=1
  - other datasets: ignore helmet, mark smoking as no-smoking
"""

from pathlib import Path

RULES = {
    "person": ("-1", "0"),
    "helmet": (None, "0"),
    "fire_smoke": ("-1", "0"),
    "smoking": ("-1", "1"),
    "water_leak": ("-1", "0"),
}


def update_file(path: Path, helmet_value, smoke_value) -> int:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    changed = 0
    out = []

    for line in lines:
        parts = line.split()
        if len(parts) >= 58 and parts[0] == "0":
            before = (parts[-2], parts[-1])
            if helmet_value is not None:
                parts[-2] = helmet_value
            if smoke_value is not None:
                parts[-1] = smoke_value
            changed += before != (parts[-2], parts[-1])
            line = " ".join(parts)
        out.append(line)

    if changed:
        path.write_text("\n".join(out) + ("\n" if lines else ""), encoding="utf-8")
    return changed


def main():
    root = Path("data/processed")
    total_files = 0
    total_rows = 0

    for dataset, (helmet_value, smoke_value) in RULES.items():
        label_dir = root / dataset / "labels"
        dataset_files = 0
        dataset_rows = 0
        for path in sorted(label_dir.glob("*.txt")):
            changed = update_file(path, helmet_value, smoke_value)
            if changed:
                dataset_files += 1
                dataset_rows += changed
        total_files += dataset_files
        total_rows += dataset_rows
        print(f"{dataset}: changed_files={dataset_files} changed_rows={dataset_rows}")

    print(f"total: changed_files={total_files} changed_rows={total_rows}")


if __name__ == "__main__":
    main()

