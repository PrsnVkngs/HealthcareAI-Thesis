import shutil
import subprocess
import os
from pathlib import Path
import cudf
from concurrent.futures import ProcessPoolExecutor


def create_config(path, only_dead=False):
    """Creates a temporary synthea.properties to force the generator state."""
    content = [
        "exporter.csv.export = true",
        "exporter.fhir.export = false",
        "exporter.hospital.csv.export = false",
        "exporter.metadata.export = false",
        f"generate.only_dead_patients = {'true' if only_dead else 'false'}",
        f"generate.only_alive_patients = {'false' if only_dead else 'true'}"
    ]
    path.write_text("\n".join(content))


def run_synthea(size, batch_id, gender, output_root, only_dead):
    output_root = Path(output_root)
    batch_dir = output_root / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    config_path = batch_dir / "temp.properties"
    create_config(config_path, only_dead=only_dead)

    jar_path = Path("/mnt/a/Code/GitHub/HealthcareAI-Thesis/synthea/synthea-with-dependencies.jar")

    # WILDCARDS: Use * to ensure it catches the nested files in the JAR
    # We include death, wellness, and encounters to ensure oncology patients
    # are actually diagnosed and can die.
    modules = "*cancer*,*death*,*wellness*,*encounter*"

    cmd = [
        "java", "-Xmx5G", "-jar", str(jar_path),
        "-s", str(batch_id),
        "-p", str(size),
        "-g", gender,
        "-a", "50-100",  # High-incidence age for oncology
        "-m", modules,
        "-c", str(config_path),  # Pass the custom properties file
        "-r", "20260324",
        "--exporter.baseDirectory", str(batch_dir)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # Check if the modules actually loaded this time
    if "[0 loaded]" in result.stdout or "Warnings" in result.stderr:
        print(f"Batch {batch_id} Log: {result.stdout.splitlines()[-1]}")

    return batch_dir


def merge_and_sample(batch_dirs, final_dir, target_n):
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Identifying Oncology-Positive Cohort...")
    all_patients = []
    for b_dir in batch_dirs:
        # Check if they have at least one record in conditions.csv (Oncology diagnosis)
        p_file = next(b_dir.rglob("patients.csv"), None)
        c_file = next(b_dir.rglob("conditions.csv"), None)

        if p_file and c_file:
            patients = cudf.read_csv(p_file, usecols=['Id', 'DEATHDATE'])
            conditions = cudf.read_csv(c_file, usecols=['PATIENT'])
            # Only keep patients who actually appear in the conditions log
            oncology_patients = patients[patients['Id'].isin(conditions['PATIENT'].unique())]
            all_patients.append(oncology_patients)

    full_df = cudf.concat(all_patients)
    dead = full_df[full_df['DEATHDATE'].notnull()]
    alive = full_df[full_df['DEATHDATE'].isnull()]

    half_n = min(target_n // 2, len(dead), len(alive))
    print(f"Found {len(dead)} dead and {len(alive)} alive oncology patients. Sampling {half_n} of each.")

    selected_ids = cudf.concat([
        dead.sample(n=half_n, random_state=42)['Id'],
        alive.sample(n=half_n, random_state=42)['Id']
    ])

    print("Step 2: Performing GPU-accelerated merge of all tables...")
    all_csvs = set(f.name for b in batch_dirs for f in b.rglob("*.csv"))
    for csv_name in all_csvs:
        first = True
        for b_dir in batch_dirs:
            f_path = next(b_dir.rglob(csv_name), None)
            if not f_path: continue
            df = cudf.read_csv(f_path)
            id_col = 'Id' if 'Id' in df.columns else 'PATIENT'
            if id_col in df.columns:
                filtered = df[df[id_col].isin(selected_ids)]
                if len(filtered) > 0:
                    mode, header = ('w', True) if first else ('a', False)
                    filtered.to_csv(final_dir / csv_name, mode=mode, header=header, index=False)
                    first = False
            del df


def main():
    target_n = 30000
    temp_root = Path.home() / "synthea_temp"
    final_dir = Path("/mnt/a/Code/GitHub/HealthcareAI-Thesis/data")

    if temp_root.exists(): shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)

    # We run 10 batches for 'Dead Only' and 10 batches for 'Alive Only'
    # Generating 40k total to ensure we have enough cancer-positive samples
    total_to_gen = 40000
    n_per_batch = total_to_gen // 20

    print(f"Generating balanced cohorts in {temp_root}...")
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(20):
            is_dead_batch = i < 10  # 50% of processes focus on dead patients
            gender = "M" if i % 2 == 0 else "F"
            futures.append(executor.submit(run_synthea, n_per_batch, i, gender, temp_root, is_dead_batch))
        batch_paths = [f.result() for f in futures]

    merge_and_sample(batch_paths, final_dir, target_n)
    shutil.rmtree(temp_root)
    print(f"Done! Dataset at {final_dir}")


if __name__ == "__main__":
    main()