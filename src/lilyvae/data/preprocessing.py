"""Preprocess raw LILY CSV files into training tensors."""

import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

from lilyvae.config import RAW_DIR, PROCESSED_DIR, BIN_SIZE_CM, TRAINING_VARIABLES


# Exact file-to-column mapping based on LILY CSV schemas
TRAINING_FILES = {
    "GRA_DataLITH.csv": {
        "measurement_col": "Bulk density (GRA)",
        "output_name": "bulk_density",
    },
    "MS_DataLITH.csv": {
        "measurement_col": "Magnetic susceptibility (instr. units)",
        "output_name": "magnetic_susceptibility",
    },
    "NGR_DataLITH.csv": {
        "measurement_col": "NGR total counts (cps)",
        "output_name": "NGR",
    },
    "RGB_DataLITH.csv": {
        "measurement_cols": ["R", "G", "B"],
        "output_names": ["R", "G", "B"],
    },
}

# Common columns across all LILY CSVs
DEPTH_COL = "Depth CSF-A (m)"
BOREHOLE_COLS = ["Exp", "Site", "Hole"]
LITHOLOGY_COL = "Principal"
LAT_COL = "Latitude (DD)"
LON_COL = "Longitude (DD)"


def make_borehole_key(df: pd.DataFrame) -> pd.Series:
    """Construct borehole ID from Exp-Site-Hole."""
    return (
        df["Exp"].astype(str) + "-" +
        df["Site"].astype(str) + "-" +
        df["Hole"].astype(str)
    )


def load_and_bin(csv_path: Path, measurement_cols: list[str], bin_size_cm: int = BIN_SIZE_CM) -> pd.DataFrame:
    """Load a CSV, create borehole key, and bin into depth intervals."""
    print(f"  Loading {csv_path.name}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"    Shape: {df.shape}")

    # Ensure depth column is numeric
    df[DEPTH_COL] = pd.to_numeric(df[DEPTH_COL], errors="coerce")
    df = df.dropna(subset=[DEPTH_COL])

    # Ensure measurement columns are numeric
    for col in measurement_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["_borehole"] = make_borehole_key(df)

    # Convert depth to bin index (20cm bins)
    bin_size_m = bin_size_cm / 100.0
    df["_depth_bin"] = (df[DEPTH_COL] / bin_size_m).round().astype(int)

    return df


def preprocess(raw_dir: Path | None = None, output_dir: Path | None = None) -> None:
    """Run full preprocessing pipeline."""
    raw_dir = raw_dir or RAW_DIR
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["_borehole", "_depth_bin"]

    # Step 1: Load and bin each training measurement
    measurement_dfs = {}
    lithology_data = None
    latlon_data = None

    for filename, spec in TRAINING_FILES.items():
        csv_path = raw_dir / filename
        if not csv_path.exists():
            print(f"  WARNING: {filename} not found, skipping")
            continue

        # Determine columns
        if "measurement_cols" in spec:
            meas_cols = spec["measurement_cols"]
            out_names = spec["output_names"]
        else:
            meas_cols = [spec["measurement_col"]]
            out_names = [spec["output_name"]]

        df = load_and_bin(csv_path, meas_cols)

        # Aggregate by borehole + depth bin
        binned = df.groupby(group_cols)[meas_cols].mean().reset_index()
        rename_map = dict(zip(meas_cols, out_names))
        binned = binned.rename(columns=rename_map)
        measurement_dfs[filename] = binned
        print(f"    Binned to {len(binned)} rows")

        # Extract lithology from first file that has it
        if lithology_data is None and LITHOLOGY_COL in df.columns:
            lith_binned = df.groupby(group_cols)[LITHOLOGY_COL].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            ).reset_index()
            lith_binned = lith_binned.rename(columns={LITHOLOGY_COL: "lithology"})
            lithology_data = lith_binned
            print(f"    Extracted lithology ({lith_binned['lithology'].nunique()} classes)")

        # Extract lat/lon from first file
        if latlon_data is None and LAT_COL in df.columns:
            ll = df.groupby(group_cols).agg({LAT_COL: "first", LON_COL: "first"}).reset_index()
            ll = ll.rename(columns={LAT_COL: "latitude", LON_COL: "longitude"})
            latlon_data = ll

    # Step 2: Merge all measurement types on borehole + depth bin
    print("\nMerging measurements...")
    merged = None
    for filename, df in measurement_dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=group_cols, how="inner")
        print(f"  After merging {filename}: {len(merged)} rows")

    if merged is None:
        raise RuntimeError("No measurement data loaded")

    # Merge lithology
    if lithology_data is not None:
        merged = merged.merge(lithology_data, on=group_cols, how="left")

    # Merge lat/lon
    if latlon_data is not None:
        merged = merged.merge(latlon_data, on=group_cols, how="left")

    # Step 3: Filter to rows with all 6 training variables
    feature_cols = ["bulk_density", "magnetic_susceptibility", "NGR", "R", "G", "B"]
    missing_cols = [c for c in feature_cols if c not in merged.columns]
    if missing_cols:
        raise RuntimeError(f"Missing feature columns after merge: {missing_cols}")

    merged = merged.dropna(subset=feature_cols)
    n_boreholes = merged["_borehole"].nunique()
    print(f"\nAfter dropping NaN: {len(merged)} samples, {n_boreholes} boreholes")

    # Step 4: Encode lithology labels
    le = LabelEncoder()
    if "lithology" in merged.columns:
        merged["lithology"] = merged["lithology"].fillna("unknown")
        labels_encoded = le.fit_transform(merged["lithology"])
        n_classes = len(le.classes_)
        print(f"Lithology classes: {n_classes}")
    else:
        labels_encoded = np.zeros(len(merged), dtype=np.int64)
        n_classes = 0
        print("WARNING: No lithology column found")

    # Step 5: Scale features
    features_np = merged[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np).astype(np.float32)

    # Step 6: Save everything
    print(f"\nSaving to {output_dir}...")
    torch.save(torch.from_numpy(features_scaled), output_dir / "train_features.pt")
    torch.save(torch.from_numpy(labels_encoded.astype(np.int64)), output_dir / "train_labels.pt")

    metadata = {
        "borehole_ids": merged["_borehole"].values,
        "depth_bins": merged["_depth_bin"].values,
        "feature_columns": feature_cols,
        "n_samples": len(merged),
        "n_boreholes": n_boreholes,
    }
    if "latitude" in merged.columns:
        metadata["latitude"] = pd.to_numeric(merged["latitude"], errors="coerce").values
        metadata["longitude"] = pd.to_numeric(merged["longitude"], errors="coerce").values

    torch.save(metadata, output_dir / "metadata.pt")

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Step 7: Collect zero-shot target variables from all other CSVs
    print("\nCollecting zero-shot target variables...")
    zeroshot_targets = {}
    training_filenames = set(TRAINING_FILES.keys())

    for csv_path in sorted(raw_dir.glob("*_DataLITH.csv")):
        if csv_path.name in training_filenames:
            continue

        try:
            print(f"  Loading {csv_path.name}...")
            df = pd.read_csv(csv_path, low_memory=False)

            # Find numeric measurement columns (skip metadata)
            skip_patterns = [
                "exp", "site", "hole", "core", "type", "sect", "a/w", "offset",
                "depth", "timestamp", "instrument", "text id", "test no", "comment",
                "prefix", "principal", "suffix", "lithology", "consolidation",
                "subtype", "expanded", "latitude", "longitude", "water depth",
            ]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            meas_cols = [
                c for c in numeric_cols
                if not any(pat in c.lower() for pat in skip_patterns)
            ]

            if not meas_cols:
                print(f"    No measurement columns found, skipping")
                continue

            # Prepare for binning
            df[DEPTH_COL] = pd.to_numeric(df[DEPTH_COL], errors="coerce")
            df = df.dropna(subset=[DEPTH_COL])
            df["_borehole"] = make_borehole_key(df)
            bin_size_m = BIN_SIZE_CM / 100.0
            df["_depth_bin"] = (df[DEPTH_COL] / bin_size_m).round().astype(int)

            for col in meas_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            binned = df.groupby(group_cols)[meas_cols].mean().reset_index()

            # Align with main dataset
            aligned = merged[["_borehole", "_depth_bin"]].merge(
                binned, on=group_cols, how="left"
            )

            csv_base = csv_path.stem.replace("_DataLITH", "")
            for col in meas_cols:
                valid_mask = aligned[col].notna()
                n_valid = int(valid_mask.sum())
                if n_valid > 100:
                    values = np.nan_to_num(aligned[col].values, nan=0.0).astype(np.float32)
                    target_name = f"{csv_base}_{col}" if len(meas_cols) > 1 else csv_base
                    zeroshot_targets[target_name] = {
                        "values": torch.from_numpy(values),
                        "valid_mask": torch.from_numpy(valid_mask.values),
                        "n_valid": n_valid,
                    }
                    print(f"    {target_name}: {n_valid} valid samples")

        except Exception as e:
            print(f"    Skipping {csv_path.name}: {e}")

    torch.save(zeroshot_targets, output_dir / "zeroshot_targets.pt")

    print(f"\nPreprocessing complete:")
    print(f"  Samples: {len(merged)}")
    print(f"  Boreholes: {n_boreholes}")
    print(f"  Features: {feature_cols}")
    print(f"  Lithology classes: {n_classes}")
    print(f"  Zero-shot targets: {len(zeroshot_targets)}")
