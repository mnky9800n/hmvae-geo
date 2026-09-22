"""Curation of the held-out variables used for zero-shot evaluation.

Why this module exists
----------------------
Zero-shot targets are harvested automatically in `data/preprocessing.py`
by dropping columns whose names match a `skip_patterns` list. That filter
is name-based, so three kinds of column survive that should not be
averaged into a headline number:

1. **Procedural columns.** Instrument settings, sample-preparation
   geometry, measurement-dispersion statistics, and ID-like numeric
   columns (the RSC "spectral data link" fields, `SRMD_Raw data`). These
   are not rock properties, so predicting them says nothing about the
   representation.
2. **Algebraic duplicates.** The nine MAD mass/volume columns and the
   six porosity-family columns are all computed from the same three
   masses and two volumes; CIELAB `L*a*b*` is a transform of tristimulus
   `XYZ`; the SRM `raw` and `background & drift corrected` variants are
   the same measurement twice. Reporting each one separately weights a
   single physical quantity by however many columns the database happens
   to store it in.
3. **Same-modality re-measurements.** Point magnetic susceptibility is
   the same physical quantity as a training input measured on a
   different instrument; spectrophotometric reflectance is colour
   measured by a different sensor than the line-scan RGB we train on.
   These are real and interesting results -- the representation
   transfers across sensors -- but they are not the same claim as
   recovering thermal conductivity, and must not sit in the same mean.

What is left after removing those is the **independent** tier: physical
properties that are neither a training variable in disguise nor a
restatement of another target. That tier is the headline set.

The hand assignment below is checked against the data by
`redundancy_clusters`, which clusters targets on their rank correlation
and demotes anything tiered `independent` that turns out to be
collinear with a training variable or another target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.stats import spearmanr

Tier = Literal["procedural", "duplicate", "same_modality", "independent"]

#: Tiers that contribute to the headline zero-shot number.
HEADLINE_TIERS: tuple[Tier, ...] = ("independent",)

#: Tiers reported in the paper, each as its own row group.
REPORTED_TIERS: tuple[Tier, ...] = ("independent", "same_modality")


@dataclass(frozen=True)
class TargetSpec:
    """Curation metadata for one held-out variable.

    Attributes:
        tier: Which row group the variable belongs to.
        group: The physical quantity it measures. Targets sharing a group
            are restatements of one another; exactly one of them is the
            representative.
        instrument: The LILY instrument code the column came from.
        label: Short display name for figures and tables.
        reason: Why a procedural or duplicate target is excluded. Printed
            in the supporting-information table so every exclusion is
            auditable rather than silent.
        winsorize: Clip the top and bottom percentile of the target
            before fitting. A blunt fallback for columns whose corruption
            has no clean physical bound.
        valid_range: (min, max) physically admissible values. Rows
            outside are dropped before fitting or scoring. Preferred over
            `winsorize` wherever the instrument defines a plausible
            range, because it removes only impossible values instead of
            trimming a fixed fraction of real ones.
    """

    tier: Tier
    group: str
    instrument: str
    label: str
    reason: str = ""
    winsorize: bool = False
    valid_range: tuple[float, float] | None = None


_P = "procedural"
_D = "duplicate"
_S = "same_modality"
_I = "independent"

#: Curation for every column emitted by preprocessing. Keys must match
#: the names in `zeroshot_targets.pt` exactly.
TARGET_REGISTRY: dict[str, TargetSpec] = {
    # --- Automated vane shear -------------------------------------------
    "AVS_Vane shear strength (kN/m^2)": TargetSpec(
        _I, "shear_strength", "AVS", "Vane shear strength",
        valid_range=(0.0, 1000.0),
    ),
    "AVS_Max torque angle (deg)": TargetSpec(
        _D, "shear_strength", "AVS", "Max torque angle",
        reason="Second reading from the same vane test as shear strength.",
    ),
    "TOR": TargetSpec(
        _D, "shear_strength", "TOR", "Torvane shear strength",
        reason="Undrained shear strength again, by torvane instead of vane.",
    ),
    # --- Moisture and density -------------------------------------------
    "MAD_Porosity (vol%)": TargetSpec(
        _I, "porosity", "MAD", "Porosity",
        # A volume fraction. The release reaches -130 vol%.
        valid_range=(0.0, 100.0),
    ),
    "MAD_Grain density (g/cm^3)": TargetSpec(
        _I, "grain_density", "MAD", "Grain density",
        # Rock-forming minerals span roughly quartz to heavy accessories.
        valid_range=(1.5, 5.0),
    ),
    "MAD_Void ratio": TargetSpec(
        _D, "porosity", "MAD", "Void ratio",
        reason="Exact transform of porosity, e = phi / (1 - phi).",
    ),
    "MAD_Moisture dry (wt%)": TargetSpec(
        _D, "porosity", "MAD", "Moisture (dry wt%)",
        reason="Porewater fraction from the same masses as porosity.",
    ),
    "MAD_Moisture wet (wt%)": TargetSpec(
        _D, "porosity", "MAD", "Moisture (wet wt%)",
        reason="Porewater fraction from the same masses as porosity.",
    ),
    "MAD_Dry density (g/cm^3)": TargetSpec(
        _D, "porosity", "MAD", "Dry density",
        reason="Bulk density and porosity recombined; no new measurement.",
    ),
    "MAD_Bulk density (g/cm^3)": TargetSpec(
        _D, "porosity", "MAD", "Bulk density (discrete)",
        reason=(
            "Same MAD measurement pair as porosity (Spearman rho = -0.97); "
            "reporting both double-counts one determination. Note it is not "
            "the GRA training input restated: the discrete and gamma-"
            "attenuation bulk densities correlate at only rho = 0.57, because "
            "they sample different material at different scales."
        ),
    ),
    **{
        name: TargetSpec(
            _P, "mad_sample_prep", "MAD", name.split("_", 1)[1],
            reason="Sample-preparation mass or volume, not a rock property.",
        )
        for name in (
            "MAD_Mass wet sample (g)",
            "MAD_Mass dried sample (g)",
            "MAD_Mass porewater (g)",
            "MAD_Mass salt (g)",
            "MAD_Mass solids (g)",
            "MAD_Vol wet sample (cm^3)",
            "MAD_Vol dried sample (cm^3)",
            "MAD_Vol porewater (cm^3)",
            "MAD_Vol salt (cm^3)",
            "MAD_Vol solids (cm^3)",
        )
    },
    # --- Point magnetic susceptibility ----------------------------------
    "MSP": TargetSpec(
        _S, "magnetic_susceptibility", "MSP", "Magnetic susceptibility (point)",
        reason="Same quantity as the MS training input, point sensor.",
    ),
    # --- Penetrometer ----------------------------------------------------
    "PEN": TargetSpec(
        _I, "penetration_resistance", "PEN", "Penetration resistance",
        valid_range=(0.0, 1000.0),
    ),
    # --- P-wave velocity --------------------------------------------------
    "PWC_P-wave velocity x (m/s)": TargetSpec(
        _I, "p_wave_discrete", "PWC", "P-wave velocity (discrete)",
        # Bounded below by water. The release contains -3.17e6 m/s.
        valid_range=(1000.0, 8000.0),
    ),
    "PWL_P-wave velocity xy (m/s)": TargetSpec(
        _I, "p_wave_whole_core", "PWL", "P-wave velocity (whole core)",
        valid_range=(1000.0, 8000.0),
    ),
    **{
        name: TargetSpec(
            _D, "p_wave_discrete", name.split("_", 1)[0], name.split("_", 1)[1],
            reason="Discrete-sample P-wave velocity on another axis or run.",
        )
        for name in (
            "PWC_P-wave velocity x [manual] (m/s)",
            "PWB_P-wave velocity y (m/s)",
            "PWB_P-wave velocity z (m/s)",
            "PWB_P-wave velocity y [manual] (m/s)",
            "PWB_P-wave velocity z [manual] (m/s)",
        )
    },
    "PWB_Caliper separation (mm)": TargetSpec(
        _P, "instrument_geometry", "PWB", "Caliper separation (PWB)",
        reason="Instrument path length, a geometry reading, not the rock.",
    ),
    "PWL_Caliper separation (mm)": TargetSpec(
        _P, "instrument_geometry", "PWL", "Caliper separation (PWL)",
        reason="Core diameter seen by the logger, set by the liner.",
    ),
    "PWL_Sonic traveltime (microsecond)": TargetSpec(
        _P, "p_wave_whole_core", "PWL", "Sonic traveltime",
        reason="Algebraically caliper separation divided by velocity.",
    ),
    # --- Spectrophotometry ------------------------------------------------
    "RSC_Reflectance L*": TargetSpec(
        _S, "color_reflectance", "RSC", "Reflectance L*",
        reason="Colour from a different sensor than the RGB training input.",
    ),
    **{
        name: TargetSpec(
            _D, "color_reflectance", "RSC", name.split("_", 1)[1],
            reason="CIELAB and tristimulus are transforms of one another.",
        )
        for name in (
            "RSC_Reflectance a*",
            "RSC_Reflectance b*",
            "RSC_Tristimulus X",
            "RSC_Tristimulus Y",
            "RSC_Tristimulus Z",
        )
    },
    "RSC_Normalized spectral data link": TargetSpec(
        _P, "record_id", "RSC", "Spectral data link (norm.)",
        reason="Numeric record identifier that survived the name filter.",
    ),
    "RSC_Unnormalized spectral data link": TargetSpec(
        _P, "record_id", "RSC", "Spectral data link (unnorm.)",
        reason="Numeric record identifier that survived the name filter.",
    ),
    # --- Discrete rock magnetometer ---------------------------------------
    "SRMD_Intensity raw (A/m)": TargetSpec(
        _I, "magnetisation_discrete", "SRMD", "Magnetisation (discrete)"
    ),
    "SRMD_Inclination raw (deg)": TargetSpec(
        _P, "drilling_overprint", "SRMD", "Inclination (discrete)",
        reason=(
            "Inclination does survive in unoriented core, unlike declination, "
            "so this is a real palaeomagnetic quantity in principle. In this "
            "release it is not: 75% of values are positive with a mean of "
            "+27 deg, where a genuine record over this much section would be "
            "roughly balanced between normal and reversed polarity. That bias "
            "is the steep downward overprint the drill string imparts, so "
            "predicting it means predicting how strongly the core took the "
            "overprint, which tracks magnetic susceptibility, a training input."
        ),
        valid_range=(-90.0, 90.0),
    ),
    "SRMD_Declination raw (deg)": TargetSpec(
        _P, "core_orientation", "SRMD", "Declination (discrete)",
        reason=(
            "Azimuth of the magnetisation in core that is not azimuthally "
            "oriented. The piece rotates freely in the barrel, so this records "
            "how it happened to sit, not the rock."
        ),
        valid_range=(-360.0, 360.0),
    ),
    **{
        name: TargetSpec(
            _D, "magnetisation_discrete", "SRMD", name.split("_", 1)[1],
            reason="Vector component or drift-corrected copy of magnetisation.",
        )
        for name in (
            "SRMD_Magnetic moment x (Am^2)",
            "SRMD_Magnetic moment y (Am^2)",
            "SRMD_Magnetic moment z (Am^2)",
            "SRMD_Intensity background & drift corrected (A/m)",
        )
    },
    "SRMD_Inclination background & drift corrected (deg)": TargetSpec(
        _P, "drilling_overprint", "SRMD", "Inclination (corrected)",
        reason="Drift-corrected copy of the raw inclination.",
    ),
    "SRMD_Declination background & drift corrected (deg)": TargetSpec(
        _P, "core_orientation", "SRMD", "Declination (corrected)",
        reason="Drift-corrected copy of the raw declination.",
    ),
    "SRMD_Demag level (mT)": TargetSpec(
        _P, "instrument_setting", "SRMD", "Demag level (discrete)",
        reason="Applied demagnetising field, chosen by the operator.",
    ),
    "SRMD_Raw data": TargetSpec(
        _P, "record_id", "SRMD", "Raw data link",
        reason="Numeric record identifier that survived the name filter.",
    ),
    # --- Whole-core rock magnetometer -------------------------------------
    "SRM_Intensity raw (A/m)": TargetSpec(
        _I, "magnetisation_whole_core", "SRM", "Magnetisation (whole core)",
        winsorize=True,
    ),
    "SRM_Inclination raw (deg)": TargetSpec(
        _P, "drilling_overprint", "SRM", "Inclination (whole core)",
        reason=(
            "Same drilling overprint as the discrete measurement: 76% positive, "
            "mean +27 deg, against the balanced polarity a real record would show."
        ),
        valid_range=(-90.0, 90.0),
    ),
    "SRM_Declination raw (deg)": TargetSpec(
        _P, "core_orientation", "SRM", "Declination (whole core)",
        reason=(
            "Azimuth of magnetisation in unoriented core; near-uniform over "
            "0-360 deg, with the modal 5-degree bin holding 1.3% of samples."
        ),
        valid_range=(-360.0, 360.0),
    ),
    **{
        name: TargetSpec(
            _D, "magnetisation_whole_core", "SRM", name.split("_", 1)[1],
            reason="Vector component or drift-corrected copy of magnetisation.",
            winsorize=True,
        )
        for name in (
            "SRM_Magnetic moment x (Am^2)",
            "SRM_Magnetic moment y (Am^2)",
            "SRM_Magnetic moment z (Am^2)",
            "SRM_Intensity background & drift corrected (A/m)",
        )
    },
    "SRM_Inclination background & drift corrected (deg)": TargetSpec(
        _P, "drilling_overprint", "SRM", "Inclination (corrected)",
        reason="Drift-corrected copy of the raw inclination.",
    ),
    "SRM_Declination background & drift corrected (deg)": TargetSpec(
        _P, "core_orientation", "SRM", "Declination (corrected)",
        reason="Drift-corrected copy of the raw declination.",
    ),
    "SRM_Demag level (mT)": TargetSpec(
        _P, "instrument_setting", "SRM", "Demag level (whole core)",
        reason="Applied demagnetising field, chosen by the operator.",
    ),
    # --- Thermal conductivity ---------------------------------------------
    "TCON_Thermal conductivity mean (W/(m*K))": TargetSpec(
        _I, "thermal_conductivity", "TCON", "Thermal conductivity",
        # Wet sediment sits near 1 W/(m K). The release reaches 3.75e9,
        # and those nine rows alone drove this column's reported R^2 from
        # 0.78 to 0.99 in-sample while destroying it out of borehole.
        valid_range=(0.2, 10.0),
    ),
    "TCON_Thermal conductivity observations (W/(m*K))": TargetSpec(
        _D, "thermal_conductivity", "TCON", "Thermal conductivity (obs.)",
        reason="Individual observations behind the reported mean.",
    ),
    "TCON_Calculated TCON value (W/(m*K))": TargetSpec(
        _D, "thermal_conductivity", "TCON", "Thermal conductivity (calc.)",
        reason="Same conductivity recomputed by the shipboard software.",
    ),
    "TCON_Conductivity std dev (W/(m*K))": TargetSpec(
        _P, "measurement_dispersion", "TCON", "Conductivity std dev",
        reason="Spread of repeat measurements, a quality statistic.",
    ),
}

#: The representative target for each group, i.e. the one column kept
#: when a physical quantity is stored under several names. Derived from
#: the registry so it cannot drift out of sync.
GROUP_REPRESENTATIVE: dict[str, str] = {
    spec.group: name
    for name, spec in TARGET_REGISTRY.items()
    if spec.tier in REPORTED_TIERS
}


class RegistryError(ValueError):
    """Raised when the registry and the preprocessed targets disagree."""


def validate_registry(target_names: Iterable[str]) -> None:
    """Check the registry covers exactly the harvested targets.

    Preprocessing discovers targets by scanning CSV columns, so a new
    LILY release or a change to `skip_patterns` can silently introduce a
    column with no curation. That must fail loudly rather than quietly
    entering a headline mean.

    Raises:
        RegistryError: if a harvested target has no spec, or a spec
            names a target that no longer exists, or two reported
            targets claim the same group.
    """
    harvested = set(target_names)
    known = set(TARGET_REGISTRY)

    missing = sorted(harvested - known)
    if missing:
        raise RegistryError(
            f"{len(missing)} harvested target(s) have no TargetSpec; add them "
            f"to TARGET_REGISTRY with a tier: {missing}"
        )

    stale = sorted(known - harvested)
    if stale:
        raise RegistryError(
            f"{len(stale)} registry entries no longer exist in the data: {stale}"
        )

    seen: dict[str, str] = {}
    for name, spec in TARGET_REGISTRY.items():
        if spec.tier not in REPORTED_TIERS:
            continue
        if spec.group in seen:
            raise RegistryError(
                f"Group {spec.group!r} has two reported representatives: "
                f"{seen[spec.group]!r} and {name!r}. Exactly one target per "
                f"physical quantity may be reported."
            )
        seen[spec.group] = name


def select_targets(
    target_names: Iterable[str],
    tiers: Iterable[Tier] = HEADLINE_TIERS,
) -> list[str]:
    """Return the harvested targets belonging to the given tiers.

    Order follows the registry, so tables and figures list variables in a
    stable, instrument-grouped order rather than dictionary order.
    """
    harvested = set(target_names)
    wanted = set(tiers)
    return [
        name
        for name, spec in TARGET_REGISTRY.items()
        if name in harvested and spec.tier in wanted
    ]


def winsorize_values(
    values: np.ndarray, percentile: float = 1.0
) -> tuple[np.ndarray, int]:
    """Clip both tails of a target at the given percentile.

    Returns the clipped values and how many rows were altered. Used for
    the SRM whole-core magnetic columns: `scripts/srm_diagnostic.py`
    shows R^2 on magnetic moment y falling from 0.9999 to 0.194 once the
    top 1% of |y| is removed, so the headline value there measured a
    handful of corrupt records rather than the representation.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values, 0
    lo, hi = np.percentile(values, [percentile, 100.0 - percentile])
    clipped = np.clip(values, lo, hi)
    return clipped, int(np.count_nonzero(clipped != values))


def redundancy_clusters(
    columns: dict[str, np.ndarray],
    threshold: float = 0.95,
) -> list[list[str]]:
    """Group columns that are rank-collinear above ``threshold``.

    Single-linkage clustering on |Spearman rho| computed over the rows
    where both columns are valid. This audits the hand tiering above: if
    a target tiered `independent` lands in a cluster with a training
    variable or with another reported target, the hand call was wrong.

    Args:
        columns: Mapping of name to 1-D values. Include the six training
            variables to catch targets that merely restate an input.
        threshold: |rho| at or above which two columns are linked.

    Returns:
        Clusters of size two or more, each sorted, ordered by size.
    """
    names = list(columns)
    parent = {n: n for n in names}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            xa, xb = columns[a], columns[b]
            both = np.isfinite(xa) & np.isfinite(xb)
            if both.sum() < 50:
                continue
            rho = spearmanr(xa[both], xb[both]).statistic
            if np.isfinite(rho) and abs(rho) >= threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted(
        (sorted(g) for g in groups.values() if len(g) > 1),
        key=len,
        reverse=True,
    )


def apply_valid_range(
    values: np.ndarray, valid_mask: np.ndarray, spec: TargetSpec | None
) -> tuple[np.ndarray, int]:
    """Drop physically impossible values from a target's valid mask.

    The LILY release contains values that cannot be measurements:
    thermal conductivity up to 3.75e9 W/(m K) against a sediment median
    of 1.24, P-wave velocity down to -3.17e6 m/s, porosity at -130 vol%.
    Such values dominate the total sum of squares, so a regressor that
    memorises a handful of them scores R^2 near 1.0 while learning
    nothing about rock, and fails completely once they land on the other
    side of a split. Nine such rows out of 5,134 were enough to move
    thermal conductivity's reported score from 0.78 to 0.99 in-sample and
    from 0.49 to -0.01 out of borehole.

    Returns the narrowed mask and the number of rows removed.
    """
    if spec is None or spec.valid_range is None:
        return valid_mask, 0
    lo, hi = spec.valid_range
    values = np.asarray(values, dtype=np.float64)
    narrowed = valid_mask & (values >= lo) & (values <= hi)
    return narrowed, int(valid_mask.sum() - narrowed.sum())
