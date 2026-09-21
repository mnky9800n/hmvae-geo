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
            before fitting. Set for the SRM whole-core magnetic columns,
            where the LILY release contains values up to 1.16e8 A m^2,
            twelve orders of magnitude above anything physically
            plausible for sediment.
    """

    tier: Tier
    group: str
    instrument: str
    label: str
    reason: str = ""
    winsorize: bool = False


_P = "procedural"
_D = "duplicate"
_S = "same_modality"
_I = "independent"

#: Curation for every column emitted by preprocessing. Keys must match
#: the names in `zeroshot_targets.pt` exactly.
TARGET_REGISTRY: dict[str, TargetSpec] = {
    # --- Automated vane shear -------------------------------------------
    "AVS_Vane shear strength (kN/m^2)": TargetSpec(
        _I, "shear_strength", "AVS", "Vane shear strength"
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
    "MAD_Porosity (vol%)": TargetSpec(_I, "porosity", "MAD", "Porosity"),
    "MAD_Grain density (g/cm^3)": TargetSpec(
        _I, "grain_density", "MAD", "Grain density"
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
    "PEN": TargetSpec(_I, "penetration_resistance", "PEN", "Penetration resistance"),
    # --- P-wave velocity --------------------------------------------------
    "PWC_P-wave velocity x (m/s)": TargetSpec(
        _I, "p_wave_discrete", "PWC", "P-wave velocity (discrete)"
    ),
    "PWL_P-wave velocity xy (m/s)": TargetSpec(
        _I, "p_wave_whole_core", "PWL", "P-wave velocity (whole core)"
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
        _I, "inclination_discrete", "SRMD", "Inclination (discrete)"
    ),
    "SRMD_Declination raw (deg)": TargetSpec(
        _I, "declination_discrete", "SRMD", "Declination (discrete)"
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
        _D, "inclination_discrete", "SRMD", "Inclination (corrected)",
        reason="Drift-corrected copy of the raw inclination.",
    ),
    "SRMD_Declination background & drift corrected (deg)": TargetSpec(
        _D, "declination_discrete", "SRMD", "Declination (corrected)",
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
        _I, "inclination_whole_core", "SRM", "Inclination (whole core)"
    ),
    "SRM_Declination raw (deg)": TargetSpec(
        _I, "declination_whole_core", "SRM", "Declination (whole core)"
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
        _D, "inclination_whole_core", "SRM", "Inclination (corrected)",
        reason="Drift-corrected copy of the raw inclination.",
    ),
    "SRM_Declination background & drift corrected (deg)": TargetSpec(
        _D, "declination_whole_core", "SRM", "Declination (corrected)",
        reason="Drift-corrected copy of the raw declination.",
    ),
    "SRM_Demag level (mT)": TargetSpec(
        _P, "instrument_setting", "SRM", "Demag level (whole core)",
        reason="Applied demagnetising field, chosen by the operator.",
    ),
    # --- Thermal conductivity ---------------------------------------------
    "TCON_Thermal conductivity mean (W/(m*K))": TargetSpec(
        _I, "thermal_conductivity", "TCON", "Thermal conductivity"
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
