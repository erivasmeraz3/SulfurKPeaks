"""
S1s XAS Peak Fitting - Sulfur K-edge

Based on the Gaussian Curve Fitting (GCF) method from:
  Manceau & Nagy (2012) Geochimica et Cosmochimica Acta 99, 206-223.
  "Quantitative analysis of sulfur functional groups in natural organic
   matter by XANES spectroscopy"

Model: 6 Gaussian peaks + 2 arctangent step functions for S K-edge XANES.

Peak assignments (6 sulfur functionalities):
  1. Exocyclic S  (thiol/sulfide)  ~2473.2 eV  (EOS ~0 to +0.5)
  2. Heterocyclic S (thiophene)    ~2474.4 eV  (EOS ~+0.6 to +0.8)
  3. Sulfoxide    (R-SO-R)         ~2476.4 eV  (EOS ~+2)
  4. Sulfone      (R-SO2-R)        ~2479.6 eV  (EOS ~+4)
  5. Sulfonate    (R-SO3-)         ~2481.3 eV  (EOS ~+5)
  6. Sulfate      (SO4^2-)         ~2482.75 eV (EOS ~+6)

Baseline: Two arctangent step functions:
  - A1: between heterocyclic S and sulfoxide (~2475.7 eV)
  - A2: above sulfate (~2483.5 eV)

FWHM groups (per Manceau & Nagy 2012):
  - Reduced:  peaks 1-3 (Exocyclic, Heterocyclic, Sulfoxide; shared FWHM, ~1.55-2.39 eV)
  - Oxidized: peaks 4-6 (Sulfone, Sulfonate, Sulfate; shared FWHM, ~1.94-2.39 eV)

Cross-section correction:
  Generic calibration curve from Manceau & Nagy 2012:
    scaling_factor(E) = 0.36841 * E - 909.97
  normalized to 1.0 at elemental sulfur (2472.70 eV).
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import pandas as pd
from pathlib import Path
import argparse


# ─── Peak definitions ───────────────────────────────────────────────────────

PEAK_NAMES = ['Exocyclic', 'Heterocyclic', 'Sulfoxide',
              'Sulfone', 'Sulfonate', 'Sulfate']

PEAK_DISPLAY_NAMES = ['Exocyclic S', 'Heterocyclic S', 'Sulfoxide',
                      'Sulfone', 'Sulfonate', 'Sulfate']

DEFAULT_PEAK_CENTERS = {
    'Exocyclic':    2473.2,
    'Heterocyclic': 2474.4,
    'Sulfoxide':    2476.4,
    'Sulfone':      2479.6,
    'Sulfonate':    2481.3,
    'Sulfate':      2482.75,
}

# Allowed range (±) for each peak center during fitting
# Per Manceau & Nagy 2012 Section 3.1.2: peaks 3-6 fixed at nominal,
# Exocyclic shifts at most -0.2 eV, Heterocyclic at most +0.3 eV
DEFAULT_PEAK_RANGES = {
    'Exocyclic':    0.2,   # paper: shifts left by at most -0.2 eV
    'Heterocyclic': 0.3,   # paper: shifts right by at most +0.3 eV
    'Sulfoxide':    0.0,   # paper: fixed at nominal
    'Sulfone':      0.0,   # paper: fixed at nominal
    'Sulfonate':    0.0,   # paper: fixed at nominal
    'Sulfate':      0.0,   # paper: fixed at nominal
}

# Electronic oxidation states for cross-section correction
PEAK_EOS = {
    'Exocyclic':    0.2,
    'Heterocyclic': 0.7,
    'Sulfoxide':    2.0,
    'Sulfone':      4.0,
    'Sulfonate':    5.0,
    'Sulfate':      6.0,
}

# Energy range for S K-edge XANES analysis
ENERGY_MIN = 2466.0
ENERGY_MAX = 2489.0

# Elemental sulfur reference energy for calibration
E_S8 = 2472.70


# ─── Primitive functions ────────────────────────────────────────────────────

def double_arctangent(x, center1, height1, width1, center2, height2, width2):
    """Double arctangent baseline for S K-edge."""
    step1 = height1 * (0.5 + (1 / np.pi) * np.arctan((x - center1) / abs(width1)))
    step2 = height2 * (0.5 + (1 / np.pi) * np.arctan((x - center2) / abs(width2)))
    return step1 + step2


def gaussian(x, center, height, fwhm):
    """Gaussian peak."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


# ─── Complete model ─────────────────────────────────────────────────────────

def total_model(x,
                arc1_center, arc1_height, arc1_width,
                arc2_center, arc2_height, arc2_width,
                c1, c2, c3, c4, c5, c6,
                h1, h2, h3, h4, h5, h6,
                red_fwhm, ox_fwhm):
    """
    Complete S K-edge model: double arctangent baseline + 6 Gaussian peaks.

    FWHM groups per Manceau & Nagy 2012:
      - red_fwhm: shared by peaks 1-3 (Exocyclic, Heterocyclic, Sulfoxide)
      - ox_fwhm:  shared by peaks 4-6 (Sulfone, Sulfonate, Sulfate)
    """
    baseline = double_arctangent(x, arc1_center, arc1_height, arc1_width,
                                 arc2_center, arc2_height, arc2_width)

    # Reduced S peaks: Exocyclic, Heterocyclic, Sulfoxide (shared FWHM)
    peaks = gaussian(x, c1, h1, red_fwhm)
    peaks += gaussian(x, c2, h2, red_fwhm)
    peaks += gaussian(x, c3, h3, red_fwhm)

    # Oxidized S peaks: Sulfone, Sulfonate, Sulfate (shared FWHM)
    peaks += gaussian(x, c4, h4, ox_fwhm)
    peaks += gaussian(x, c5, h5, ox_fwhm)
    peaks += gaussian(x, c6, h6, ox_fwhm)

    return baseline + peaks


# ─── Parameter estimation ──────────────────────────────────────────────────

def estimate_baseline_parameters(energy, intensity):
    """
    Estimate baseline parameters from data characteristics for S K-edge.

    Returns smart initial guesses and bounds based on actual data.
    """
    data_min = np.min(intensity)
    data_max = np.max(intensity)
    data_range = data_max - data_min

    # Pre-edge region (below ~2470 eV)
    pre_edge_mask = energy < 2470
    if np.any(pre_edge_mask):
        pre_edge_level = np.median(intensity[pre_edge_mask])
    else:
        pre_edge_level = data_min

    # Post-edge region (~2486-2489 eV)
    post_edge_mask = (energy >= 2486) & (energy <= 2489)
    if np.any(post_edge_mask):
        post_edge_level = np.median(intensity[post_edge_mask])
    else:
        post_edge_level = data_max * 0.8

    # Mid-edge region (~2476-2478 eV, between reduced and oxidized peaks)
    mid_edge_mask = (energy >= 2476) & (energy <= 2478)
    if np.any(mid_edge_mask):
        mid_edge_level = np.median(intensity[mid_edge_mask])
    else:
        mid_edge_level = (pre_edge_level + post_edge_level) / 2

    # Step heights: the total baseline step = post_edge - pre_edge.
    # Split between A1 and A2 roughly equally. Using mid-edge directly
    # overestimates A1 because it includes Gaussian peak contributions.
    total_step = max(0.01, post_edge_level - pre_edge_level)
    step1_height = total_step * 0.45
    step2_height = total_step * 0.55

    step1_height = max(0.01, min(step1_height, data_range * 0.8))
    step2_height = max(0.01, min(step2_height, data_range * 0.8))

    return {
        'data_min': data_min,
        'data_max': data_max,
        'data_range': data_range,
        'pre_edge_level': pre_edge_level,
        'post_edge_level': post_edge_level,
        'mid_edge_level': mid_edge_level,
        'step1_height': step1_height,
        'step2_height': step2_height,
    }


def setup_parameters_optimized(energy, intensity, fwhm_mode='two_group'):
    """
    Set up lmfit Parameters with data-adaptive constraints for S K-edge.

    Key constraints per Manceau & Nagy 2012:
      - A1 position: between heterocyclic S and sulfoxide
      - A2 position: above sulfate
      - FWHM mode: 'two_group' (reduced vs oxidized) or 'single' (all peaks share one FWHM)
      - Peak heights non-negative

    Parameters
    ----------
    fwhm_mode : str
        'two_group' — peaks 1-2 share red_fwhm, peaks 3-6 share ox_fwhm (default)
        'single' — all 6 peaks share one FWHM (as in Table EA-4 optimal fits)
    """
    params = Parameters()
    est = estimate_baseline_parameters(energy, intensity)

    # ── Baseline: Arctangent 1 (between heterocyclic and sulfoxide) ──
    # Paper fits: center 2475.51-2475.85 eV, FWHM 0.42-1.01
    # width parameter = FWHM/2, so range 0.21-0.51
    total_baseline = est['post_edge_level'] - est['pre_edge_level']
    max_arc_height = max(0.1, total_baseline * 0.75)
    # Paper Section 3.1.2: A1 must be between heterocyclic and sulfoxide
    # Table EA-4 fitted range: 2475.51-2475.85
    params.add('arc1_center', value=2475.7, min=2474.5, max=2476.4)
    params.add('arc1_height',
               value=est['step1_height'],
               min=total_baseline * 0.05,
               max=max_arc_height)
    # Covaried arc widths: both arctangents share one width (paper constraint)
    params.add('arc_width', value=0.4, min=0.15, max=0.6)
    params.add('arc1_width', expr='arc_width')

    # ── Baseline: Arctangent 2 (above sulfate) ──
    # Paper fits: center 2482.20-2484.40 eV, FWHM 0.42-1.01
    # Paper Section 3.1.2: A2 > sulfate when sulfate > sulfonate,
    # or between sulfonate and sulfate when sulfonate dominates
    # Table EA-4 fitted range: 2482.30-2484.40
    params.add('arc2_center', value=2483.5, min=2482.0, max=2484.5)
    params.add('arc2_height',
               value=est['step2_height'],
               min=total_baseline * 0.05,
               max=max_arc_height)
    params.add('arc2_width', expr='arc_width')

    # Constraint: combined baseline should approximate the total edge step
    params.add('baseline_total',
               expr='arc1_height + arc2_height',
               min=total_baseline * 0.5,
               max=total_baseline * 1.5)

    # ── Peak parameters ──
    max_peak_height = est['data_range'] * 2.0

    for i, name in enumerate(PEAK_NAMES, 1):
        center = DEFAULT_PEAK_CENTERS[name]
        prange = DEFAULT_PEAK_RANGES[name]
        if prange == 0:
            params.add(f'c{i}', value=center, vary=False)
        else:
            params.add(f'c{i}', value=center, min=center - prange, max=center + prange)

        # Data-driven height estimate: use intensity at peak center minus baseline
        center_mask = (energy >= center - 0.5) & (energy <= center + 0.5)
        if np.any(center_mask):
            local_intensity = np.max(intensity[center_mask]) - est['pre_edge_level']
            init_height = max(0.01, local_intensity * 0.5)
        else:
            init_height = est['data_range'] * 0.3
        params.add(f'h{i}', value=init_height, min=0, max=max_peak_height)

    # ── FWHM groups ──
    # Per Manceau & Nagy 2012:
    #   red_fwhm = peaks 1-3 (Exocyclic, Heterocyclic, Sulfoxide)
    #   ox_fwhm  = peaks 4-6 (Sulfone, Sulfonate, Sulfate)
    # 6 of 8 IHSS samples use single FWHM (2.12-2.39 eV);
    # ES-HA/ES-FA use two groups (red 1.55-1.75, ox 1.94-2.05)
    if fwhm_mode == 'single':
        # Single FWHM for all 6 peaks (Paper Table EA-4: 2.12-2.39 eV)
        params.add('shared_fwhm', value=2.2, min=1.5, max=2.5)
        params.add('red_fwhm', expr='shared_fwhm')
        params.add('ox_fwhm', expr='shared_fwhm')
    else:
        # Two-group FWHM (Paper Table EA-4: red 1.55-1.75, ox 1.94-2.05)
        params.add('red_fwhm', value=1.7, min=1.2, max=2.40)
        params.add('ox_fwhm', value=2.0, min=1.5, max=2.50)

    return params


# ─── File I/O ───────────────────────────────────────────────────────────────

def load_spectrum(file_path):
    """Load spectrum from various formats (.csv, .xmu, .xdi, .nor, .prj)."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(file_path)
        if 'energy' in df.columns and 'normalized_mu' in df.columns:
            energy = df['energy'].values
            intensity = df['normalized_mu'].values
        elif 'energy' in df.columns and 'y' in df.columns:
            energy = df['energy'].values
            intensity = df['y'].values
        else:
            energy = df.iloc[:, 0].values
            intensity = df.iloc[:, 1].values

    elif suffix in ['.xmu', '.xdi', '.nor']:
        data = np.loadtxt(file_path, comments='#')
        energy = data[:, 0]
        intensity = data[:, 1]

    else:
        data = np.loadtxt(file_path)
        energy = data[:, 0]
        intensity = data[:, 1]

    # Filter to analysis range
    mask = (energy >= ENERGY_MIN) & (energy <= ENERGY_MAX)
    return energy[mask], intensity[mask]


# ─── Fitting ────────────────────────────────────────────────────────────────

def fit_spectrum(energy, intensity, fwhm_mode='two_group'):
    """Fit S K-edge spectrum using lmfit with optimized parameters."""
    model = Model(total_model)
    params = setup_parameters_optimized(energy, intensity, fwhm_mode=fwhm_mode)
    result = model.fit(intensity, params, x=energy, method='leastsq', max_nfev=10000)
    return result


# ─── Quantification & Cross-Section Correction ─────────────────────────────

# Default calibration: Manceau & Nagy 2012 generic curve
# y = slope * E + intercept, normalized to 1.0 at E_S8
DEFAULT_CALIB_SLOPE = 0.36841
DEFAULT_CALIB_INTERCEPT = -909.97


def scaling_factor(peak_energy, slope=None, intercept=None):
    """
    Compute the absorption cross-section scaling factor for a given peak energy.

    The scaling factor accounts for the increase in 1s->3p transition
    probability with sulfur oxidation state. Raw Gaussian areas must be
    divided by this factor to obtain true fractional amounts of each
    sulfur functionality.

    Default: Manceau & Nagy 2012 generic calibration curve
        y = 0.36841 * E - 909.97
    normalized to 1.0 at elemental sulfur E = 2472.70 eV.

    Users can supply custom slope/intercept from their own standards.
    """
    if slope is None:
        slope = DEFAULT_CALIB_SLOPE
    if intercept is None:
        intercept = DEFAULT_CALIB_INTERCEPT

    raw = slope * peak_energy + intercept
    ref = slope * E_S8 + intercept  # normalize to 1.0 at S8
    if abs(ref) < 1e-10:
        return 1.0
    return raw / ref


def derive_calibration_curve(standard_results):
    """
    Derive a custom calibration curve from user-measured standard compounds.

    Parameters
    ----------
    standard_results : list of dict
        Each dict has:
          'name': str (compound name)
          'peak_energy': float (fitted white-line energy in eV)
          'peak_area': float (fitted Gaussian area)
        At least 2 standards spanning a range of oxidation states are needed.

    Returns
    -------
    dict with keys:
      'slope': float (calibration slope)
      'intercept': float (calibration intercept)
      'r_squared': float (quality of linear fit)
      'standards': list (the input data)
      'n_standards': int

    Usage
    -----
    To build a calibration curve from your own reference compounds:

    1. Fit each pure standard with the S K-edge model
    2. Record the dominant peak's fitted energy and raw Gaussian area
    3. Pass those as standard_results to this function
    4. Use the returned slope/intercept in calculate_peak_areas()

    The more standards you use (spanning from reduced to oxidized S),
    the more reliable the curve. Minimum 2 standards needed; 4+ recommended.
    """
    if len(standard_results) < 2:
        raise ValueError("Need at least 2 standards to derive a calibration curve")

    energies = np.array([s['peak_energy'] for s in standard_results])
    areas = np.array([s['peak_area'] for s in standard_results])

    # Linear fit: area = slope * energy + intercept
    coeffs = np.polyfit(energies, areas, 1)
    slope, intercept = coeffs

    # R² of the fit
    predicted = np.polyval(coeffs, energies)
    ss_res = np.sum((areas - predicted) ** 2)
    ss_tot = np.sum((areas - areas.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'standards': standard_results,
        'n_standards': len(standard_results),
    }


def calculate_peak_areas(energy, result, correct_cross_section=True,
                         calib_slope=None, calib_intercept=None):
    """
    Calculate peak areas using the analytical Gaussian integral.

    Area = height * fwhm * sqrt(pi / (4*ln(2)))

    If correct_cross_section=True, areas are divided by the energy-dependent
    scaling factor to obtain true fractional amounts.

    Parameters
    ----------
    calib_slope, calib_intercept : float, optional
        Custom calibration curve parameters. If None, uses the default
        Manceau & Nagy 2012 generic curve.
    """
    areas = {}
    area_factor = np.sqrt(np.pi / (4 * np.log(2)))

    for i, name in enumerate(PEAK_NAMES, 1):
        height = result.params[f'h{i}'].value
        center = result.params[f'c{i}'].value

        if i <= 3:
            fwhm = result.params['red_fwhm'].value
        else:
            fwhm = result.params['ox_fwhm'].value

        raw_area = height * fwhm * area_factor

        if correct_cross_section:
            sf = scaling_factor(center, slope=calib_slope, intercept=calib_intercept)
            areas[name] = raw_area / sf if sf > 0 else raw_area
        else:
            areas[name] = raw_area

    return areas


def calculate_peak_areas_raw(energy, result):
    """Calculate raw peak areas without cross-section correction."""
    return calculate_peak_areas(energy, result, correct_cross_section=False)


# ─── Validation ─────────────────────────────────────────────────────────────

def validate_baseline(energy, result, intensity):
    """
    Validate that the fitted baseline is physically reasonable.

    Returns (is_valid, issues) where issues is a list of problems found.
    """
    issues = []

    baseline = double_arctangent(
        energy,
        result.params['arc1_center'].value,
        result.params['arc1_height'].value,
        result.params['arc1_width'].value,
        result.params['arc2_center'].value,
        result.params['arc2_height'].value,
        result.params['arc2_width'].value)

    data_max = np.max(intensity)
    data_min = np.min(intensity)
    data_range = data_max - data_min
    baseline_max = np.max(baseline)
    baseline_min = np.min(baseline)

    if baseline_max > data_max + data_range * 0.2:
        issues.append(f"Baseline exceeds data max: {baseline_max:.3f} > {data_max:.3f}")

    if baseline_min < data_min - data_range * 0.2:
        issues.append(f"Baseline below data min: {baseline_min:.3f} < {data_min:.3f}")

    # Check arctangent ordering
    a1_center = result.params['arc1_center'].value
    a2_center = result.params['arc2_center'].value
    if a1_center >= a2_center:
        issues.append(f"A1 center ({a1_center:.2f}) >= A2 center ({a2_center:.2f})")

    # A1 should be between heterocyclic and sulfoxide
    if a1_center < 2474.0 or a1_center > 2477.0:
        issues.append(f"A1 center ({a1_center:.2f}) outside expected range (2474-2477 eV)")

    return len(issues) == 0, issues


# ─── Output ─────────────────────────────────────────────────────────────────

def print_summary(result, energy, intensity):
    """Print fit summary with cross-section corrected fractions."""
    r_squared = 1 - result.residual.var() / np.var(result.data)

    print("\n" + "=" * 80)
    print("S K-edge XANES PEAK FITTING SUMMARY (Manceau & Nagy 2012 GCF)")
    print("=" * 80)

    print(f"\nFit Quality:")
    print(f"  R^2 = {r_squared:.4f}")
    print(f"  Reduced chi^2 = {result.redchi:.4e}")
    print(f"  AIC = {result.aic:.2f}")
    print(f"  BIC = {result.bic:.2f}")

    is_valid, issues = validate_baseline(energy, result, intensity)
    print(f"\nBaseline Validation:")
    if is_valid:
        print("  Status: VALID")
    else:
        print("  Status: WARNING")
        for issue in issues:
            print(f"    - {issue}")

    # Raw areas (for display)
    raw_areas = calculate_peak_areas(energy, result, correct_cross_section=False)
    # Cross-section corrected areas (for quantification)
    corr_areas = calculate_peak_areas(energy, result, correct_cross_section=True)
    total_corr = sum(corr_areas.values())
    fractions = {k: v / total_corr * 100 for k, v in corr_areas.items()} if total_corr > 0 else {}

    print(f"\nPeak Parameters:")
    print(f"{'Peak':<16} {'Center (eV)':<12} {'FWHM (eV)':<12} "
          f"{'Scale Factor':<14} {'S_i (%)':<10}")
    print("-" * 80)

    for i, name in enumerate(PEAK_NAMES, 1):
        center = result.params[f'c{i}'].value
        fwhm = result.params['red_fwhm' if i <= 2 else 'ox_fwhm'].value
        sf = scaling_factor(center)
        frac = fractions.get(name, 0)
        print(f"{name:<16} {center:<12.3f} {fwhm:<12.3f} {sf:<14.3f} {frac:<10.1f}")

    print(f"\nBaseline Parameters:")
    print(f"  A1: center={result.params['arc1_center'].value:.2f} eV, "
          f"height={result.params['arc1_height'].value:.3f}, "
          f"width={result.params['arc1_width'].value:.3f}")
    print(f"  A2: center={result.params['arc2_center'].value:.2f} eV, "
          f"height={result.params['arc2_height'].value:.3f}, "
          f"width={result.params['arc2_width'].value:.3f}")

    print(f"\nTotal reduced S (Exo + Hetero): "
          f"{fractions.get('Exocyclic', 0) + fractions.get('Heterocyclic', 0):.1f}%")
    print(f"Total oxidized S: "
          f"{sum(fractions.get(n, 0) for n in ['Sulfoxide', 'Sulfone', 'Sulfonate', 'Sulfate']):.1f}%")
    print("=" * 80 + "\n")


def plot_fit(energy, intensity, result, title='S K-edge XANES Fit', save_path=None):
    """Create publication-quality plot of the fit."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Data and fit
    ax1.plot(energy, intensity, 'ko', markersize=3, alpha=0.6, label='Data')
    ax1.plot(energy, result.best_fit, 'r-', linewidth=2, label='Total Fit')

    # Baseline
    baseline = double_arctangent(
        energy,
        result.params['arc1_center'].value,
        result.params['arc1_height'].value,
        result.params['arc1_width'].value,
        result.params['arc2_center'].value,
        result.params['arc2_height'].value,
        result.params['arc2_width'].value)
    ax1.plot(energy, baseline, 'k--', linewidth=1.5, alpha=0.5, label='Baseline')

    # Individual peaks
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for i, (name, color) in enumerate(zip(PEAK_DISPLAY_NAMES, colors), 1):
        center = result.params[f'c{i}'].value
        height = result.params[f'h{i}'].value
        fwhm = result.params['red_fwhm' if i <= 2 else 'ox_fwhm'].value
        peak = gaussian(energy, center, height, fwhm)
        ax1.plot(energy, peak + baseline, '--', color=color,
                 linewidth=1, alpha=0.7, label=name)

    ax1.set_xlabel('Energy (eV)', fontsize=11)
    ax1.set_ylabel('Normalized Absorption', fontsize=11)

    is_valid, _ = validate_baseline(energy, result, intensity)
    status = "\u2713" if is_valid else "\u26A0"
    ax1.set_title(f'{title} {status}', fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(energy.min(), ENERGY_MAX)

    # Residuals
    r_squared = 1 - result.residual.var() / np.var(result.data)
    ax2.plot(energy, result.residual, 'b-', linewidth=1)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Energy (eV)', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title(f'Residuals (R\u00b2 = {r_squared:.4f})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(energy.min(), ENERGY_MAX)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


# ─── CLI entry point ────────────────────────────────────────────────────────

def main():
    """Command-line interface for S K-edge fitting."""
    parser = argparse.ArgumentParser(
        description='S K-edge XANES fitter (Manceau & Nagy 2012 GCF method)')
    parser.add_argument('input_file', help='Spectrum file (.csv, .xmu, etc.)')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Show plot')
    parser.add_argument('--no-correction', action='store_true',
                        help='Skip cross-section correction')

    args = parser.parse_args()

    print(f"Loading: {args.input_file}")
    energy, intensity = load_spectrum(args.input_file)
    print(f"Loaded {len(energy)} points ({energy.min():.2f} - {energy.max():.2f} eV)")

    est = estimate_baseline_parameters(energy, intensity)
    print(f"\nData characteristics:")
    print(f"  Min: {est['data_min']:.3f}")
    print(f"  Max: {est['data_max']:.3f}")
    print(f"  Range: {est['data_range']:.3f}")

    print("\nFitting with lmfit (Manceau & Nagy 2012 GCF)...")
    result = fit_spectrum(energy, intensity)

    print_summary(result, energy, intensity)

    input_path = Path(args.input_file)
    output_dir = Path(args.output) if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    plot_path = output_dir / f"{base_name}_s1s_fit.pdf"
    report_path = output_dir / f"{base_name}_s1s_report.txt"

    with open(report_path, 'w') as f:
        f.write(result.fit_report())
        f.write("\n\nBaseline Validation:\n")
        is_valid, issues = validate_baseline(energy, result, intensity)
        f.write(f"Status: {'VALID' if is_valid else 'WARNING'}\n")
        for issue in issues:
            f.write(f"  - {issue}\n")
    print(f"Fit report: {report_path}")

    plot_fit(energy, intensity, result, title=f'S K-edge: {base_name}',
             save_path=plot_path)

    if args.plot:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    main()
