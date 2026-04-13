"""
Extract normalized S K-edge XANES spectra from an Athena .prj file.

Athena project files are gzip-compressed Perl Data::Dumper output.
Each record has: $old_group, @args (metadata), @x (energy), @y (raw mu).

This script:
1. Parses the .prj file
2. Normalizes each spectrum using the stored Athena parameters
3. Exports individual CSV files (energy, normalized_mu)
"""

import gzip
import re
import os
import numpy as np
from pathlib import Path


def parse_athena_prj(filepath):
    """Parse an Athena .prj file and return list of spectrum dicts."""
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        content = f.read()

    spectra = []

    # Split by $old_group to get each record block
    blocks = re.split(r"\$old_group\s*=\s*'([^']+)';", content)
    # blocks[0] is header, then alternating: group_name, block_content

    for i in range(1, len(blocks), 2):
        group_name = blocks[i]
        block = blocks[i + 1] if i + 1 < len(blocks) else ""

        # Extract label
        label_match = re.search(r"'label','([^']*)'", block)
        label = label_match.group(1) if label_match else group_name

        # Extract @args as key-value pairs
        args_match = re.search(r"@args\s*=\s*\((.*?)\);", block, re.DOTALL)
        params = {}
        if args_match:
            args_str = args_match.group(1)
            # Parse key-value pairs from Perl list
            tokens = re.findall(r"'([^']*)'|([^,'\s]+)", args_str)
            flat = [t[0] if t[0] else t[1] for t in tokens]
            for j in range(0, len(flat) - 1, 2):
                params[flat[j]] = flat[j + 1]

        # Extract @x (energy) array
        x_match = re.search(r"@x\s*=\s*\(([^)]+)\)", block)
        if not x_match:
            continue
        x_vals = [float(v.strip("' ")) for v in x_match.group(1).split(",") if v.strip("' ")]

        # Extract @y (mu) array
        y_match = re.search(r"@y\s*=\s*\(([^)]+)\)", block)
        if not y_match:
            continue
        y_vals = [float(v.strip("' ")) for v in y_match.group(1).split(",") if v.strip("' ")]

        energy = np.array(x_vals)
        mu = np.array(y_vals)

        if len(energy) != len(mu):
            min_len = min(len(energy), len(mu))
            energy = energy[:min_len]
            mu = mu[:min_len]

        spectra.append({
            'group': group_name,
            'label': label,
            'energy': energy,
            'mu': mu,
            'params': params,
        })

    return spectra


def normalize_spectrum_athena(energy, mu, params):
    """
    Normalize a spectrum using Athena's stored parameters.

    Uses the stored pre-edge line (bkg_slope, bkg_int) and edge step
    (bkg_step / bkg_fitted_step) for standard XANES normalization:
        norm(E) = (mu(E) - pre_edge_line(E)) / edge_step
    """
    e0 = float(params.get('bkg_e0', 2472.0))
    eshift = float(params.get('bkg_eshift', 0))
    e0 += eshift

    # Pre-edge line: mu_pre = slope * E + intercept
    slope = float(params.get('bkg_slope', 0))
    intercept = float(params.get('bkg_int', 0))
    pre_edge = slope * energy + intercept

    # Remove pre-edge
    mu_sub = mu - pre_edge

    # Edge step for normalization
    step = float(params.get('bkg_step', 0))
    if step == 0 or abs(step) < 1e-12:
        step = float(params.get('bkg_fitted_step', 1.0))

    if abs(step) > 1e-12:
        normalized = mu_sub / step
    else:
        # Fallback: data-driven normalization
        pre1 = float(params.get('bkg_pre1', -20))
        pre2 = float(params.get('bkg_pre2', -10))
        nor1 = float(params.get('bkg_nor1', 25))
        nor2 = float(params.get('bkg_nor2', 50))

        pre_mask = (energy >= (e0 + pre1)) & (energy <= (e0 + pre2))
        post_mask = (energy >= (e0 + nor1)) & (energy <= (e0 + nor2))

        if np.sum(pre_mask) >= 2 and np.sum(post_mask) >= 2:
            pre_val = np.mean(mu[pre_mask])
            post_val = np.mean(mu[post_mask])
            fallback_step = post_val - pre_val
            if abs(fallback_step) > 1e-12:
                normalized = (mu - pre_val) / fallback_step
            else:
                normalized = mu_sub
        else:
            normalized = mu_sub

    return normalized


def extract_and_save(prj_path, output_dir, energy_range=(2465, 2500)):
    """Extract all spectra from an Athena .prj and save as CSV files."""
    prj_path = Path(prj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {prj_path}")
    spectra = parse_athena_prj(prj_path)
    print(f"Found {len(spectra)} spectra\n")

    summary = []
    for spec in spectra:
        label = spec['label']
        energy = spec['energy']
        mu = spec['mu']
        params = spec['params']

        # Normalize
        try:
            norm = normalize_spectrum_athena(energy, mu, params)
        except Exception as e:
            print(f"  WARNING: Could not normalize {label}: {e}")
            norm = mu

        # Trim to energy range
        if energy_range:
            mask = (energy >= energy_range[0]) & (energy <= energy_range[1])
            energy_trimmed = energy[mask]
            norm_trimmed = norm[mask]
        else:
            energy_trimmed = energy
            norm_trimmed = norm

        if len(energy_trimmed) == 0:
            print(f"  SKIPPED {label}: no data in energy range {energy_range}")
            continue

        # Save CSV
        safe_label = re.sub(r'[^\w\-.]', '_', label)
        csv_path = output_dir / f"{safe_label}.csv"
        header = "energy,normalized_mu"
        np.savetxt(csv_path, np.column_stack([energy_trimmed, norm_trimmed]),
                   delimiter=',', header=header, comments='', fmt='%.6f')

        e0 = float(params.get('bkg_e0', 0))
        summary.append({
            'label': label,
            'file': csv_path.name,
            'e0': e0,
            'n_points': len(energy_trimmed),
            'e_min': energy_trimmed.min(),
            'e_max': energy_trimmed.max(),
        })
        print(f"  Saved: {csv_path.name} ({len(energy_trimmed)} points, "
              f"E0={e0:.1f} eV)")

    # Save summary
    summary_path = output_dir / "_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"{'Label':<45} {'File':<50} {'E0':>8} {'Points':>7} "
                f"{'E_min':>10} {'E_max':>10}\n")
        f.write("-" * 135 + "\n")
        for s in summary:
            f.write(f"{s['label']:<45} {s['file']:<50} {s['e0']:>8.1f} "
                    f"{s['n_points']:>7d} {s['e_min']:>10.2f} {s['e_max']:>10.2f}\n")

    print(f"\nExtracted {len(summary)} spectra to {output_dir}")
    print(f"Summary: {summary_path}")
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract spectra from Athena .prj file')
    parser.add_argument('prj_file', nargs='?',
                        default='reference_data/Sulfur_XANES_Database_Cron_Jan2017_share.prj',
                        help='Path to Athena .prj file')
    parser.add_argument('--output', '-o', default='reference_data/spectra',
                        help='Output directory for CSV files')
    parser.add_argument('--emin', type=float, default=2465,
                        help='Minimum energy (eV)')
    parser.add_argument('--emax', type=float, default=2500,
                        help='Maximum energy (eV)')
    args = parser.parse_args()

    extract_and_save(args.prj_file, args.output, (args.emin, args.emax))
