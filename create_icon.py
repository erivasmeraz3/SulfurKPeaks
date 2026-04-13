"""
Create SulfurKPeaks icon with sulfurous yellow Gaussian curves.
Generates a spectral peak logo similar to CarbonKPeaks but in yellow tones.
"""

import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow', '-q'])
    from PIL import Image, ImageDraw, ImageFont, ImageFilter


def gaussian(x, center, height, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def arctangent_step(x, center, height, width):
    return height * (0.5 + (1 / np.pi) * np.arctan((x - center) / width))


def create_sulfur_icon(size=256):
    """Create a sulfurous yellow spectral peak icon."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Margins
    margin = int(size * 0.08)
    plot_w = size - 2 * margin
    plot_h = size - 2 * margin

    # Energy axis (normalized 0-1 mapped to pixel x)
    n_points = 500
    x = np.linspace(0, 1, n_points)

    # Simulated S K-edge peaks (normalized positions)
    # Exocyclic ~0.15, Heterocyclic ~0.20, Sulfoxide ~0.30,
    # Sulfone ~0.55, Sulfonate ~0.62, Sulfate ~0.70
    baseline = arctangent_step(x, 0.35, 0.15, 0.04) + arctangent_step(x, 0.72, 0.20, 0.03)

    peaks = (
        gaussian(x, 0.15, 0.45, 0.06) +   # Exocyclic (tall, sharp)
        gaussian(x, 0.22, 0.55, 0.07) +    # Heterocyclic (tallest)
        gaussian(x, 0.33, 0.20, 0.08) +    # Sulfoxide
        gaussian(x, 0.55, 0.15, 0.09) +    # Sulfone
        gaussian(x, 0.64, 0.25, 0.08) +    # Sulfonate
        gaussian(x, 0.72, 0.60, 0.07)      # Sulfate (tall, sharp)
    )

    total = baseline + peaks
    total = total / total.max()  # Normalize to 0-1

    # Convert to pixel coordinates
    def to_px(xf, yf):
        px = margin + int(xf * plot_w)
        py = margin + int((1.0 - yf) * plot_h)
        return px, py

    # Sulfurous yellow color palette
    # Peak fills from warm yellow to golden amber
    peak_defs = [
        (0.15, 0.45, 0.06, (255, 230, 50, 140)),    # Bright yellow (Exocyclic)
        (0.22, 0.55, 0.07, (240, 200, 30, 140)),     # Golden yellow (Heterocyclic)
        (0.33, 0.20, 0.08, (220, 180, 20, 140)),     # Dark gold (Sulfoxide)
        (0.55, 0.15, 0.09, (200, 160, 10, 130)),     # Amber (Sulfone)
        (0.64, 0.25, 0.08, (230, 190, 40, 140)),     # Warm gold (Sulfonate)
        (0.72, 0.60, 0.07, (255, 210, 60, 150)),     # Bright gold (Sulfate)
    ]

    # Draw filled peaks (from back to front)
    for center, height, fwhm, color in reversed(peak_defs):
        peak = gaussian(x, center, height, fwhm)
        peak_total = baseline + peak

        # Build polygon points for this peak
        points = []
        for j in range(n_points):
            if peak[j] > 0.005:  # Only where peak is visible
                px, py = to_px(x[j], (baseline[j] + peak[j]) / total.max())
                points.append((px, py))

        if len(points) > 2:
            # Close polygon along baseline
            for j in range(n_points - 1, -1, -1):
                if gaussian(x[j], center, height, fwhm) > 0.005:
                    px, py = to_px(x[j], baseline[j] / total.max())
                    points.append((px, py))

            overlay = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.polygon(points, fill=color)
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

    # Draw baseline as dark line
    baseline_points = [to_px(x[j], baseline[j] / total.max()) for j in range(n_points)]
    for j in range(len(baseline_points) - 1):
        draw.line([baseline_points[j], baseline_points[j + 1]],
                  fill=(80, 60, 10, 180), width=max(1, size // 128))

    # Draw total spectrum as bold dark outline
    total_points = [to_px(x[j], total[j]) for j in range(n_points)]
    line_width = max(2, size // 80)
    for j in range(len(total_points) - 1):
        draw.line([total_points[j], total_points[j + 1]],
                  fill=(100, 70, 0, 255), width=line_width)

    # Draw a thin red dashed fit line (slightly offset for visual interest)
    fit_offset = total * 0.99 + 0.005
    fit_points = [to_px(x[j], fit_offset[j]) for j in range(n_points)]
    for j in range(0, len(fit_points) - 1, 3):  # Dashed effect
        draw.line([fit_points[j], fit_points[min(j + 2, len(fit_points) - 1)]],
                  fill=(180, 30, 30, 200), width=max(1, size // 128))

    return img


def create_ico_pillow(images, ico_path):
    """Create ICO using Pillow."""
    sorted_images = sorted(images, key=lambda x: x.size[0])
    base = sorted_images[-1]
    others = sorted_images[:-1]
    base.save(ico_path, format='ICO', append_images=others, bitmap_format='bmp')


def main():
    output_dir = Path(__file__).parent
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = []

    print("Creating SulfurKPeaks icons...")
    for size in sizes:
        print(f"  Generating {size}x{size}...")
        img = create_sulfur_icon(size)
        images.append(img)

        # Save individual PNG
        png_path = output_dir / f"sulfurpeaks_{size}.png"
        img.save(png_path, format='PNG')

    # Save ICO
    ico_path = output_dir / "sulfurpeaks.ico"
    print(f"\nCreating ICO: {ico_path}")
    create_ico_pillow(images, ico_path)

    print("Done!")


if __name__ == '__main__':
    main()
