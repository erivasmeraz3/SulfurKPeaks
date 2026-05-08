"""
Build script for SulfurKPeaks application.
Creates a standalone executable with embedded icon.
"""

import subprocess
import sys
from pathlib import Path

def install_pyinstaller():
    """Ensure PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller', '-q'])

def build():
    """Build the executable."""
    install_pyinstaller()

    base_dir = Path(__file__).parent
    main_script = base_dir / "s1s_peak_viewer_gui_final.py"
    icon_file = base_dir / "sulfurpeaks.ico"

    # Splash image for PyInstaller bootloader
    splash_image = None
    for size in [256, 128, 64]:
        candidate = base_dir / f"sulfurpeaks_{size}.png"
        if candidate.exists():
            splash_image = candidate
            break

    # Data files to include (icon PNGs for runtime)
    data_files = []
    for png in base_dir.glob("sulfurpeaks_*.png"):
        data_files.append(f"--add-data={png};.")
    if icon_file.exists():
        data_files.append(f"--add-data={icon_file};.")

    # Include reference data and documentation
    ref_dir = base_dir / "reference_data" / "ihss_manceau2012"
    if ref_dir.exists():
        data_files.append(f"--add-data={ref_dir};reference_data/ihss_manceau2012")
    gcf_doc = base_dir / "GCF_METHOD.md"
    if gcf_doc.exists():
        data_files.append(f"--add-data={gcf_doc};.")

    # Hidden imports that PyInstaller might miss
    hidden_imports = [
        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=scipy.special._cdflib',
        '--hidden-import=scipy._lib.array_api_compat.numpy.fft',
        '--hidden-import=lmfit',
        '--hidden-import=lmfit.models',
    ]

    # PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name=SulfurKPeaks',
        '--onefile',
        '--windowed',
        f'--icon={icon_file}',
        '--clean',
        '--noconfirm',
    ]

    if splash_image is not None:
        cmd.append(f'--splash={splash_image}')

    cmd += data_files + hidden_imports + [str(main_script)]

    print("\nBuilding SulfurKPeaks...")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(base_dir))

    if result.returncode == 0:
        exe_path = base_dir / "dist" / "SulfurKPeaks.exe"
        print(f"\n{'='*60}")
        print(f"Build successful!")
        print(f"Executable: {exe_path}")
        print(f"{'='*60}")
    else:
        print(f"\nBuild failed with return code {result.returncode}")

if __name__ == '__main__':
    build()
