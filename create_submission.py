"""
Create submission package for STAT 8289 RL Project
Includes: paper, code, essential files
Excludes: large data files, temporary files
"""
import os
import zipfile
from pathlib import Path

# Define base directory
base_dir = Path(__file__).parent

# Define what to include
include_patterns = [
    # Required: Paper
    "paper/main.pdf",

    # Required: Code - Source files
    "src/**/*.py",

    # Required: Code - Scripts
    "scripts/**/*.py",

    # Essential notebooks (key analysis notebooks)
    "notebooks/LEG_interplate_v2.ipynb",

    # Configuration and requirements
    "README.md",
    "requirements.txt",
    "gym-sepsis/requirements.txt",

    # Gym-sepsis environment (essential for reproduction)
    "gym-sepsis/**/*.py",
    "gym-sepsis/setup.py",
    "gym-sepsis/README.md",

    # Results - figures only (not large model files)
    "results/figures/**/*.png",
    "results/figures/**/*.pdf",

    # Small result files
    "results/*.pkl",
]

# Define what to explicitly exclude
exclude_patterns = [
    "**/data/**",  # Large data files
    "**/*.d3",     # Large trained models
    "**/__pycache__/**",
    "**/.git/**",
    "**/.ipynb_checkpoints/**",
    "**/records/**",
    "**/prompts/**",
    "**/jasa_template/**",
    "**/Yalun_Ideas/**",
    "**/github_models/**",
    "**/*.egg-info/**",
    "**/temp_*.py",
    "**/test_*.py",
    "**/extract_*.py",
    "**/check_*.py",
]

def should_exclude(path_str):
    """Check if a path should be excluded"""
    path_str = path_str.replace('\\', '/')
    for pattern in exclude_patterns:
        pattern = pattern.replace('**/', '').replace('/**', '').replace('*', '')
        if pattern in path_str:
            return True
    return False

def get_all_files():
    """Get all files to include"""
    files_to_zip = set()

    # Always include these specific files
    must_include = [
        "paper/main.pdf",
        "README.md",
        "requirements.txt",
    ]

    for file_path in must_include:
        full_path = base_dir / file_path
        if full_path.exists():
            files_to_zip.add(full_path)

    # Include source and scripts directories
    for pattern in ["src", "scripts", "gym-sepsis"]:
        pattern_dir = base_dir / pattern
        if pattern_dir.exists():
            for root, dirs, files in os.walk(pattern_dir):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]

                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        if not should_exclude(str(file_path)):
                            files_to_zip.add(file_path)

    # Include key notebooks
    notebooks_dir = base_dir / "notebooks"
    if notebooks_dir.exists():
        for file in notebooks_dir.glob("*.ipynb"):
            if "LEG_interplate" in file.name:
                files_to_zip.add(file)

    # Include result figures (not models)
    results_figures = base_dir / "results" / "figures"
    if results_figures.exists():
        for root, dirs, files in os.walk(results_figures):
            for file in files:
                if file.endswith(('.png', '.pdf')):
                    files_to_zip.add(Path(root) / file)

    # Include small result files
    results_dir = base_dir / "results"
    if results_dir.exists():
        for file in results_dir.glob("*.pkl"):
            file_size = file.stat().st_size / (1024 * 1024)  # Size in MB
            if file_size < 10:  # Only include files smaller than 10MB
                files_to_zip.add(file)

    return sorted(files_to_zip)

def create_zip():
    """Create the submission ZIP file"""
    output_file = base_dir / "STAT8289_Project_Submission.zip"

    print("Collecting files for submission...")
    files_to_zip = get_all_files()

    print(f"\nCreating ZIP file: {output_file.name}")
    print(f"Total files to include: {len(files_to_zip)}\n")

    total_size = 0
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            if file_path.exists():
                # Get relative path for ZIP archive
                rel_path = file_path.relative_to(base_dir)

                # Add file to ZIP
                zipf.write(file_path, arcname=rel_path)

                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb

                # Use ASCII-safe printing
                try:
                    print(f"Added: {rel_path} ({size_mb:.2f} MB)")
                except UnicodeEncodeError:
                    print(f"Added: {rel_path.name} ({size_mb:.2f} MB)")

    zip_size = output_file.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print("[SUCCESS] Submission package created successfully!")
    print(f"  Output: {output_file.name}")
    print(f"  Files included: {len(files_to_zip)}")
    print(f"  Uncompressed size: {total_size:.1f} MB")
    print(f"  Compressed size: {zip_size:.1f} MB")
    print(f"{'='*60}")

    # Verify paper is included
    with zipfile.ZipFile(output_file, 'r') as zipf:
        files_in_zip = zipf.namelist()
        if 'paper/main.pdf' in files_in_zip:
            print("[OK] Paper PDF included")
        else:
            print("[WARNING] Paper PDF not found!")

        py_files = [f for f in files_in_zip if f.endswith('.py')]
        print(f"[OK] {len(py_files)} Python files included")

    return output_file

if __name__ == "__main__":
    print("="*60)
    print("STAT 8289 RL Project - Submission Package Creator")
    print("="*60)
    print("\nSubmission Requirements:")
    print("  1. Project Report (Required): PDF, max 25 pages")
    print("  2. Code File (Required): Single file or ZIP")
    print("  3. Supplementary (Optional): Not included\n")

    output_file = create_zip()

    print("\n" + "="*60)
    print("Next steps:")
    print(f"  1. Review the ZIP file: {output_file.name}")
    print("  2. Verify paper/main.pdf is included")
    print("  3. Test that code can run from the ZIP")
    print("  4. Submit before deadline: Oct 27th, 11:59 PM EST")
    print("="*60)
