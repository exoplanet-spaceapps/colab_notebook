"""
Google Colab Setup & Best Practices - October 2025
Quick reference code snippets for common operations
"""

# ============================================================================
# 1. ENVIRONMENT VERSION CHECK
# ============================================================================

def check_environment():
    """Check all package versions and Python version"""
    import sys
    import numpy as np
    import pandas as pd
    import sklearn
    import matplotlib
    import seaborn as sns

    print("=" * 60)
    print("GOOGLE COLAB ENVIRONMENT - OCTOBER 2025")
    print("=" * 60)

    print(f"\n📍 Python Version:")
    print(f"   {sys.version}")

    print(f"\n📦 Core Data Science Packages:")
    print(f"   NumPy:        {np.__version__}")
    print(f"   Pandas:       {pd.__version__}")
    print(f"   Scikit-learn: {sklearn.__version__}")

    print(f"\n📊 Visualization Packages:")
    print(f"   Matplotlib:   {matplotlib.__version__}")
    print(f"   Seaborn:      {sns.__version__}")

    print("\n" + "=" * 60)
    print("✅ Environment check complete!")
    print("=" * 60)

    # Compatibility warnings
    if np.__version__ < "2.0.0":
        print("\n⚠️  WARNING: NumPy < 2.0.0 - consider upgrading")
    if pd.__version__ < "2.2.0":
        print("\n⚠️  WARNING: Pandas < 2.2.0 - consider upgrading")


# ============================================================================
# 2. FILE UPLOAD - GOOGLE DRIVE (RECOMMENDED)
# ============================================================================

def setup_google_drive():
    """Mount Google Drive for persistent storage"""
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted at /content/drive")
    print("📁 Access your files at: /content/drive/MyDrive/")


def copy_from_drive_to_local(drive_path, local_path='/content/data'):
    """
    Copy data from Drive to local storage for faster access
    CRITICAL: Always do this before training models (10x faster)

    Args:
        drive_path: Path in Google Drive (e.g., '/content/drive/MyDrive/data')
        local_path: Local destination path
    """
    import os
    os.makedirs(local_path, exist_ok=True)

    # Copy files
    !cp -r {drive_path} {local_path}
    print(f"✅ Data copied from {drive_path} to {local_path}")
    print("⚡ Training on local data will be ~10x faster than Drive")


# ============================================================================
# 3. FILE UPLOAD - EXTERNAL URLs
# ============================================================================

def download_from_url(url, output_name=None):
    """
    Download file from external URL using wget
    Faster than manual upload - Colab backend downloads directly

    Args:
        url: URL to download from
        output_name: Optional output filename
    """
    if output_name:
        !wget {url} -O {output_name}
        print(f"✅ Downloaded to {output_name}")
    else:
        !wget {url}
        print(f"✅ Downloaded from {url}")


def download_from_github(user, repo, branch, filepath, output_name=None):
    """
    Download specific file from GitHub repository

    Args:
        user: GitHub username
        repo: Repository name
        branch: Branch name (usually 'main' or 'master')
        filepath: Path to file in repo
        output_name: Optional output filename
    """
    url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{filepath}"
    download_from_url(url, output_name)


# ============================================================================
# 4. FILE UPLOAD - UI & PYTHON
# ============================================================================

def upload_files_ui():
    """Upload files using UI (Chrome only, lost on disconnect)"""
    from google.colab import files
    uploaded = files.upload()
    print(f"✅ Uploaded {len(uploaded)} file(s)")
    return uploaded


def upload_files_programmatic():
    """Upload files programmatically"""
    from google.colab import files
    import io
    import pandas as pd

    print("📤 Please select files to upload...")
    uploaded = files.upload()

    # Example: Load CSV files
    dataframes = {}
    for filename in uploaded.keys():
        if filename.endswith('.csv'):
            dataframes[filename] = pd.read_csv(io.BytesIO(uploaded[filename]))
            print(f"✅ Loaded {filename} as DataFrame")

    return dataframes


# ============================================================================
# 5. PACKAGE UPGRADE
# ============================================================================

def upgrade_packages(packages=['numpy', 'pandas', 'scikit-learn']):
    """
    Upgrade specified packages to latest versions

    Args:
        packages: List of package names to upgrade
    """
    for package in packages:
        print(f"⬆️  Upgrading {package}...")
        !pip install {package} --upgrade --quiet
        print(f"✅ {package} upgraded")

    print("\n⚠️  Remember to restart runtime after upgrading packages!")
    print("    Runtime -> Restart runtime")


# ============================================================================
# 6. PANDAS 2.2+ BEST PRACTICES
# ============================================================================

def pandas_best_practices_example():
    """Demonstrate pandas 2.2+ best practices"""
    import pandas as pd
    import numpy as np

    # Create sample DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'x']
    })

    print("📊 Pandas 2.2+ Best Practices\n")

    # ✅ CORRECT: Use .loc for conditional assignment
    print("✅ CORRECT: Using .loc for assignment")
    df.loc[df['A'] > 2, 'B'] = 999
    print(df)
    print()

    # ❌ AVOID: Chained assignment (deprecated)
    print("❌ AVOID: Chained assignment (deprecated)")
    print("   df[df['A'] > 2]['B'] = 999  # Don't do this!")
    print()

    # ✅ CORRECT: GroupBy operations
    print("✅ CORRECT: GroupBy with ffill/bfill")
    df_with_na = df.copy()
    df_with_na.loc[2, 'B'] = np.nan
    result = df_with_na.groupby('C').ffill()
    print(result)
    print()

    # ❌ AVOID: GroupBy.fillna() (deprecated)
    print("❌ AVOID: grouped.fillna() - deprecated")
    print()

    return df


# ============================================================================
# 7. NUMPY 2.0+ TYPE HANDLING
# ============================================================================

def numpy_type_precision_example():
    """Demonstrate NumPy 2.0 type precision preservation"""
    import numpy as np

    print("🔢 NumPy 2.0 Type Precision Preservation\n")

    # NumPy 2.0: Scalar precision is preserved
    result_float32 = np.float32(3) + 3.
    print(f"np.float32(3) + 3. = {result_float32}")
    print(f"Result type: {type(result_float32).__name__}")
    print(f"Result dtype: {result_float32.dtype}")
    print("✅ In NumPy 2.0, this returns float32 (was float64 in 1.x)")
    print()

    # Be explicit with dtypes
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    print(f"Explicit dtype array: {arr.dtype}")
    print("✅ Always be explicit with dtypes for predictable behavior")


# ============================================================================
# 8. COMPLETE SETUP WORKFLOW
# ============================================================================

def complete_colab_setup(drive_data_path=None):
    """
    Complete Colab setup workflow

    Args:
        drive_data_path: Optional path to data in Google Drive
    """
    print("🚀 Starting Google Colab Setup (October 2025)\n")

    # Step 1: Check environment
    print("Step 1: Checking environment...")
    check_environment()
    print()

    # Step 2: Mount Google Drive
    print("Step 2: Mounting Google Drive...")
    setup_google_drive()
    print()

    # Step 3: Copy data to local if provided
    if drive_data_path:
        print("Step 3: Copying data to local storage...")
        copy_from_drive_to_local(drive_data_path)
        print()

    print("✅ Setup complete!")
    print("\n📝 Next steps:")
    print("   1. Load your data from /content/data (if copied)")
    print("   2. Start your analysis or model training")
    print("   3. Remember: Train on local data, not Drive-mounted data")


# ============================================================================
# 9. PERFORMANCE OPTIMIZATION
# ============================================================================

def performance_tips():
    """Print performance optimization tips"""
    tips = """
    ⚡ PERFORMANCE OPTIMIZATION TIPS FOR GOOGLE COLAB 2025

    1. 🚫 NEVER train models on Drive-mounted data
       ✅ Always copy to local first (10x faster)

    2. 📦 Use compressed archives for large datasets
       ✅ Upload .zip to Drive, then unzip in Colab

    3. 🗂️  Organize files efficiently
       ✅ 100 archives of 1000 files each
       ❌ 100,000 individual files

    4. ⏰ Remember timeouts
       - Idle: 90 minutes
       - Absolute: 12 hours
       - Save progress frequently!

    5. 💾 Storage limits
       - Total: 108 GB
       - Available: 77 GB
       - Plan accordingly

    6. 🔄 Use runtime pinning for reproducibility
       - Pin to specific version (e.g., 2025.07)
       - Ensures consistent package versions

    7. 📥 Download method priority
       1. Google Drive (persistent)
       2. wget (fast)
       3. UI upload (convenient)
       4. Python upload (last resort)
    """
    print(tips)


# ============================================================================
# 10. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GOOGLE COLAB SETUP SNIPPETS - OCTOBER 2025")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - check_environment()")
    print("  - setup_google_drive()")
    print("  - copy_from_drive_to_local(drive_path, local_path)")
    print("  - download_from_url(url, output_name)")
    print("  - download_from_github(user, repo, branch, filepath)")
    print("  - upload_files_ui()")
    print("  - upgrade_packages(packages)")
    print("  - pandas_best_practices_example()")
    print("  - numpy_type_precision_example()")
    print("  - complete_colab_setup(drive_data_path)")
    print("  - performance_tips()")
    print("\n" + "=" * 70)

    # Run environment check
    check_environment()


# ============================================================================
# QUICK START TEMPLATE
# ============================================================================

"""
QUICK START TEMPLATE - Copy this to your Colab notebook:

# 1. Check environment
import sys
import numpy as np
import pandas as pd
print(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
print(f"NumPy: {np.__version__}, Pandas: {pd.__version__}")

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Copy data to local (CRITICAL for performance)
!cp -r /content/drive/MyDrive/your_data /content/data

# 4. Load data from local storage
import pandas as pd
df = pd.read_csv('/content/data/your_file.csv')

# 5. Start your analysis
# Your code here...

"""
