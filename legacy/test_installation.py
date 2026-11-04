"""
Test script to verify Phase-1 installation and dependencies
Run this before executing main_pipeline.py
"""

import sys
import importlib

def test_imports():
    """Test all required imports."""
    print("=" * 80)
    print("TESTING PHASE-1 INSTALLATION".center(80))
    print("=" * 80)
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('torch', 'PyTorch'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
    ]
    
    print("\n1. Checking Python packages...")
    print("-" * 80)
    
    all_ok = True
    
    for module_name, display_name in required_packages:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {display_name:<20} {version}")
        except ImportError:
            print(f"  ✗ {display_name:<20} NOT INSTALLED")
            all_ok = False
    
    return all_ok


def test_pytorch():
    """Test PyTorch and CUDA availability."""
    print("\n2. Testing PyTorch configuration...")
    print("-" * 80)
    
    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ℹ Using CPU (training will be slower but works fine)")
        return True
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")
        return False


def test_data_files():
    """Test if NASA data files are present."""
    import os
    
    print("\n3. Checking data files...")
    print("-" * 80)
    
    # Check NASA dataset
    nasa_path = "NASA_DataSets/1. BatteryAgingARC-FY08Q4/B0005.mat"
    
    if os.path.exists(nasa_path):
        file_size = os.path.getsize(nasa_path) / (1024 * 1024)  # MB
        print(f"  ✓ B0005.mat found ({file_size:.2f} MB)")
        return True
    else:
        print(f"  ✗ B0005.mat NOT FOUND")
        print(f"    Expected location: {nasa_path}")
        print(f"    Please download NASA Battery Dataset")
        return False


def test_config():
    """Test configuration file."""
    import os
    
    print("\n4. Checking configuration...")
    print("-" * 80)
    
    if os.path.exists('config.yaml'):
        print(f"  ✓ config.yaml found")
        
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Check key sections
            required_sections = ['dataset', 'filtering', 'features', 'model', 'paths']
            for section in required_sections:
                if section in config:
                    print(f"  ✓ Section '{section}' present")
                else:
                    print(f"  ✗ Section '{section}' missing")
                    return False
            
            return True
        
        except Exception as e:
            print(f"  ✗ Error loading config: {e}")
            return False
    else:
        print(f"  ✗ config.yaml NOT FOUND")
        return False


def test_source_code():
    """Test if source code modules can be imported."""
    print("\n5. Testing source code modules...")
    print("-" * 80)
    
    sys.path.append('src')
    
    modules = [
        'utils',
        'data_loader',
        'cycle_segmentation',
        'feature_extraction',
        'baseline_model'
    ]
    
    all_ok = True
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {module_name}.py")
        except ImportError as e:
            print(f"  ✗ {module_name}.py: {e}")
            all_ok = False
    
    return all_ok


def test_directories():
    """Test if output directories will be created."""
    print("\n6. Checking directory structure...")
    print("-" * 80)
    
    required_dirs = [
        'src',
        'notebooks',
        'data',
        'models',
        'results'
    ]
    
    import os
    
    all_ok = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ⚠ {dir_name}/ (will be created)")
    
    return all_ok


def main():
    """Run all tests."""
    
    results = {
        "Packages": test_imports(),
        "PyTorch": test_pytorch(),
        "Data Files": test_data_files(),
        "Configuration": test_config(),
        "Source Code": test_source_code(),
        "Directories": test_directories()
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY".center(80))
    print("=" * 80)
    
    all_passed = True
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python main_pipeline.py")
        print("\nOr explore with Jupyter:")
        print("  jupyter notebook")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Download NASA B0005.mat dataset")
        print("  - Check file paths in config.yaml")
    
    print()


if __name__ == "__main__":
    main()
