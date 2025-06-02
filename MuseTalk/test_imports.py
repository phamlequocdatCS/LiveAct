# L:\1 Things\Code11\LiveAct\MuseTalk\test_imports.py
import os
import sys

print(f"Current CWD: {os.getcwd()}") # Should be MuseTalk
print(f"sys.path: {sys.path}")

print("\n--- Checking directory contents ---")
muse_talk_dir = os.getcwd()
scripts_dir_path = os.path.join(muse_talk_dir, "scripts")
scripts_utils_dir_path = os.path.join(scripts_dir_path, "utils")

print(f"Does 'scripts' directory exist? {os.path.isdir(scripts_dir_path)}")
if os.path.isdir(scripts_dir_path):
    print(f"Contents of 'scripts': {os.listdir(scripts_dir_path)}")
    scripts_init_path = os.path.join(scripts_dir_path, "__init__.py")
    print(f"Does 'scripts/__init__.py' exist? {os.path.isfile(scripts_init_path)}")

print(f"Does 'scripts/utils' directory exist? {os.path.isdir(scripts_utils_dir_path)}")
if os.path.isdir(scripts_utils_dir_path):
    print(f"Contents of 'scripts/utils': {os.listdir(scripts_utils_dir_path)}")
    scripts_utils_init_path = os.path.join(scripts_utils_dir_path, "__init__.py")
    print(f"Does 'scripts/utils/__init__.py' exist? {os.path.isfile(scripts_utils_init_path)}")
    scripts_utils_utils_py_path = os.path.join(scripts_utils_dir_path, "utils.py")
    print(f"Does 'scripts/utils/utils.py' exist? {os.path.isfile(scripts_utils_utils_py_path)}")


print("\n--- Attempting imports ---")
try:
    print("1. Attempting: import scripts")
    import scripts # Try to import the top-level package first
    print("   SUCCESS: import scripts")
    print(f"   scripts module location: {scripts.__file__ if hasattr(scripts, '__file__') else 'Namespace package'}")

    try:
        print("2. Attempting: from scripts import utils")
        from scripts import utils # Then try to import a submodule
        print("   SUCCESS: from scripts import utils")
        print(f"   scripts.utils module location: {utils.__file__ if hasattr(utils, '__file__') else 'Namespace package'}")

        try:
            print("3. Attempting: from scripts.utils import utils as utils_module") # Import the .py file itself
            from scripts.utils import utils as utils_module
            print("   SUCCESS: from scripts.utils import utils as utils_module")
            print(f"   scripts.utils.utils module location: {utils_module.__file__}")

            try:
                print("4. Attempting: from scripts.utils.utils import load_model_unet")
                from scripts.utils.utils import load_model_unet
                print("   SUCCESS: from scripts.utils.utils import load_model_unet")
            except ImportError as e4:
                print(f"   FAIL (4): {e4} (Is load_model_unet actually in scripts.utils.utils.py?)")

        except ImportError as e3:
            print(f"   FAIL (3): {e3} (Cannot import scripts.utils.utils module itself)")

    except ImportError as e2:
        print(f"   FAIL (2): {e2} (Cannot import scripts.utils submodule)")

except ImportError as e1:
    print(f"   FAIL (1): {e1} (Cannot import scripts package)")
except Exception as e_other:
    print(f"   UNEXPECTED ERROR: {e_other}")