"""
Quick test script to verify TensorFlow environment and basic functionality.
"""

import sys
import os

print("=" * 60)
print("TensorFlow Environment Test")
print("=" * 60)

# Test 1: TensorFlow import
print("\n[Test 1] Testing TensorFlow import...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"✓ GPUs available: {len(gpus)}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
except Exception as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

# Test 2: PyTorch import (needed for model)
print("\n[Test 2] Testing PyTorch import...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test 3: Other dependencies
print("\n[Test 3] Testing other dependencies...")
deps = [
    'transformers',
    'PIL',
    'numpy',
    'matplotlib',
    'seaborn',
    'tqdm'
]

for dep in deps:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError:
        print(f"✗ {dep} not found")

# Test 4: Check directory structure
print("\n[Test 4] Checking directory structure...")
required_dirs = [
    'llava_tf',
    'llava_tf_utils',
    'harmful_corpus',
    'adversarial_images'
]

for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"✓ {dir_name}/")
    else:
        print(f"✗ {dir_name}/ not found")

# Test 5: Check files
print("\n[Test 5] Checking required files...")
required_files = [
    'llava_tf/model_loader.py',
    'llava_tf_utils/visual_attacker_tf.py',
    'llava_tf_visual_attack.py',
    'environment_tf.yml'
]

for file_name in required_files:
    if os.path.exists(file_name):
        print(f"✓ {file_name}")
    else:
        print(f"✗ {file_name} not found")

# Test 6: Try importing custom modules
print("\n[Test 6] Testing custom module imports...")
sys.path.insert(0, os.getcwd())

try:
    from llava_tf import get_model, TFLLaVAWrapper
    print("✓ llava_tf module")
except Exception as e:
    print(f"✗ llava_tf module: {e}")

try:
    from llava_tf_utils import TFAttacker
    print("✓ llava_tf_utils module")
except Exception as e:
    print(f"✗ llava_tf_utils module: {e}")

print("\n" + "=" * 60)
print("Environment test complete!")
print("=" * 60)
print("\nIf all tests passed, you can run:")
print("  python llava_tf_visual_attack.py --n_iters 10 --constrained --eps 16 --alpha 1")
