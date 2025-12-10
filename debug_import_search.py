
import sys
import os

src_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, src_path)
print(f"Added {src_path}")

try:
    import formal_nas
    print(f"formal_nas file: {formal_nas.__file__}")
    import formal_nas.search
    print(f"formal_nas.search file: {formal_nas.search.__file__}")
except ImportError as e:
    print(f"Failed: {e}")
