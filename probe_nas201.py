
import torch
import sys

path = "NAS-Bench-201-v1_1-096897.pth"
print(f"Loading {path}...")
try:
    data = torch.load(path, map_location='cpu')
    print("Keys in root:", data.keys())
    
    if 'meta_archs' in data:
        print(f"Num Meta Archs: {len(data['meta_archs'])}")
        print("Sample Arch:", data['meta_archs'][0])
        
    # Check data for index 0
    if '0' in data:
        print("Entry '0' keys:", data['0'].keys())
        # Check cifar10-valid
        if 'cifar10-valid' in data['0']:
            print("cifar10-valid data keys:", data['0']['cifar10-valid'].keys())
            
except Exception as e:
    print(f"Error: {e}")
