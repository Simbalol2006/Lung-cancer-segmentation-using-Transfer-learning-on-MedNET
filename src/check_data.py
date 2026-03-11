import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Point this to your actual data folder
DATA_DIR = Path("/home/bharadhwaj/LIDC_PROJECT/processed_data/images")

def check_files():
    files = list(DATA_DIR.glob("*.npy"))
    print(f"Checking {len(files)} files...")
    
    bad_files = []
    
    for f in tqdm(files):
        try:
            # We try to load just the metadata first, then the data
            np.load(f, mmap_mode='r')
        except ValueError as e:
            print(f"\n[!] CORRUPTED: {f.name}")
            print(f"    Error: {e}")
            bad_files.append(f.name)
        except Exception as e:
            print(f"\n[!] ERROR: {f.name} - {e}")
            bad_files.append(f.name)
            
    print("\n--- SUMMARY ---")
    if bad_files:
        print(f"Found {len(bad_files)} bad files:")
        for bf in bad_files:
            print(bf)
    else:
        print("All files look healthy!")

if __name__ == "__main__":
    check_files()
