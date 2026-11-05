import os
import glob

def generate_list_file(root_dir, output_file):
    """
    Generate a list file containing absolute paths of all .npy feature files
    within the given root directory (recursively).
    """
    # Recursively find all .npy files
    npy_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True))
    
    if not npy_files:
        print(f"[WARN] No .npy files found in {root_dir}")
        return
    
    # Write to list file
    with open(output_file, "w") as f:
        for path in npy_files:
            f.write(path + "\n")

    print(f"[INFO] Saved {len(npy_files)} entries to {output_file}")

def main():
    # Paths
    train_root = "/mnt/c/code/data/I3D/Train/RGB"
    test_root  = "/mnt/c/code/data/I3D/Test/RGB"

    # Output list files
    train_list_file = "/mnt/c/code/ae_anomaly/UR-DMU/list/UCF_Train.list"
    test_list_file  = "/mnt/c/code/ae_anomaly/UR-DMU/list/UCF_Test.list"

    # Generate lists
    print("[INFO] Generating train list...")
    generate_list_file(train_root, train_list_file)

    print("[INFO] Generating test list...")
    generate_list_file(test_root, test_list_file)

    print("[DONE] List files created successfully.")

if __name__ == "__main__":
    main()
