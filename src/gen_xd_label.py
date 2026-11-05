import os
import glob

# --- Configuration ---
train_dir = r'/mnt/c/code/ae_anomaly/UR-DMU/xdv/i3d-features/RGB'
test_dir  = r'/mnt/c/code/ae_anomaly/UR-DMU/xdv/i3d-features/RGBTest'
out_train_list = 'list/XD_Train.list'
out_test_list  = 'list/XD_Test.list'
# ---------------------

def write_list(feature_dir, out_path):
    # Find all .npy files in the directory
    files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
    # Extract base names only
    names = [os.path.basename(f) for f in files]
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for name in names:
            out_file_path = os.path.join(feature_dir, name)
            f.write(out_file_path + '\n')
    print(f'[INFO] {len(names)} files written to {out_path}')

if __name__ == '__main__':
    write_list(train_dir, out_train_list)
    write_list(test_dir,  out_test_list)