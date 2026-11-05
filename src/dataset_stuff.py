class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        if seed >= 0 and hasattr(utils, "set_seed"):
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.root_dir = root_dir

        list_path_repo = os.path.join('list', f'UCF_{self.mode}.list')
        alt_list_path = os.path.join(self.root_dir, f'{self.mode.lower()}_list.txt')

        if os.path.exists(list_path_repo):
            list_file = list_path_repo
        elif os.path.exists(alt_list_path):
            list_file = alt_list_path
        else:
            raise FileNotFoundError(f"Cannot find list file for mode={self.mode}. Looked for {list_path_repo} and {alt_list_path}.")

        samples = []
        if self.mode == 'Train':
            numAbnorm = 8100
        else:
            numAbnorm = 1400
        with open(list_file, 'r') as f:
            for (i,line) in enumerate(f):
                if (i < numAbnorm):
                    samples.append((line.strip(),1))
                else:
                    samples.append((line.strip(),0))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label
      

    def get_data(self, index):
        vid, label = self.samples[index]
        #name = os.path.basename(base)
        video_feature = np.load(vid).astype(np.float32)   # [T, feat_dim]

        if self.mode == "Train":
            T = video_feature.shape[0]
            D = video_feature.shape[1]
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i+1], :], axis=0)
                else:
                    new_feat[i, :] = video_feature[r[i], :]
            video_feature = new_feat
        return video_feature, label


def taml_meta_loss(query_losses, lambda_var=0.1, lambda_ent=0.05):
    losses = torch.stack(query_losses)

    mean_loss = losses.mean()
    var_loss = losses.var()

    weights = torch.softmax(losses.detach(), dim=0)
    entropy = -torch.sum(weights * torch.log(weights + 1e-8))

    return mean_loss + lambda_var * var_loss - lambda_ent * entropy

# -------------------------
# Helper: random sampling by video (k videos)
# -------------------------

def check_tensor(name, x):
    if x is None:
        print(f"[NAN CHECK] {name}: None")
        return True
    if not torch.is_tensor(x):
        return False
    if torch.isnan(x).any():
        print(f"[NAN CHECK] {name}: NaN detected")
        return True
    if torch.isinf(x).any():
        print(f"[NAN CHECK] {name}: Inf detected")
        return True
    return False

import os
import pickle

def build_index_cache(dataset, cache_path="index_cache.pkl", rebuild=False):
    """
    Builds or loads cached indices for normal/abnormal samples.
    """

    # -------------------------
    # LOAD CACHE
    # -------------------------
    if os.path.exists(cache_path) and not rebuild:
        print(f"[INFO] Loading index cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["normal"], cache["abnormal"]

    # -------------------------
    # BUILD CACHE
    # -------------------------
    print("[INFO] Building index cache (one-time cost)...")

    normal_idxs = []
    abnormal_idxs = []

    for i in range(len(dataset)):
        label = dataset.samples[i][1]   # 🔴 FAST: avoids loading npy file
        if label == 0:
            normal_idxs.append(i)
        else:
            abnormal_idxs.append(i)

    # -------------------------
    # SAVE CACHE
    # -------------------------
    cache = {
        "normal": normal_idxs,
        "abnormal": abnormal_idxs
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"[INFO] Saved index cache to {cache_path}")

    return normal_idxs, abnormal_idxs

def get_random_batch_from_dataset_fast(dataset, normal_idxs, abnormal_idxs, k, device):
    half_k = k // 2

    idxs = random.sample(normal_idxs, half_k) + \
           random.sample(abnormal_idxs, k - half_k)
    #random.shuffle(idxs)

    feats, labels = [], []

    for i in idxs:
        feat, label = dataset[i]

        arr = np.asarray(feat).astype(np.float32)

        feats.append(torch.from_numpy(arr))
        labels.append(torch.tensor(label).float())

    x = torch.stack(feats).to(device)
    y = torch.stack(labels).to(device)

    return x, y
def get_random_batch_from_dataset(dataset, k, device, do_norm=False):
    """
    Sample k videos from dataset (dataset.__getitem__ -> (T,D,label)).
    Returns x: (k, T, D) tensor, y: (k,) tensor
    """
    idxs = random.sample(range(len(dataset)), k)
    feats, labels = [], []
    for i in idxs:
        feat, label = dataset[i]
        arr = np.asarray(feat).astype(np.float32)
        if do_norm:
            mu = arr.mean(axis=0, keepdims=True)
            s = arr.std(axis=0, keepdims=True) + 1e-6
            arr = (arr - mu) / s
        feats.append(torch.from_numpy(arr))
        labels.append(torch.tensor(label).float())
    x = torch.stack(feats).to(device)   # (k, T, D)
    y = torch.stack(labels).to(device)  # (k,)
    return x, y
