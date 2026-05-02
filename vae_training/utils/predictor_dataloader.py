import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class PredictorDataset(Dataset):
    """
    Dataset for training the predictor model.
    Loads sequences of (latent, action) -> next_latent
    """
    def __init__(self, data_dir, vae_model, sequence_length=11, prediction_step=10,
                 use_noisy=False, noise_level='025', device='cpu'):
        """
        Args:
            data_dir: Directory containing .npz files
            vae_model: Pre-trained VAE model for extracting latents
            sequence_length: Length of input sequences (11 for training)
            prediction_step: How many steps to skip for prediction (10 for training, 1 for testing)
            use_noisy: Whether to use noisy states
            noise_level: Noise level for states
            device: Device to run VAE on
        """
        self.data_dir = Path(data_dir)
        self.vae_model = vae_model
        self.sequence_length = sequence_length
        self.prediction_step = prediction_step
        self.use_noisy = use_noisy
        self.noise_level = noise_level
        self.device = device

        # Get all npz files
        self.files = list(self.data_dir.glob("*.npz"))

        # Pre-extract all latents and prepare sequences
        self.sequences = []
        self.actions = []
        self.targets = []

        print(f"Loading predictor data from {data_dir}...")
        print(f"Sequence length: {sequence_length}, Prediction step: {prediction_step}")

        for file_path in self.files:
            data = np.load(file_path)

            # Get images and extract latents
            imgs = data['imgs'].astype(np.float32) / 255.0  # Normalize to [0,1]
            imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)  # (N, C, H, W)

            # Extract latents using VAE
            with torch.no_grad():
                mu, logvar = vae_model.encode(imgs_tensor)
                latents = mu.cpu().numpy()  # Use mean as latent representation

            # Get actions
            actions = data['acts'].astype(np.float32)

            # Create sequences
            traj_length = len(latents)

            print(f"File {file_path.name}: trajectory length = {traj_length}")

            # Check if trajectory is long enough
            min_required_length = sequence_length + prediction_step
            if traj_length < min_required_length:
                print(f"  Skipping file {file_path.name}: too short ({traj_length} < {min_required_length})")
                continue

            sequences_from_file = 0
            # For each possible starting position
            for start_idx in range(traj_length - sequence_length - prediction_step + 1):
                # Input sequence: latents[start_idx:start_idx+sequence_length]
                input_latents = latents[start_idx:start_idx + sequence_length]
                input_actions = actions[start_idx:start_idx + sequence_length]

                # Target: latent at start_idx + sequence_length + prediction_step - 1
                target_idx = start_idx + sequence_length + prediction_step - 1
                target_latent = latents[target_idx]

                self.sequences.append(input_latents)
                self.actions.append(input_actions)
                self.targets.append(target_latent)
                sequences_from_file += 1

            print(f"  Added {sequences_from_file} sequences from {file_path.name}")

        # Convert to numpy arrays
        if self.sequences:
            self.sequences = np.array(self.sequences)  # (N, seq_len, latent_dim)
            self.actions = np.array(self.actions)      # (N, seq_len, action_dim)
            self.targets = np.array(self.targets)      # (N, latent_dim)
        else:
            print(f"ERROR: No valid sequences found in directory {data_dir}")
            print(f"Required minimum trajectory length: {sequence_length + prediction_step}")
            print(f"Files checked: {[f.name for f in self.files]}")
            raise ValueError(f"No valid sequences found in directory {data_dir}")

        print(f"Created {len(self.sequences)} sequences")
        print(f"Sequence shape: {self.sequences.shape}")
        print(f"Actions shape: {self.actions.shape}")
        print(f"Targets shape: {self.targets.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.from_numpy(self.sequences[idx]).float()
        actions = torch.from_numpy(self.actions[idx]).float()
        target = torch.from_numpy(self.targets[idx]).float()

        return sequence, actions, target

class PredictorTestDataset(Dataset):
    """
    Dataset for testing the predictor model.
    Loads sequences of length 31 for step-by-step prediction
    """
    def __init__(self, data_dir, vae_model, sequence_length=31,
                 use_noisy=False, noise_level='025', device='cpu'):
        """
        Args:
            data_dir: Directory containing .npz files
            vae_model: Pre-trained VAE model for extracting latents
            sequence_length: Length of test sequences (31)
            use_noisy: Whether to use noisy states
            noise_level: Noise level for states
            device: Device to run VAE on
        """
        self.data_dir = Path(data_dir)
        self.vae_model = vae_model
        self.sequence_length = sequence_length
        self.use_noisy = use_noisy
        self.noise_level = noise_level
        self.device = device

        # Get all npz files
        self.files = list(self.data_dir.glob("*.npz"))

        # Pre-extract all latents and prepare test sequences
        self.latent_sequences = []
        self.action_sequences = []
        self.state_sequences = []

        print(f"Loading predictor test data from {data_dir}...")
        print(f"Test sequence length: {sequence_length}")

        for file_path in self.files:
            data = np.load(file_path)

            # Get images and extract latents
            imgs = data['imgs'].astype(np.float32) / 255.0
            imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)

            # Extract latents using VAE
            with torch.no_grad():
                mu, logvar = vae_model.encode(imgs_tensor)
                latents = mu.cpu().numpy()

            # Get actions and states
            actions = data['acts'].astype(np.float32)

            if use_noisy:
                states = data[f'states_noisy_{noise_level}'].astype(np.float32)
            else:
                states = data['states'].astype(np.float32)

            # Create test sequences
            traj_length = len(latents)

            # For each possible starting position that allows a full sequence
            for start_idx in range(traj_length - sequence_length + 1):
                latent_seq = latents[start_idx:start_idx + sequence_length]
                action_seq = actions[start_idx:start_idx + sequence_length]
                state_seq = states[start_idx:start_idx + sequence_length]

                self.latent_sequences.append(latent_seq)
                self.action_sequences.append(action_seq)
                self.state_sequences.append(state_seq)

        # Convert to numpy arrays
        if self.latent_sequences:
            self.latent_sequences = np.array(self.latent_sequences)  # (N, seq_len, latent_dim)
            self.action_sequences = np.array(self.action_sequences)  # (N, seq_len, action_dim)
            self.state_sequences = np.array(self.state_sequences)    # (N, seq_len, state_dim)
        else:
            raise ValueError(f"No valid test sequences found in directory {data_dir}")

        print(f"Created {len(self.latent_sequences)} test sequences")
        print(f"Latent sequences shape: {self.latent_sequences.shape}")
        print(f"Action sequences shape: {self.action_sequences.shape}")
        print(f"State sequences shape: {self.state_sequences.shape}")

    def __len__(self):
        return len(self.latent_sequences)

    def __getitem__(self, idx):
        latent_seq = torch.from_numpy(self.latent_sequences[idx]).float()
        action_seq = torch.from_numpy(self.action_sequences[idx]).float()
        state_seq = torch.from_numpy(self.state_sequences[idx]).float()

        return latent_seq, action_seq, state_seq

def create_predictor_dataloaders(base_dir, fold_num, vae_model, batch_size=32, num_workers=0,
                                use_noisy=False, noise_level='025', device='cpu'):
    """
    Create train and validation dataloaders for predictor training

    Args:
        base_dir: Base directory containing fold_1, fold_2, etc.
        fold_num: Which fold to use as validation (1-5)
        vae_model: Pre-trained VAE model
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        use_noisy: Whether to use noisy states
        noise_level: Noise level for states
        device: Device to run VAE on

    Returns:
        train_loader, val_loader
    """
    base_path = Path(base_dir)

    # Collect all fold directories
    all_folds = [f"fold_{i}" for i in range(1, 6)]
    val_fold = f"fold_{fold_num}"
    train_folds = [f for f in all_folds if f != val_fold]

    print(f"Using {val_fold} for validation")
    print(f"Using {train_folds} for training")

    # Create custom dataset that can handle multiple directories
    class MultiDirPredictorDataset(PredictorDataset):
        def __init__(self, data_dirs, vae_model, sequence_length=11, prediction_step=10,
                     use_noisy=False, noise_level='025', device='cpu'):
            self.vae_model = vae_model
            self.sequence_length = sequence_length
            self.prediction_step = prediction_step
            self.use_noisy = use_noisy
            self.noise_level = noise_level
            self.device = device

            # Collect all files from multiple directories
            self.files = []
            for data_dir in data_dirs:
                data_path = Path(data_dir)
                if data_path.exists():
                    files_in_dir = list(data_path.glob("*.npz"))
                    print(f"Found {len(files_in_dir)} files in {data_path}")
                    self.files.extend(files_in_dir)
                else:
                    print(f"Directory not found: {data_path}")

            # Pre-extract all latents and prepare sequences
            self.sequences = []
            self.actions = []
            self.targets = []

            print(f"Loading predictor training data from {len(self.files)} files...")
            print(f"Looking for sequence_length={sequence_length}, prediction_step={prediction_step}")

            for file_path in self.files:
                data = np.load(file_path)

                imgs = data['imgs'].astype(np.float32) / 255.0
                imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)

                with torch.no_grad():
                    mu, logvar = vae_model.encode(imgs_tensor)
                    latents = mu.cpu().numpy()

                actions = data['acts'].astype(np.float32)
                traj_length = len(latents)

                print(f"File {file_path.name}: trajectory length = {traj_length}")

                # Check if trajectory is long enough
                min_required_length = sequence_length + prediction_step
                if traj_length < min_required_length:
                    print(f"  Skipping file {file_path.name}: too short ({traj_length} < {min_required_length})")
                    continue

                sequences_from_file = 0
                for start_idx in range(traj_length - sequence_length - prediction_step + 1):
                    input_latents = latents[start_idx:start_idx + sequence_length]
                    input_actions = actions[start_idx:start_idx + sequence_length]
                    target_idx = start_idx + sequence_length + prediction_step - 1
                    target_latent = latents[target_idx]

                    self.sequences.append(input_latents)
                    self.actions.append(input_actions)
                    self.targets.append(target_latent)
                    sequences_from_file += 1

                print(f"  Added {sequences_from_file} sequences from {file_path.name}")

            if self.sequences:
                self.sequences = np.array(self.sequences)
                self.actions = np.array(self.actions)
                self.targets = np.array(self.targets)
                print(f"Total sequences created: {len(self.sequences)}")
            else:
                print(f"ERROR: No valid training sequences found!")
                print(f"Required minimum trajectory length: {sequence_length + prediction_step}")
                print(f"Files checked: {[f.name for f in self.files]}")
                raise ValueError(f"No valid training sequences found")

            print(f"Training set: {len(self.sequences)} sequences")

    # Create datasets
    train_dirs = [base_path / fold for fold in train_folds]
    train_dataset = MultiDirPredictorDataset(
        train_dirs,
        vae_model,
        sequence_length=11,
        prediction_step=10,
        use_noisy=use_noisy,
        noise_level=noise_level,
        device=device
    )

    val_dataset = PredictorDataset(
        base_path / val_fold,
        vae_model,
        sequence_length=11,
        prediction_step=10,
        use_noisy=use_noisy,
        noise_level=noise_level,
        device=device
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader

def create_predictor_test_dataloader(data_dir, vae_model, batch_size=1, num_workers=0,
                                   use_noisy=False, noise_level='025', device='cpu'):
    """
    Create test dataloader for predictor evaluation

    Args:
        data_dir: Directory containing test data
        vae_model: Pre-trained VAE model
        batch_size: Batch size (usually 1 for testing)
        num_workers: Number of worker processes
        use_noisy: Whether to use noisy states
        noise_level: Noise level for states
        device: Device to run VAE on

    Returns:
        test_loader
    """
    test_dataset = PredictorTestDataset(
        data_dir,
        vae_model,
        sequence_length=31,
        use_noisy=use_noisy,
        noise_level=noise_level,
        device=device
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return test_loader

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class NPZPredictorDataset(Dataset):
    """
    Build predictor sequences directly from a single .npz file by:
      1) resizing frames to the VAE's training size,
      2) encoding frames to latents with the frozen VAE,
      3) emitting (latent_seq, action_seq, target_latent).

    target_latent is the latent at index: start + seq_len - 1 + pred_step
    So valid_starts = N - (seq_len + pred_step) + 1
    """
    def __init__(
        self,
        npz_path: str,
        vae_model,
        device: torch.device,
        seq_len: int = 11,
        pred_step: int = 10,
        resize_to=(120, 160),
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_step = pred_step

        d = np.load(npz_path)
        if "frame" not in d or "action" not in d:
            raise ValueError(f"Expected 'frame' and 'action' arrays in {npz_path}. Keys: {list(d.keys())}")

        frames = d["frame"]      # (N, C, H, W) — you printed this as (N, 3, 224, 224)
        actions = d["action"]    # (N, action_dim)

        # Normalize frames to [0,1]
        X = frames.astype(np.float32)
        if X.max() > 1.5:
            X /= 255.0

        # Encode to latents with VAE (frozen)
        vae_model.eval()
        latents = []
        with torch.no_grad():
            for i in range(len(X)):
                img = torch.from_numpy(X[i]).unsqueeze(0).to(self.device)  # (1, C, H, W)
                # Resize to VAE input (trained on 120x160)
                img = F.interpolate(img, size=resize_to, mode="bilinear", align_corners=False)
                mu, logvar = vae_model.encode(img)
                z = mu.squeeze(0)  # (latent_dim,)
                latents.append(z.cpu())

        self.latents = torch.stack(latents, dim=0)           # (N, latent_dim)
        self.actions = torch.from_numpy(actions).float()      # (N, action_dim)

        if len(self.latents) != len(self.actions):
            raise ValueError(f"Latents length {len(self.latents)} != actions length {len(self.actions)}")

        self.N = len(self.latents)
        # Number of valid starting indices for (seq_len, pred_step)
        self.valid_starts = self.N - (self.seq_len + self.pred_step) + 1
        if self.valid_starts < 1:
            raise ValueError(
                f"Not enough timesteps: N={self.N}, require at least seq_len+pred_step={self.seq_len + self.pred_step}"
            )

    def __len__(self):
        return self.valid_starts

    def __getitem__(self, idx):
        # latent_seq: [idx : idx+seq_len]
        start = idx
        end = idx + self.seq_len
        target_idx = (end - 1) + self.pred_step

        latent_seq = self.latents[start:end]        # (seq_len, latent_dim)
        action_seq = self.actions[start:end]        # (seq_len, action_dim)
        target_latent = self.latents[target_idx]    # (latent_dim,)

        return latent_seq, action_seq, target_latent

# in utils/predictor_dataloader.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NPZRelativePredictorDataset(Dataset):
    """
    Supports BOTH NPZ formats:

    Format A (windowed under 'data'):
      - npz['data'] is either:
          (a) a numpy.ndarray of windows (N, seq_len, ...) often dtype=object
          (b) a dict that contains 'frame' and 'rel_state' etc.

    Format B (top-level wrapped keys):
      - top-level keys include 'frame' (T, C, H, W) and 'rel_state' (N, seq_len, >=3)
      - optional: 'timestamps' (T,)

    Returns:
      latents_ctx   : (seq_len-1, D)   # z_0..z_{seq_len-2}
      actions       : (seq_len-1, 3)   # body-frame deltas for transitions
      target_latent : (D,)             # z_{seq_len-1}
      pose_tgt      : (3,)             # [x_rel, y_rel, yaw_rel] at final step
    """

    def __init__(
        self,
        npz_path,
        vae_model,
        device,
        resize_to=(120, 160),
        action_gain=1.0,
        normalize_actions=True,
    ):
        super().__init__()
        self.device = device
        self.vae = vae_model.eval()  # frozen VAE
        self.resize_to = resize_to
        self.action_gain = float(action_gain)
        self.normalize_actions = bool(normalize_actions)

        raw_npz = np.load(npz_path, allow_pickle=True)

        # -----------------------------
        # Helpers (ROBUST GUARDS)
        # -----------------------------
        def _unwrap_zero_dim_object(x):
            while isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
                x = x.item()
            return x

        def _safe_timestamps(ts_all, T):
            """
            Return a valid timestamps array of length T.
            - If missing, empty, wrong shape, or scalar/object weirdness -> fallback to arange(T).
            """
            if ts_all is None:
                return np.arange(T, dtype=np.int64)

            ts_all = _unwrap_zero_dim_object(ts_all)

            if not isinstance(ts_all, np.ndarray):
                return np.arange(T, dtype=np.int64)

            # zero-length timestamps -> fallback
            if ts_all.ndim < 1 or ts_all.shape[0] == 0:
                return np.arange(T, dtype=np.int64)

            # wrong length -> fallback
            if ts_all.shape[0] != T:
                return np.arange(T, dtype=np.int64)

            return ts_all

        def _build_windows_from_frame_relstate(base_dict):
            if "frame" not in base_dict or "rel_state" not in base_dict:
                raise ValueError(
                    f"Expected 'frame' and 'rel_state'. Keys: {list(base_dict.keys())}"
                )

            frames_all = _unwrap_zero_dim_object(base_dict["frame"])
            rel_state  = _unwrap_zero_dim_object(base_dict["rel_state"])
            ts_all_raw = base_dict.get("timestamps", None)

            # --- HARD GUARDS ---
            if not isinstance(frames_all, np.ndarray) or frames_all.ndim < 1:
                raise RuntimeError(
                    f"'frame' is not a valid ndarray. type={type(frames_all)} "
                    f"shape={getattr(frames_all,'shape',None)}"
                )
            if frames_all.shape[0] == 0:
                raise RuntimeError(
                    f"'frame' has 0 length on axis 0. shape={frames_all.shape}. "
                    f"Your NPZ frames are empty / wrong file."
                )
            if not isinstance(rel_state, np.ndarray) or rel_state.ndim != 3 or rel_state.shape[2] < 3:
                raise RuntimeError(
                    f"'rel_state' must be (N, seq_len, >=3). got {getattr(rel_state,'shape',None)}"
                )

            N_rel, seq_len, _ = rel_state.shape
            T = frames_all.shape[0]

            # IMPORTANT: robust timestamps handling (fixes your exact crash)
            ts_all = _safe_timestamps(ts_all_raw, T)

            # This dataset assumes frames are packed consecutively per window:
            # window i uses frames [i*seq_len : i*seq_len + seq_len]
            max_by_frames = T // seq_len
            N_eff = min(N_rel, max_by_frames)

            if N_eff <= 0:
                raise RuntimeError(
                    f"Cannot form any windows: T={T}, seq_len={seq_len}, "
                    f"N_rel={N_rel}, max_by_frames={max_by_frames}."
                )

            windows = []
            for i in range(N_eff):
                base_idx = i * seq_len
                f_slice  = frames_all[base_idx: base_idx + seq_len]
                ts_slice = ts_all[base_idx: base_idx + seq_len]
                p_slice  = rel_state[i, :, :3].astype(np.float32)  # (seq_len,3)

                # extra guards (avoid silent empty slices)
                if f_slice.shape[0] != seq_len:
                    raise RuntimeError(
                        f"Bad frame slice: i={i}, base_idx={base_idx}, expected {seq_len} frames "
                        f"but got {f_slice.shape[0]}. T={T}, frames_all.shape={frames_all.shape}"
                    )
                if ts_slice.shape[0] != seq_len:
                    # should not happen due to _safe_timestamps, but guard anyway
                    raise RuntimeError(
                        f"Bad timestamp slice: i={i}, base_idx={base_idx}, expected {seq_len} "
                        f"but got {ts_slice.shape[0]}. T={T}, ts_all.shape={ts_all.shape}"
                    )
                if p_slice.shape[0] != seq_len:
                    raise RuntimeError(
                        f"Bad pose slice: i={i}, expected {seq_len} poses but got {p_slice.shape[0]}. "
                        f"rel_state.shape={rel_state.shape}"
                    )

                win = [(f_slice[t], ts_slice[t], p_slice[t]) for t in range(seq_len)]
                windows.append(win)

            return np.array(windows, dtype=object), seq_len

        # -----------------------------
        # Decide which format we have
        # -----------------------------
        self.meta = {k: raw_npz[k] for k in raw_npz.files}  # keep for debugging

        if "data" in raw_npz.files:
            base = _unwrap_zero_dim_object(raw_npz["data"])

            # Format A(a): already windows ndarray
            if isinstance(base, np.ndarray):
                if base.ndim < 2:
                    raise RuntimeError(
                        f"'data' ndarray must be at least 2D (N, seq_len, ...). Got {base.shape}"
                    )
                self.data = base
                self.N = base.shape[0]
                self.seq_len = base.shape[1]

            # Format A(b): dict wrapper -> build windows from frame/rel_state
            elif isinstance(base, dict):
                self.meta = base
                self.data, self.seq_len = _build_windows_from_frame_relstate(base)
                self.N = self.data.shape[0]

            else:
                raise RuntimeError(
                    f"Unsupported type for npz['data'] after unwrapping: {type(base)}"
                )

        else:
            # Format B: top-level keys
            base_dict = {k: raw_npz[k] for k in raw_npz.files}
            self.data, self.seq_len = _build_windows_from_frame_relstate(base_dict)
            self.N = self.data.shape[0]
            self.meta = base_dict

        # -----------------------------
        # Precompute action mean/std
        # -----------------------------
        if self.normalize_actions:
            acts = []
            for i in range(self.N):
                poses = np.array(
                    [self.data[i][t][2] for t in range(self.seq_len)],
                    dtype=np.float32,
                )  # (seq_len,3)
                actions = self._build_body_actions(poses)  # (seq_len-1,3)
                acts.append(actions)

            A = np.concatenate(acts, axis=0)  # ((seq_len-1)*N, 3)
            self.action_mean = A.mean(axis=0, keepdims=True)
            self.action_std  = A.std(axis=0, keepdims=True) + 1e-8
        else:
            self.action_mean = None
            self.action_std = None

    @torch.no_grad()
    def _encode_frame(self, frame_np):
        """
        Encode a single frame via the frozen VAE into latent vector z.

        Accepts frames in:
          - (H,W) or (H,W,C) or (C,H,W)
          - uint8 (0..255) or float (0..1)
        """
        x = frame_np.astype(np.float32)

        if x.ndim == 2:
            x = x[..., None]  # (H,W,1)

        if x.max() > 1.5:
            x /= 255.0

        if x.ndim == 3:
            # (C,H,W) or (H,W,C)
            if x.shape[0] in (1, 3) and x.shape[1] > 8 and x.shape[2] > 8:
                chw = x
            else:
                chw = np.transpose(x, (2, 0, 1))
        else:
            raise RuntimeError(f"Unexpected frame shape: {x.shape}")

        t = torch.from_numpy(chw).unsqueeze(0)  # (1,C,H,W)
        t = F.interpolate(t, size=self.resize_to, mode="bilinear", align_corners=False)

        mu, logvar = self.vae.encode(t.to(self.device))
        return mu.squeeze(0)  # (D,)

    def _build_body_actions(self, poses):
        """
        poses: (seq_len,3) [x_rel, y_rel, yaw_rel]
        Returns actions: (seq_len-1,3) [dx_body, dy_body, d_yaw]
        """
        P = poses.astype(np.float32).copy()

        yaw = P[:, 2]
        yaw_u = np.unwrap(yaw)
        P[:, 2] = yaw_u

        dP = P[1:] - P[:-1]  # (seq_len-1,3)

        yaw_prev = P[:-1, 2]
        c, s = np.cos(-yaw_prev), np.sin(-yaw_prev)

        dx_w, dy_w = dP[:, 0], dP[:, 1]
        dx_b = c * dx_w - s * dy_w
        dy_b = s * dx_w + c * dy_w

        actions = np.stack([dx_b, dy_b, dP[:, 2]], axis=1)  # (seq_len-1,3)

        if self.action_gain != 1.0:
            actions[:, 0:2] *= self.action_gain
            actions[:, 2]   *= self.action_gain

        return actions

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        Returns:
            latents_ctx   : (seq_len-1, D)
            actions       : (seq_len-1, 3)
            target_latent : (D,)
            pose_tgt      : (3,)
        """
        batch = self.data[idx]  # list/array length seq_len

        frames = [batch[t][0] for t in range(self.seq_len)]
        poses  = np.array([batch[t][2] for t in range(self.seq_len)], dtype=np.float32)

        latents = torch.stack([self._encode_frame(fr) for fr in frames], dim=0)  # (seq_len,D)

        actions_np = self._build_body_actions(poses)  # (seq_len-1,3)
        if self.normalize_actions:
            actions_np = (actions_np - self.action_mean) / self.action_std
        actions = torch.from_numpy(actions_np).float()

        latents_ctx   = latents[:-1]
        target_latent = latents[-1]
        pose_tgt      = torch.from_numpy(poses[-1]).float()

        return latents_ctx, actions, target_latent, pose_tgt
