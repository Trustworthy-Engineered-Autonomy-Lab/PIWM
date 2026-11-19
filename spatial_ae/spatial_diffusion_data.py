"""Strict trajectory parsing for spatial-AE future-frame diffusion.

NumPy-only helpers. Pairs never cross an episode or terminal boundary.
lander_raw columns 0,1,4 and the original PIWM raw-to-normalized conversion
are explicit defaults, not inferred from an unrelated flat image archive.
"""
from pathlib import Path
import hashlib
import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def terminal_flags(flags, leading, path, key):
    """Normalize step flags or an episode-end summary to the frame shape.

    A scalar/single-value flag describes how the whole archive ended. For an
    E,T archive, E values describe the ends of its E independent episodes.
    Such summaries mark only the final frame, never every transition. They
    provide no information about interior resets; use step flags for those.
    """
    flags = np.asarray(flags)
    original_shape = flags.shape
    if flags.dtype.kind not in 'biuf' or not np.isfinite(flags).all() or np.any((flags != 0) & (flags != 1)):
        raise ValueError(f'{path}: {key} must contain boolean or numeric 0/1 flags; '
                         f'got shape={original_shape}, dtype={flags.dtype}')
    length = leading[-1]
    transition_shape = (*leading[:-1], length - 1)
    if flags.shape in ((*leading, 1), (*transition_shape, 1)):
        flags = flags[..., 0]
    # Single episodes sometimes retain an explicit episode axis.
    if len(leading) == 1 and flags.shape in ((1, length), (1, length - 1)):
        flags = flags[0]
    if flags.shape == leading:
        return flags.astype(bool)
    result = np.zeros(leading, dtype=bool)
    if flags.shape == transition_shape:
        result[..., :-1] = flags.astype(bool)
        return result
    if flags.size == 1 and flags.ndim <= 2:
        result[..., -1] = bool(flags.reshape(-1)[0])
        return result
    if len(leading) == 2 and flags.shape in ((leading[0],), (leading[0], 1)):
        result[..., -1] = flags.reshape(leading[0]).astype(bool)
        return result
    raise ValueError(f'{path}: {key} does not align with frames: got shape={original_shape}; '
                     f'expected frame flags {leading}, transition flags {transition_shape} '
                     '(optionally with a trailing singleton axis), a scalar/single-value '
                     'episode-end flag, or one end flag per episode')


def load_episodes(path):
    """Return independent episodes as byte NHWC frames, raw states, actions, done."""
    with np.load(path, allow_pickle=False) as archive:
        frame_key = next((k for k in ('image', 'frame', 'images', 'frames') if k in archive), None)
        if frame_key is None or 'lander_raw' not in archive or 'action' not in archive:
            raise ValueError(f'{path}: need image/frame, lander_raw and action; got {archive.files}')
        images, states, actions = archive[frame_key], archive['lander_raw'], archive['action']
        if images.shape[-3:] == (3, 96, 128):
            images = np.moveaxis(images, -3, -1)
        if images.shape[-3:] != (96, 128, 3) or images.ndim not in (4, 5):
            raise ValueError(f'{path}: expected T,H,W,3 or episodes,T,H,W,3 at 96x128; got {images.shape}')
        leading = images.shape[:-3]
        if any(size < 1 for size in leading):
            raise ValueError(f'{path}: empty episode/frame axis: {leading}')
        if states.shape[:-1] != leading or states.shape[-1] < 5:
            raise ValueError(f'{path}: images/states do not align: {images.shape} / {states.shape}')
        if not np.isfinite(states).all():
            raise ValueError(f'{path}: nonfinite raw state')
        if images.dtype != np.uint8:
            if not np.issubdtype(images.dtype, np.floating) or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError(f'{path}: images must be uint8 or floating [0,1]')
            images = np.rint(images * 255).astype(np.uint8)
        episodes, length = (1, leading[0]) if len(leading) == 1 else leading
        if actions.shape == (*leading, 1):
            actions = actions[..., 0]
        if len(leading) == 1 and actions.ndim == 2 and actions.shape[-1] == 1:
            actions = actions[:, 0]
        if len(leading) == 2 and actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions[..., 0]
        valid_action_shapes = ((length,), (length - 1,)) if len(leading) == 1 else ((episodes, length), (episodes, length - 1))
        if actions.shape not in valid_action_shapes:
            raise ValueError(f'{path}: need scalar discrete action per transition; got {actions.shape}')
        if not np.isfinite(actions).all() or np.any(actions != np.rint(actions)) or np.any((actions < 0) | (actions > 3)):
            raise ValueError(f'{path}: actions must be integers 0..3')
        done = np.zeros(leading, dtype=bool)
        for key in ('done', 'terminated', 'truncated'):
            if key in archive:
                done |= terminal_flags(archive[key], leading, path, key)
    images = images.reshape(episodes, length, 96, 128, 3)
    states = states.reshape(episodes, length, states.shape[-1]).astype(np.float32)
    actions = actions.reshape(episodes, -1).astype(np.int64)
    done = done.reshape(episodes, length)
    return [(images[k], states[k], actions[k], done[k]) for k in range(episodes)]


def pair_indices(length, actions, done, horizon):
    if horizon < 1:
        raise ValueError('horizon must be positive')
    return [i for i in range(max(0, length - horizon))
            if i + horizon <= len(actions) and not np.any(done[i:i+horizon])]


def physical_pose(raw, cols=(0, 1, 4), scales=(.1, .15), offsets=(-1., -.59)):
    if max(cols) >= raw.shape[-1] or min(cols) < 0:
        raise ValueError('Pose columns outside raw state')
    p = np.asarray(raw[..., list(cols)], dtype=np.float32).copy()
    p[..., 0] = p[..., 0] * scales[0] + offsets[0]
    p[..., 1] = p[..., 1] * scales[1] + offsets[1]
    p[..., 2] = np.arctan2(np.sin(p[..., 2]), np.cos(p[..., 2]))
    return p


def split_archives(paths, fraction, seed):
    if len(paths) < 3 or not 0 < fraction < .5:
        raise ValueError('Need at least 3 training trajectories and validation fraction between 0 and .5')
    order = np.random.default_rng(seed).permutation(len(paths))
    nval = max(1, int(round(len(paths) * fraction)))
    return [paths[i] for i in order[nval:]], [paths[i] for i in order[:nval]]


def self_test():
    import tempfile
    actions = np.zeros(7, dtype=np.int64)
    done = np.zeros(8, dtype=bool)
    assert pair_indices(8, actions, done, 2) == list(range(6))
    done[3] = True
    assert pair_indices(8, actions, done, 2) == [0, 1, 4, 5]
    assert pair_indices(2, np.zeros(1), np.zeros(2, dtype=bool), 2) == []
    pose = physical_pose(np.array([[10., 4., 0., 0., .2]], dtype=np.float32))
    assert np.allclose(pose, [[0., .01, .2]], atol=1e-6)
    paths = [Path(str(i)) for i in range(20)]
    train, val = split_archives(paths, .1, 42)
    assert len(train) == 18 and len(val) == 2 and not set(train) & set(val)
    # Binary fixture exercises the real archive loader and multi-episode boundaries.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'multi.npz'
        frames = np.zeros((2, 5, 96, 128, 3), dtype=np.uint8)
        frames[1] = 120
        raw = np.zeros((2, 5, 8), dtype=np.float32)
        act = np.zeros((2, 4, 1), dtype=np.int64)
        flags = np.zeros((2, 4), dtype=bool); flags[0, 1] = True
        np.savez(path, image=frames, lander_raw=raw, action=act, terminated=flags)
        episodes = load_episodes(path)
        assert len(episodes) == 2
        assert pair_indices(5, episodes[0][2], episodes[0][3], 2) == [2]
        assert pair_indices(5, episodes[1][2], episodes[1][3], 2) == [0, 1, 2]
        assert episodes[1][0][0, 0, 0, 0] == 120
        chw = np.moveaxis(frames[0], -1, -3)
        np.savez(path, image=chw, lander_raw=raw[0], action=act[0])
        assert load_episodes(path)[0][0].shape == (5, 96, 128, 3)
        # End-of-episode metadata must not be broadcast onto every step.
        for summary in (np.array(True), np.array([True]), np.array([[True]])):
            np.savez(path, image=frames[0], lander_raw=raw[0], action=act[0],
                     terminated=summary, truncated=np.array(False))
            episode = load_episodes(path)[0]
            assert episode[3].tolist() == [False, False, False, False, True]
            assert pair_indices(5, episode[2], episode[3], 2) == [0, 1, 2]
        for summary in (np.array([True, False]), np.array([[True], [False]])):
            np.savez(path, image=frames, lander_raw=raw, action=act, terminated=summary)
            episodes = load_episodes(path)
            assert episodes[0][3].tolist() == [False, False, False, False, True]
            assert not episodes[1][3].any()
            assert all(pair_indices(5, e[2], e[3], 2) == [0, 1, 2] for e in episodes)
        # Per-transition columns retain their interior boundary semantics.
        step_flags = np.array([[False], [True], [False], [False]])
        np.savez(path, image=frames[0], lander_raw=raw[0], action=act[0], terminated=step_flags)
        episode = load_episodes(path)[0]
        assert pair_indices(5, episode[2], episode[3], 2) == [2]
        for invalid in (np.zeros(3, dtype=bool), np.array(np.nan), np.array(2), np.array('True')):
            np.savez(path, image=frames[0], lander_raw=raw[0], action=act[0], terminated=invalid)
            try:
                load_episodes(path)
            except ValueError as error:
                assert 'terminated' in str(error) and 'shape=' in str(error)
            else:
                raise AssertionError('Invalid terminal metadata accepted')
        # A label/image count mismatch must fail, not shift every future target.
        np.savez(path, image=frames[0], lander_raw=raw[0, :-1], action=act[0])
        try:
            load_episodes(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Misaligned archive accepted')
    print('DATA SELF-TEST PASSED: archive parsing, scalar/episode-end metadata, interior terminal boundaries, action horizon, pose units, trajectory split, malformed metadata and misalignment rejection.')


if __name__ == '__main__':
    self_test()
