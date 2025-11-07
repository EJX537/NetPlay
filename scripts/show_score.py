#!/usr/bin/env python3
"""Show final score (and a few BLStats) from a trajectories.h5py file.
Usage: python scripts/show_score.py <path/to/trajectories.h5py>
"""
import sys
import h5py
import argparse

# CLI
parser = argparse.ArgumentParser(description="Show final score (and a few BLStats) from a trajectories.h5py file.")
parser.add_argument('path', help='path to trajectories.h5py')
parser.add_argument('--ascii-map', action='store_true', dest='ascii_map', help='Print the final tty_chars ASCII map to stdout')
parser.add_argument('--save-map', dest='save_map', help='Save the final tty_chars ASCII map to the given file path')
args = parser.parse_args()
path = args.path
try:
    # Prefer the library's canonical BLStats helper which knows the correct ordering
    from netplay.nethack_agent.tracking import make_blstats, BLStats
except Exception:
    from collections import namedtuple
    # Fallback BLStats definition (kept in sync with tracking.BLStats ordering)
    BLStats = namedtuple('BLStats', 'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask alignment')
    def make_blstats(seq):
        s = list(seq)
        # Pad or trim to expected length; best-effort fallback
        expected = len(BLStats._fields)
        if len(s) == expected:
            return BLStats(*s)
        if len(s) == expected - 1:
            return BLStats(*s, 0)
        # Pad with zeros
        s = (s + [0] * expected)[:expected]
        return BLStats(*s)

try:
    f = h5py.File(path, 'r')
except Exception as e:
    print(f"Failed to open '{path}': {e}")
    sys.exit(1)

if 'trajectories' not in f:
    print("No 'trajectories' group found in file.")
    f.close()
    sys.exit(1)

trajs = f['trajectories']
ids = sorted([int(k) for k in trajs.keys()])
if not ids:
    print('No trajectories found in file')
    f.close()
    sys.exit(1)

tid = str(ids[-1])
print(f"Using trajectory {tid}")
obs_grp = trajs[tid]['observations']
if 'blstats' not in obs_grp:
    print("No 'blstats' dataset found in observations")
    f.close()
    sys.exit(1)

bl = obs_grp['blstats'][:]
if bl.shape[0] == 0:
    print('blstats dataset is empty')
    f.close()
    sys.exit(1)
# Sometimes the monitor writes a trailing all-zero row at the end (e.g. after reset).
# Pick the last non-empty row instead.
import numpy as _np
nonzero_rows = _np.where(_np.any(bl != 0, axis=1))[0]
if len(nonzero_rows) > 0:
    last_idx = int(nonzero_rows[-1])
else:
    last_idx = bl.shape[0] - 1
last_bl = bl[last_idx]
print(f'Using blstats row index: {last_idx} (of {bl.shape[0]})')

# Optionally print or save the ASCII tty map corresponding to the chosen blstats row
if (getattr(args, 'ascii_map', False) or getattr(args, 'save_map', None)):
    if 'tty_chars' not in obs_grp:
        print('No tty_chars dataset available to render ASCII map')
    else:
        tty = obs_grp['tty_chars'][:]
        # Choose matching tty index if available, otherwise pick last non-empty
        try:
            if tty.shape[0] > last_idx:
                tty_last = tty[last_idx]
            else:
                nonzero = _np.where(_np.any(tty != 0, axis=1))[0]
                tty_last = tty[nonzero[-1]] if len(nonzero) else tty[-1]
        except Exception:
            # Fallback: pick final row
            tty_last = tty[-1]

        # tty_last may be a 2D array of char codes (rows, cols) or an array of bytes per row
        ascii_rows = []
        try:
            # If rows are sequences of bytes, decode each
            for row in tty_last:
                if isinstance(row, (bytes, bytearray)):
                    ascii_rows.append(row.decode('latin1').rstrip('\x00\r\n'))
                else:
                    # row may be an array of ints
                    ascii_rows.append(''.join(chr(int(c)) for c in row).rstrip('\x00\r\n'))
        except Exception:
            # Last resort: try to coerce to bytes
            try:
                ascii_rows = [bytes(tty_last).decode('latin1').splitlines()]
            except Exception:
                ascii_rows = ['<failed to decode tty_chars>']

        if getattr(args, 'ascii_map', False):
            print('\n'.join(ascii_rows))
        if getattr(args, 'save_map', None):
            try:
                with open(args.save_map, 'w', encoding='utf-8') as wf:
                    wf.write('\n'.join(ascii_rows))
                print(f'Saved ASCII map to {args.save_map}')
            except Exception as e:
                print('Failed to save ASCII map:', e)
try:
    bl_named = make_blstats(last_bl)
except Exception as e:
    print('Failed to construct BLStats from sequence:', e)
    bl_named = None

if bl_named is not None:
    # Print common fields with sanity checks
    # Score
    score = getattr(bl_named, 'score', None)
    if score is None:
        # Some older formats might place score elsewhere; try heuristic lookup
        try:
            score = int(last_bl[9])
        except Exception:
            score = 'unknown'
    print('Final score:', score)

    # Hitpoints
    hp = getattr(bl_named, 'hitpoints', None)
    max_hp = getattr(bl_named, 'max_hitpoints', None)
    if hp is None or max_hp is None:
        # try common alternate indices
        try:
            hp = int(last_bl[10])
        except Exception:
            hp = None
        try:
            max_hp = int(last_bl[11])
        except Exception:
            max_hp = None
    if hp is not None and max_hp is not None:
        print(f'Final HP: {hp}/{max_hp}')
    else:
        print('Final HP: unknown')

    # Depth / level_number
    depth = getattr(bl_named, 'depth', None)
    if depth is None:
        try:
            depth = int(last_bl[12])
        except Exception:
            depth = 'unknown'
    print('Final depth:', depth)

    # For debugging, if any values look zero/invalid, print raw tail of blstats
    try:
        if isinstance(score, int) and score == 0:
            tail = list(last_bl[-6:])
            print('blstats tail (for debugging):', tail)
    except Exception:
        pass

# Print sum of rewards if available
if 'rewards' in trajs[tid]:
    try:
        rewards = trajs[tid]['rewards'][:]
        print('Sum of rewards:', float(rewards.sum()))
    except Exception:
        pass

f.close()
