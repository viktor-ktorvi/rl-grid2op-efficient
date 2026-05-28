#!/usr/bin/env python3

"""
Load all saved checkpoints and reconstruct:

    - Environment
    - Observation

for each checkpoint.

This fully restores the simulator state by:
    1. Recreating the environment
    2. Loading the same chronic
    3. Resetting with the same seed
    4. Replaying all saved actions

The resulting observation is exactly the saved state.
"""

import glob
import os
import pickle
from typing import Tuple

import grid2op
import numpy as np
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend

# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_DIR = "rho_checkpoints"


# ============================================================
# RESTORE FUNCTION
# ============================================================


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[Environment, BaseObservation, dict]:
    """
    Restore an environment and observation
    from a saved checkpoint.
    """

    # --------------------------------------------------------
    # Load checkpoint file
    # --------------------------------------------------------

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # --------------------------------------------------------
    # Recreate environment
    # --------------------------------------------------------

    env = grid2op.make(checkpoint["env_name"], backend=LightSimBackend())

    # --------------------------------------------------------
    # Load same chronic
    # --------------------------------------------------------

    env.set_id(checkpoint["chronics_id"])

    # --------------------------------------------------------
    # Reset with same seed
    # --------------------------------------------------------

    obs = env.reset(seed=checkpoint["seed"])

    # --------------------------------------------------------
    # Replay actions
    # --------------------------------------------------------

    for i, act in enumerate(checkpoint["actions"], start=1):

        obs, reward, done, info = env.step(act)

        if done:
            raise RuntimeError(
                f"Replay failed for checkpoint:\n"
                f"{checkpoint_path}\n"
                f"Environment terminated during replay "
                f"at step {i}"
            )

    return env, obs, checkpoint


# ============================================================
# FIND CHECKPOINT FILES
# ============================================================

checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pkl")))

print(f"\nFound {len(checkpoint_files)} checkpoint files.\n")


# ============================================================
# LOAD ALL CHECKPOINTS
# ============================================================

for checkpoint_path in checkpoint_files:

    print("=" * 80)
    print(f"LOADING: {checkpoint_path}")
    print("=" * 80)

    env, obs, checkpoint = load_checkpoint(checkpoint_path)

    max_rho = float(np.max(obs.rho))

    print("Successfully restored state:")
    print(f"  chronic_id : {checkpoint['chronics_id']}")
    print(f"  nb_steps   : {checkpoint['nb_steps']}")
    print(f"  max_rho    : {max_rho:.4f}")

    print()
    env.close()
