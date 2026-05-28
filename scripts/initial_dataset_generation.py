#!/usr/bin/env python3

"""
For each available chronic:

1. Load the chronic
2. Run the DoNothing agent
3. Stop when max(rho) > 0.9
4. Save a checkpoint
5. Move to the next chronic

Each checkpoint contains enough information
to fully reconstruct the simulator state later.
"""

import os
import pickle
from typing import List

import grid2op
import numpy as np
from grid2op.Agent import DoNothingAgent
from grid2op.Environment import Environment
from lightsim2grid import LightSimBackend

# ============================================================
# CONFIG
# ============================================================

ENV_NAME = "l2rpn_case14_sandbox"

RHO_THRESHOLD = 0.9

MAX_STEPS_PER_CHRONIC = 10000

SEED = 12345

CHECKPOINT_DIR = "rho_checkpoints"


# ============================================================
# CHECKPOINT FUNCTION
# ============================================================


def save_checkpoint(
    filename: str,
    env: Environment,
    seed: int,
    actions_taken: List,
) -> None:
    """
    Save enough information to reconstruct
    the exact simulator state later.
    """

    checkpoint = {
        "env_name": ENV_NAME,
        "seed": seed,
        "chronics_id": env.chronics_handler.get_id(),
        "nb_steps": len(actions_taken),
        "actions": actions_taken,
    }

    with open(filename, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"\nCheckpoint saved to: {filename}")
    print(f"Saved timestep: {len(actions_taken)}")
    print(f"Saved chronic id: {checkpoint['chronics_id']}")


# ============================================================
# CREATE CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\nCheckpoint directory: {CHECKPOINT_DIR}")


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = grid2op.make(ENV_NAME, backend=LightSimBackend())

agent = DoNothingAgent(env.action_space)


# ============================================================
# GET CHRONICS
# ============================================================

chronics = env.chronics_handler.available_chronics()

print(f"\nFound {len(chronics)} chronics.\n")


# ============================================================
# MAIN LOOP
# ============================================================

for chronic_idx, chronic_name in enumerate(chronics):

    print("=" * 70)
    print(f"CHRONIC INDEX : {chronic_idx}")
    print(f"CHRONIC NAME  : {chronic_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load chronic
    # --------------------------------------------------------

    env.set_id(chronic_idx)

    obs = env.reset(seed=SEED)

    actions_taken = []

    found_threshold = False

    # --------------------------------------------------------
    # Run episode
    # --------------------------------------------------------

    for step in range(MAX_STEPS_PER_CHRONIC):

        act = agent.act(obs, reward=0.0, done=False)

        actions_taken.append(act)

        obs, reward, done, info = env.step(act)

        max_rho = float(np.max(obs.rho))

        print(f"Step={step:04d} " f"max_rho={max_rho:.4f} " f"reward={reward:8.4f} " f"done={done}")

        # ----------------------------------------------------
        # Threshold reached
        # ----------------------------------------------------

        if max_rho > RHO_THRESHOLD:

            print("\n>>> THRESHOLD REACHED <<<")
            print(f"max_rho = {max_rho:.4f}")
            print(f"step     = {step}")
            print(f"chronic  = {chronic_name}")

            found_threshold = True

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            checkpoint_filename = f"chronic_{chronic_idx:04d}_" f"step_{step:05d}_" f"rho_{max_rho:.3f}.pkl"

            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)

            save_checkpoint(
                checkpoint_path,
                env,
                SEED,
                actions_taken,
            )

            print()

            break

        # ----------------------------------------------------
        # Episode terminated
        # ----------------------------------------------------

        if done:

            print("\nEpisode terminated before threshold.\n")

            break

    # --------------------------------------------------------
    # Chronic summary
    # --------------------------------------------------------

    if not found_threshold:

        print(f"No rho > {RHO_THRESHOLD} found " f"within {MAX_STEPS_PER_CHRONIC} steps.")

    print("\nMoving to next chronic...\n")


# ============================================================
# CLEANUP
# ============================================================

env.close()

print("\nFinished scanning all chronics.")
