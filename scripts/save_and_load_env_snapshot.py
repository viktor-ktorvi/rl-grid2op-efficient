#!/usr/bin/env python3

"""
Robust Grid2Op checkpoint / resume example
using the built-in GreedyTopologyAgent.

This demonstrates how to:
1. Run a non-trivial topology-changing agent
2. Save progress safely
3. Restore later
4. Continue exactly from the same state

Recommended approach:
    save (seed + chronic_id + action history)
instead of pickling the environment itself.
"""

import os
import pickle
from typing import List

import grid2op
from grid2op.Agent import TopologyGreedy
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend

# ============================================================
# CONFIG
# ============================================================

ENV_NAME = "l2rpn_case14_sandbox"

CHECKPOINT_FILE = "grid2op_checkpoint.pkl"

SEED = 12345

INITIAL_RUN_STEPS = 50
CONTINUE_RUN_STEPS = 20


# ============================================================
# CHECKPOINT SAVE
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
# CHECKPOINT LOAD + REPLAY
# ============================================================


def load_checkpoint(filename: str) -> tuple[Environment, BaseObservation, dict]:
    """
    Recreate environment and replay all actions.
    """

    with open(filename, "rb") as f:
        checkpoint = pickle.load(f)

    print("\nLoading checkpoint...")
    print(f"Checkpoint timestep: {checkpoint['nb_steps']}")
    print(f"Checkpoint chronic id: {checkpoint['chronics_id']}")

    # Recreate environment
    env = grid2op.make(checkpoint["env_name"], backend=LightSimBackend())

    # Select same chronic/scenario
    env.set_id(checkpoint["chronics_id"])

    # Reset with same RNG seed
    obs = env.reset(seed=checkpoint["seed"])

    # Replay all prior actions
    for i, act in enumerate(checkpoint["actions"], start=1):

        obs, reward, done, info = env.step(act)

        if done:
            raise RuntimeError(f"Replay failed: environment terminated at step {i}")

    print("Replay complete.")
    print(f"Environment restored to timestep {len(checkpoint['actions'])}")

    return env, obs, checkpoint


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    # --------------------------------------------------------
    # FIRST SESSION
    # --------------------------------------------------------

    print("\n==============================")
    print("FIRST SESSION")
    print("==============================")

    env = grid2op.make(ENV_NAME, backend=LightSimBackend())

    # Greedy topology agent:
    # tries topology actions and picks the best immediate reward
    agent = TopologyGreedy(env.action_space)

    obs = env.reset(seed=SEED)

    actions_taken = []

    for step in range(INITIAL_RUN_STEPS):

        act = agent.act(obs, reward=0.0, done=False)

        actions_taken.append(act)

        obs, reward, done, info = env.step(act)

        print(f"[Session 1] " f"Step={step + 1:03d} " f"Reward={reward:8.4f} " f"Done={done}")

        if done:
            print("Environment terminated early.")
            break

    save_checkpoint(
        CHECKPOINT_FILE,
        env,
        SEED,
        actions_taken,
    )

    env.close()

    # --------------------------------------------------------
    # SECOND SESSION (RESTORE)
    # --------------------------------------------------------

    print("\n==============================")
    print("SECOND SESSION (RESTORED)")
    print("==============================")

    restored_env, obs, checkpoint = load_checkpoint(CHECKPOINT_FILE)

    restored_agent = TopologyGreedy(restored_env.action_space)

    for step in range(CONTINUE_RUN_STEPS):

        act = restored_agent.act(obs, reward=0.0, done=False)

        checkpoint["actions"].append(act)

        obs, reward, done, info = restored_env.step(act)

        absolute_step = checkpoint["nb_steps"] + step + 1

        print(f"[Session 2] " f"Step={absolute_step:03d} " f"Reward={reward:8.4f} " f"Done={done}")

        if done:
            print("Environment terminated.")
            break

    restored_env.close()

    print("\nFinished successfully.")

    # Optional cleanup
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
