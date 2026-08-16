#!/usr/bin/env python3
"""Serve pi0.5-LIBERO with deterministic per-query flow noise."""

import argparse
import logging
import socket

import numpy as np


class DeterministicNoisePolicy:
    """Make stochastic flow sampling match exactly across paired arms."""

    def __init__(self, policy):
        self._policy = policy
        self.metadata = {
            **policy.metadata,
            "libero_max_deterministic_noise": True,
        }
        self._action_horizon = int(policy._model.action_horizon)
        self._action_dim = int(policy._model.action_dim)

    def infer(self, observation):
        observation = dict(observation)
        try:
            seed = int(observation.pop("libero_max_noise_seed"))
        except KeyError as exc:
            raise ValueError("request is missing libero_max_noise_seed") from exc
        noise = np.random.default_rng(seed).standard_normal(
            (self._action_horizon, self._action_dim), dtype=np.float32
        )
        return self._policy.infer(observation, noise=noise)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_libero")
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    policy = policy_config.create_trained_policy(
        config.get_config(args.config), args.checkpoint
    )
    policy = DeterministicNoisePolicy(policy)
    hostname = socket.gethostname()
    logging.info("Serving deterministic %s on %s:%d", args.config, hostname, args.port)
    WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
