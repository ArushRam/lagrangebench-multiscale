from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import jax.random as random
import numpy as np

from .base import BasePooling

class RandomSampling(BasePooling):
    def __init__(self, sampling_ratio, num_particle_types):
        super().__init__()
        self.sampling_ratio = sampling_ratio
        self.num_particle_types = num_particle_types

    def __call__(self, positions: jnp.ndarray, particle_type: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """Forward pass of VoxelClustering.

        We specify the dimensions of the inputs and outputs using the number of nodes N,
        the number of edges E, number of historic velocities K (=input_seq_length - 1),
        and the dimensionality of the feature vectors dim.

        Args:
            positions: Array containing positional coordinates for each node.
                - shape: (N, dim), absolute positions
                - filler values according to particle_type == -1
            particle_type: Array containing particle types for each node.
                - shape: (N), integers
                - filler values are -1
        Returns:
            Tuple with dictionary of cluster information and coarse particle types. 
                - Dictionary contains:
                    - "coarse_ids" (N) - assignments for each original node
                - Particle types: (N') - types for each coarse-level particle
        """

        index_assignments = jnp.zeros_like(particle_type, dtype=int)
        n_particles = positions.shape[0]

        # 1. sample K nodes
        sample_size = int(self.sampling_ratio * n_particles)
        rng = hk.next_rng_key()
        sample_idx = random.choice(
            key = rng,
            a = jnp.arange(n_particles),
            shape = (sample_size,),
            replace = False,
        )

        # 2. for each of N nodes, compute K distances
        sampled_positions = positions[sample_idx]
        displacements = positions[:, None, :] - sampled_positions[None, :, :]
        l2_dists = jnp.sqrt(jnp.sum(displacements ** 2, axis=2))

        # 3. assign each of the N nodes to closest sampled node
        index_assignments = jnp.argmin(l2_dists, axis=1)

        coarse_sample = {
            'coarse_ids': index_assignments
        }

        return coarse_sample, jnp.ones(sample_size)