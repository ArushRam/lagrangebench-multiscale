from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import jax.random as random
import numpy as np
from ott.tools.k_means import k_means

from .base import BasePooling

class KMeans(BasePooling):
    def __init__(self, cluster_ratio, num_particle_types):
        super().__init__()
        self.cluster_ratio = cluster_ratio
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

        # 1. use k-means algorithm
        k = int(self.cluster_ratio * n_particles)
        kmeans_out = k_means(positions, k)
        index_assignments = kmeans_out.assignment

        coarse_sample = {
            'coarse_ids': index_assignments
        }

        return coarse_sample, jnp.ones(k)