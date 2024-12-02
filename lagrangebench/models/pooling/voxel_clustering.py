from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import numpy as np

from .base import BasePooling

class VoxelClustering(BasePooling):
    '''
    Forward pass usage:

    voxel_size = ...
    bounds = ...

    def voxel_clustering_forward(features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray):
        model = VoxelClustering(voxel_size, bounds)
        return model(features, particle_type)

    # Transform the function using Haiku
    voxel_clustering_fn = hk.without_apply_rng(hk.transform(voxel_clustering_forward))

    # Initialize the Haiku module
    params = voxel_clustering_fn.init(None, features, particle_type)

    # Apply the module
    coarse_sample, new_particle_types = voxel_clustering_fn.apply(params, features, particle_type)
    '''
    def __init__(self, voxel_size, bounds, num_particle_types, size):
    # def __init__(self, voxel_size, bounds, num_particle_types):
        """
        Arguments:
            - voxel_size (float): Dimension of a single voxel.
            - bounds (jnp.ndarray): Bounds of the simulation space.
            - num_particle_types (int): Number of unique particle types.
            - size (int): Precomputed size for unique combinations.
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.dim = bounds.shape[0]
        self.bounds_max = bounds[:, 1]
        self.bounds_min = bounds[:, 0]
        self.grid_size = (self.bounds_max - self.bounds_min) / self.voxel_size
        self.num_particle_types = num_particle_types
        self.size = size  # Directly use precomputed size

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

        # 1. map each node position to an index
        voxel_indices = ((positions - self.bounds_min) // self.voxel_size).astype(int)
        voxel_indices = jnp.clip(voxel_indices, 0, jnp.array(self.grid_size) - 1)
        
        # 2. Combine voxel indices and particle type
        voxel_indices = jnp.where(particle_type[:, None] == -1, -1, voxel_indices)
        combined_indices = jnp.concatenate([voxel_indices, particle_type[:, None]], axis=1)
        
        # 3. Get unique combinations and inverse indices
        unique_combinations, inverse_indices = jnp.unique(
            combined_indices,
            axis=0,
            return_inverse=True,
            size=self.size + 1,
            fill_value=-jnp.ones(combined_indices.shape[1], dtype=combined_indices.dtype)
        )
        
        # 4. Check if there are any filler values; if so, adjust unique_combinations and inverse_indices
        has_filler = jnp.any((combined_indices == -1).all(axis=1))
        
        unique_combinations = jnp.where(
            has_filler,
            unique_combinations[1:],
            unique_combinations[:-1]
        )
        
        inverse_indices = jnp.where(
            has_filler,
            jnp.where(inverse_indices == 0, -1, inverse_indices - 1),
            inverse_indices
        )

        # 5. Extract unique particle types from unique_combinations
        unique_particle_types = unique_combinations[:, 1].astype(int)

        index_assignments = inverse_indices[:, 0]
        coarse_sample = {
            'coarse_ids': index_assignments
        }

        return coarse_sample, unique_particle_types