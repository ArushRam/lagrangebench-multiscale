from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp

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
    def __init__(self, voxel_size, bounds, num_particle_types):
        '''
        Arguments:
            - voxel_size (dimension of a single voxel)
            - bounds (dx2 array, rows correspond to dimension, columns correspond to min/max bound)
        '''
        super().__init__()
        self.voxel_size = voxel_size
        self.dim = bounds.shape[0]
        self.bounds_max = bounds[:,1]
        self.bounds_min = bounds[:,0]
        self.grid_size = (self.bounds_max - self.bounds_min) / self.voxel_size
        self.num_particle_types = num_particle_types
        self.size = (self.num_particle_types * jnp.prod(self.grid_size)).astype(int)

    def __call__(self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:

        index_assignments = jnp.zeros_like(particle_type, dtype=int)

        positions = features['abs_pos'][:,-1]

        # 1. map each node position to an index
        voxel_indices = ((positions - self.bounds_min) // self.voxel_size).astype(int)
        voxel_indices = jnp.clip(voxel_indices, 0, jnp.array(self.grid_size) - 1)

        # 2. flatten voxel indices
        flattened_voxel_indices = voxel_indices[:,0] * self.grid_size[1] + voxel_indices[:,1]
        if self.dim == 3:
            flattened_voxel_indices = flattened_voxel_indices * self.grid_size[2] + voxel_indices[:,2]

        # 3. Combine flattened indices and particle type into a single tensor
        combined_indices = jnp.stack([flattened_voxel_indices, particle_type], axis=-1)

        # 4. Get unique combinations and inverse indices
        unique_combinations, inverse_indices = jnp.unique(combined_indices, axis=0, return_inverse=True, size=self.size)

        # 5. Extract unique particle types from unique_combinations
        unique_particle_types = unique_combinations[:, 1].astype(int)

        index_assignments = inverse_indices
        coarse_sample = {
            'coarse_ids': index_assignments
        }

        return coarse_sample, unique_particle_types