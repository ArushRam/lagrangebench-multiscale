from abc import ABC, abstractmethod
from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp


class BasePooling(hk.Module, ABC):
    """Base model class. All models must inherit from this class."""

    @abstractmethod
    def __call__(self, positions: jnp.ndarray, particle_type: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """Forward pass.

        We specify the dimensions of the inputs and outputs using the number of nodes N,
        the number of edges E, number of historic velocities K (=input_seq_length - 1),
        and the dimensionality of the feature vectors dim.

        Args:
            positions: Array containing positional coordinates for each node.
                - shape: (N, dim), absolute positions
            particle_type: Array containing particle types for each node.
                - shape: (N), integers
        Returns:
            Tuple with dictionary of cluster information and coarse particle types. 
                - Dictionary contains:
                    - "coarse_ids" (N) - assignments for each original node
                - Particle types: (N') - types for each coarse-level particle
        """
        raise NotImplementedError