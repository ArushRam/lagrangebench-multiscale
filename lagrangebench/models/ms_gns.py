"""
Graph Network-based Simulator.
GNS model and feature transform.
"""

from typing import Dict, Tuple, List

import haiku as hk
import jax.numpy as jnp
import jraph
import jax
import numpy as np
from functools import partial
from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import build_mlp, scatter_mean_new, scatter_mean
from .pooling import VoxelClustering, RandomSampling, KMeans

MAX_SIZE = 40000
SAMPLING_RATIO = 0.2
KMEANS_CLUSTER_RATIO = 0.05

class MultiScaleGNS(BaseModel):
    r"""Multi-scale Graph Network-based Simulator.
    """

    def __init__(
        self,
        particle_dimension: int,
        latent_size: int,
        blocks_per_step: int,
        # num_mp_steps: int,
        particle_type_embedding_size: int,
        num_particle_types: int = NodeType.SIZE,
        # Multiscale GNS parameters
        num_scales: int = 1,
        mp_steps_per_scale: List[int] = [10],
        clustering_type: str = "voxel",
        max_size: int = MAX_SIZE,
    ):
        """Initialize the model.

        Args:
            particle_dimension: Space dimensionality (e.g. 2 or 3).
            latent_size: Size of the latent representations.
            blocks_per_step: Number of MLP layers per block.
            particle_type_embedding_size: Size of the particle type embedding.
            num_particle_types: Max number of particle types.
            num_scales: Number of scales for multi-scale message passing.
            mp_steps_per_scale: List of message passing steps per scale.
            clustering_type: Type of clustering to use ("voxel").
        """
        super().__init__()
        self._output_size = particle_dimension
        self._latent_size = latent_size
        self._blocks_per_step = blocks_per_step
        # self._mp_steps = num_mp_steps
        self._num_particle_types = num_particle_types
        
        assert len(mp_steps_per_scale) == num_scales
        self._num_scales = num_scales
        self._mp_steps_per_scale = mp_steps_per_scale

        self._max_size = max_size
        
        if clustering_type == "voxel":
            self._base_voxel_size = 5e-2
            self._voxel_size_per_scale = [self._base_voxel_size * (2 ** i) for i in range(num_scales - 1)]
            bounds = np.array([[0.0, 1.0], [0.0, 2.0]])  # Example bounds
            
            self._clustering_fn_per_scale = [
                VoxelClustering(
                    voxel_size=self._voxel_size_per_scale[i],
                    bounds=jnp.array(bounds),
                    num_particle_types=self._num_particle_types,
                    size=self._max_size,
                )
                for i in range(num_scales - 1)
            ]
            self._scatter_fn = partial(scatter_mean_new, num_segments=self._max_size)

        elif clustering_type == "random":
            self._sampling_ratio = SAMPLING_RATIO
            self._clustering_fn_per_scale = [
                RandomSampling(
                    sampling_ratio = self._sampling_ratio,
                    num_particle_types = self._num_particle_types,
                )
                for i in range(num_scales - 1)
            ]
            # self._scatter_fn = partial(scatter_mean)
            self._scatter_fn = partial(scatter_mean_new, num_segments=self._max_size)

        elif clustering_type == "kmeans":
            self._clustering_fn_per_scale = [
                KMeans(
                    cluster_ratio = KMEANS_CLUSTER_RATIO,
                    num_particle_types = self._num_particle_types,
                )
                for i in range(num_scales - 1)
            ]
            # self._scatter_fn = partial(scatter_mean)
            self._scatter_fn = partial(scatter_mean_new, num_segments=self._max_size)
        else:
            raise ValueError(f"Unknown clustering type: {clustering_type}")

        self._embedding = hk.Embed(
            num_particle_types, particle_type_embedding_size
        )  # (9, 16)

    def _encoder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """MLP graph encoder."""
        node_latents = build_mlp(
            self._latent_size, self._latent_size, self._blocks_per_step
        )(graph.nodes)
        edge_latents = build_mlp(
            self._latent_size, self._latent_size, self._blocks_per_step
        )(graph.edges)
        return jraph.GraphsTuple(
            nodes=node_latents,
            edges=edge_latents,
            globals=graph.globals,
            receivers=graph.receivers,
            senders=graph.senders,
            n_node=jnp.asarray([node_latents.shape[0]]),
            n_edge=jnp.asarray([edge_latents.shape[0]]),
        )

    def _processor(self, graph: jraph.GraphsTuple, particle_type: jnp.ndarray, positions: jnp.ndarray) -> jraph.GraphsTuple:
        """Sequence of Graph Network blocks with multi-scale processing."""
        
        graphs = []
        if self._num_scales == 1:
            # No multi-scale processing
            for _ in range(self._mp_steps_per_scale[0]):
                graph = self._within_scale_message_passing(graph)
            graphs[0] = graph
        else:
            # Store graphs and positions at each scale
            clusters_at_scales = []

            # Original resolution message passing
            for _ in range(self._mp_steps_per_scale[0] // 2):
                graph = self._within_scale_message_passing(graph)
                
            # Fill graph attributes with filler values
            graph = self._fill_graph(graph, self._max_size)
            particle_type = jnp.concatenate([particle_type, jnp.full((self._max_size - particle_type.shape[0],), -1)])
            positions = jnp.concatenate([positions, jnp.full((self._max_size - positions.shape[0], self._output_size), 0.0)])
            graphs.append(graph)
            
            # Downward pass: scale -> scale + 1
            for scale in range(self._num_scales - 1):
                # Down message passing with positions
                coarse_graph, coarse_particle_types, coarse_pos, clusters = self._down_mp(
                    scale,
                    graphs[-1], 
                    particle_type, 
                    positions
                )

                # Buffer coarse graph and cluster assignments
                graphs.append(coarse_graph)
                clusters_at_scales.append(clusters)
                
                # Message passing at this scale
                if scale + 1 == self._num_scales:
                    n_mp_steps = self._mp_steps_per_scale[scale + 1]
                else:
                    n_mp_steps = self._mp_steps_per_scale[scale + 1] // 2
                for _ in range(n_mp_steps):
                    coarse_graph = self._within_scale_message_passing(coarse_graph)
                graphs[-1] = coarse_graph

                # Update positions and particle_type for next iteration
                positions = coarse_pos
                particle_type = coarse_particle_types
            
            # Upward pass: scale + 1 -> scale
            for scale in reversed(range(self._num_scales - 1)):
                graphs[scale] = self._up_mp(graphs[scale], graphs[scale + 1], clusters_at_scales[scale])
                for _ in range(self._mp_steps_per_scale[scale] // 2):
                    graphs[scale] = self._within_scale_message_passing(graphs[scale])
            
            graphs[0] = self._unfill_graph(graphs[0])
                
        return graphs[0]

    def _decoder(self, graph: jraph.GraphsTuple):
        """MLP graph node decoder."""
        return build_mlp(
            self._latent_size,
            self._output_size,
            self._blocks_per_step,
            is_layer_norm=False,
        )(graph.nodes)

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]:
        """Convert physical features to jraph.GraphsTuple."""
        n_total_points = features["vel_hist"].shape[0]
        node_features = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features
        ]
        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features]

        graph = jraph.GraphsTuple(
            nodes=jnp.concatenate(node_features, axis=-1),
            edges=jnp.concatenate(edge_features, axis=-1),
            receivers=features["receivers"],
            senders=features["senders"],
            n_node=jnp.array([n_total_points]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )

        return graph, particle_type, features["abs_pos"]

    def _fill_graph(
        self, 
        graph: jraph.GraphsTuple, 
        fill_size: int, 
        feature_filler_value: float=0.0, 
        idx_filler_value: int=-1, 
    ) -> jraph.GraphsTuple:
        """Fill node and edge features with filler values."""
        self.original_node_size = graph.nodes.shape[0]
        self.original_edge_size = graph.edges.shape[0]
        return graph._replace(
            nodes=jnp.concatenate([graph.nodes, jnp.full((fill_size - graph.nodes.shape[0], graph.nodes.shape[1]), feature_filler_value)]),
            edges=jnp.concatenate([graph.edges, jnp.full((fill_size - graph.edges.shape[0], graph.edges.shape[1]), feature_filler_value)]),
            senders=jnp.concatenate([graph.senders, jnp.full((fill_size - graph.senders.shape[0],), idx_filler_value)]),
            receivers=jnp.concatenate([graph.receivers, jnp.full((fill_size - graph.receivers.shape[0],), idx_filler_value)]),
            n_node=jnp.array([fill_size]),
            n_edge=jnp.array([fill_size]),
        )
    
    def _unfill_graph(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Unfill node and edge features with filler values."""
        return graph._replace(
            nodes=graph.nodes[:self.original_node_size],
            edges=graph.edges[:self.original_edge_size],
            senders=graph.senders[:self.original_edge_size],
            receivers=graph.receivers[:self.original_edge_size],
            n_node=jnp.array([self.original_node_size]),
            n_edge=jnp.array([self.original_edge_size]),
        )

    def _within_scale_message_passing(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Message passing within a single scale."""
        def update_edge_features(
            edge_features,
            sender_node_features,
            receiver_node_features,
            _,  # globals_
        ):
            # Create mask for valid edges (assuming edge_features contains a mask or can derive one)
            mask = (edge_features != 0.0).any(axis=-1, keepdims=True)
            
            # Concatenate features
            concatenated = jnp.concatenate(
                [sender_node_features, receiver_node_features, edge_features],
                axis=-1,
            )
            
            # Apply MLP
            update_fn = build_mlp(
                self._latent_size, self._latent_size, self._blocks_per_step
            )
            result = update_fn(concatenated)
            
            # Mask out invalid edges
            return jnp.where(mask, result, 0.0)

        def update_node_features(
            node_features,
            _,  # aggr_sender_edge_features,
            aggr_receiver_edge_features,
            __,  # globals_,
        ):
            # Create mask for valid nodes (assuming node_features contains valid data)
            mask = (node_features != 0.0).any(axis=-1, keepdims=True)
            
            # Concatenate features
            features = [node_features, aggr_receiver_edge_features]
            concatenated = jnp.concatenate(features, axis=-1)
            
            # Apply MLP
            update_fn = build_mlp(
                self._latent_size, self._latent_size, self._blocks_per_step
            )
            result = update_fn(concatenated)
            
            # Mask out invalid nodes
            return jnp.where(mask, result, 0.0)
        
        return jraph.GraphNetwork(
            update_edge_fn=update_edge_features,
            update_node_fn=update_node_features
        )(graph)
    
    def _down_mp(
        self, 
        scale: int,
        graph: jraph.GraphsTuple, 
        particle_type: jnp.ndarray, 
        pos: jnp.ndarray
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Downscale graph through clustering based on positions"""
        
        def create_cluster_edges(
            edges: jnp.ndarray, 
            senders: jnp.ndarray, 
            receivers: jnp.ndarray, 
            clusters: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            # Mask out filler values
            mask = (senders != -1) & (receivers != -1)
            valid_senders = jnp.where(mask, senders, 0)
            valid_receivers = jnp.where(mask, receivers, 0)
            
            # Get unique cluster pairs from original edges
            cluster_pairs = jnp.where(
                mask[:, None],
                jnp.stack([clusters[valid_senders], clusters[valid_receivers]], axis=1),
                -jnp.ones((2,), dtype=clusters.dtype)
            )
            
            unique_pairs, inverse_indices = jnp.unique(
                cluster_pairs, 
                axis=0, 
                return_inverse=True, 
                size=self._max_size + 1, 
                fill_value=-jnp.ones(cluster_pairs.shape[1], dtype=cluster_pairs.dtype)
            )
            
            # Remove filler values from unique pairs (always the first element)
            unique_pairs = unique_pairs[1:]
            inverse_indices = jnp.where(inverse_indices == 0, -1, inverse_indices - 1)
            
            # Create new edges between unique cluster pairs
            new_senders = unique_pairs[:,0]
            new_receivers = unique_pairs[:,1]
            
            # Pool edge features by taking mean within each unique cluster pair
            new_edges = self._scatter_fn(edges, inverse_indices[:, 0])
            
            return new_edges, new_senders, new_receivers
        
        # Get cluster assignments using positions
        coarse_sample, pooled_particle_types = self._clustering_fn_per_scale[scale](pos, particle_type)
        clusters = coarse_sample['coarse_ids']
        
        # Pool node features and positions
        pooled_nodes = self._scatter_fn(graph.nodes, clusters)
        pooled_pos = self._scatter_fn(pos, clusters)
        
        # Create new edges between clusters
        new_edges, new_senders, new_receivers = create_cluster_edges(graph.edges, graph.senders, graph.receivers, clusters)
        
        return (
            jraph.GraphsTuple(
                nodes=pooled_nodes,
                edges=new_edges,
                senders=new_senders,
                receivers=new_receivers,
                n_node=jnp.array([pooled_nodes.shape[0]]),
                n_edge=jnp.array([new_edges.shape[0]]),
                globals=None
            ),
            pooled_particle_types, 
            pooled_pos,
            clusters, 
        )
    
    def _up_mp(self, graph_fine: jraph.GraphsTuple, graph_coarse: jraph.GraphsTuple, clusters: jnp.ndarray) -> jraph.GraphsTuple:
        """Upscale graph through unpooling based on cluster assignments"""
        # Create masks for valid nodes and edges
        fine_node_mask = clusters != -1
        fine_edge_mask = (graph_fine.senders != -1) & (graph_fine.receivers != -1)
        coarse_edge_mask = (graph_coarse.senders != -1) & (graph_coarse.receivers != -1)

        # Unpool node features from coarse to fine graph using cluster assignments
        fine_nodes = graph_coarse.nodes[clusters]

        # Unpool edge features by mapping coarse edges back to fine edges
        fine_cluster_pairs = jnp.where(
            fine_edge_mask[:, None],
            jnp.stack([clusters[graph_fine.senders], clusters[graph_fine.receivers]], axis=1),
            -1
        )
        coarse_cluster_pairs = jnp.where(
            coarse_edge_mask[:, None],
            jnp.stack([graph_coarse.senders, graph_coarse.receivers], axis=1),
            -1
        )
        
        # For each valid fine edge, find matching coarse edge and copy features using a loop
        # Vectorized version that works with JAX tracers
        matches = jnp.all(fine_cluster_pairs[:, None] == coarse_cluster_pairs[None, :], axis=2) # (n_fine_edges, n_coarse_edges)
        coarse_edge_indices = jnp.argmax(matches, axis=1)
        coarse_edge_indices = jnp.where(fine_edge_mask, coarse_edge_indices, 0)
        fine_edges = graph_coarse.edges[coarse_edge_indices]

        return jraph.GraphsTuple(
            nodes=jnp.where(fine_node_mask[:, None], graph_fine.nodes + fine_nodes, graph_fine.nodes),
            edges=jnp.where(fine_edge_mask[:, None], graph_fine.edges + fine_edges, graph_fine.edges),
            senders=graph_fine.senders,
            receivers=graph_fine.receivers,
            n_node=graph_fine.n_node,
            n_edge=graph_fine.n_edge,
            globals=None
        )
    
    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        graph, particle_type, positions = self._transform(*sample)
        positions = positions[:,-1]

        if self._num_particle_types > 1:
            particle_type_embeddings = self._embedding(particle_type)
            new_node_features = jnp.concatenate(
                [graph.nodes, particle_type_embeddings], axis=-1
            )
            graph = graph._replace(nodes=new_node_features)
        acc = self._decoder(self._processor(self._encoder(graph), particle_type, positions))
        return {"acc": acc}
