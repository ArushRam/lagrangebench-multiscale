"""
Graph Network-based Simulator.
GNS model and feature transform.
"""

from typing import Dict, Tuple, List

import haiku as hk
import jax.numpy as jnp
import jraph
import jax

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import build_mlp, scatter_mean
from .pooling import VoxelClustering


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
        
        if clustering_type == "voxel":
            self._base_voxel_size = 5e-2
            self._voxel_size_per_scale = [self._base_voxel_size * (2 ** i) for i in range(num_scales - 1)]
            self._clustering_fn_per_scale = [
                VoxelClustering(
                    voxel_size=self._voxel_size_per_scale[i], 
                    bounds=jnp.array([[0.0, 1.0], [0.0, 2.0]]), 
                ) for i in range(num_scales - 1)
            ]
        else:
            raise ValueError(f"Unknown clustering type: {clustering_type}")
        self._scatter_fn = scatter_mean

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

    def _within_scale_message_passing(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Message passing within a single scale."""
        def update_edge_features(
            edge_features,
            sender_node_features,
            receiver_node_features,
            _,  # globals_
        ):
            update_fn = build_mlp(
                self._latent_size, self._latent_size, self._blocks_per_step
            )
            # Calculate sender node features from edge features
            return update_fn(
                jnp.concatenate(
                    [sender_node_features, receiver_node_features, edge_features],
                    axis=-1,
                )
            )

        def update_node_features(
            node_features,
            _,  # aggr_sender_edge_features,
            aggr_receiver_edge_features,
            __,  # globals_,
        ):
            update_fn = build_mlp(
                self._latent_size, self._latent_size, self._blocks_per_step
            )
            features = [node_features, aggr_receiver_edge_features]
            return update_fn(jnp.concatenate(features, axis=-1))
        
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
            # Get unique cluster pairs from original edges
            cluster_pairs = jnp.stack([clusters[senders], clusters[receivers]], axis=1)
            unique_pairs, inverse_indices = jnp.unique(cluster_pairs, axis=0, return_inverse=True)
            
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
        pooled_pos = self._scatter_fn(pos, clusters)  # e.g., cluster centroids
        
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
        # Unpool node features from coarse to fine graph using cluster assignments
        fine_nodes = graph_coarse.nodes[clusters]
        
        # Unpool edge features by mapping coarse edges back to fine edges
        fine_cluster_pairs = jnp.stack([clusters[graph_fine.senders], clusters[graph_fine.receivers]], axis=1) # (n_fine_edges, 2)
        coarse_cluster_pairs = jnp.stack([graph_coarse.senders, graph_coarse.receivers], axis=1) # (n_coarse_edges, 2)
        
        # For each fine edge, find matching coarse edge and copy features
        # NOTE: fallback to for loop if memory is an issue
        coarse_edge_indices = jnp.argmax(
            jnp.all(fine_cluster_pairs[:, None] == coarse_cluster_pairs[None, :], axis=2), # (n_fine_edges, n_coarse_edges)
            axis=1
        )
        fine_edges = graph_coarse.edges[coarse_edge_indices]

        return jraph.GraphsTuple(
            nodes=graph_fine.nodes + fine_nodes, # residual connection
            edges=graph_fine.edges + fine_edges, # residual connection
            senders=graph_fine.senders,
            receivers=graph_fine.receivers,
            n_node=graph_fine.n_node,
            n_edge=graph_fine.n_edge,
            globals=None
        )
    
    def _processor(self, graph: jraph.GraphsTuple, particle_type: jnp.ndarray, positions: jnp.ndarray) -> jraph.GraphsTuple:
        """Sequence of Graph Network blocks with multi-scale processing."""
        
        if self._num_scales == 1:
            # No multi-scale processing
            for _ in range(self._mp_steps_per_scale[0]):
                graph = self._within_scale_message_passing(graph)
            graphs[0] = graph
        else:
            # Store graphs and positions at each scale
            graphs = [graph]
            clusters_at_scales = []

            # Original resolution message passing
            for _ in range(self._mp_steps_per_scale[0] // 2):
                graph = self._within_scale_message_passing(graph)
            graphs[0] = graph
            
            # Downward pass: scale -> scale + 1
            for scale in range(self._num_scales - 1):
                print(f"Downward pass: scale {scale} -> {scale + 1}")
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
                print(f"Coarse graph nodes: {coarse_graph.n_node}")
                
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
                print(f"Upward pass: scale {scale + 1} -> {scale}")
                graphs[scale] = self._up_mp(graphs[scale + 1], graphs[scale], clusters_at_scales[scale])
                for _ in range(self._mp_steps_per_scale[scale] // 2):
                    graphs[scale] = self._within_scale_message_passing(graphs[scale])
                
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
