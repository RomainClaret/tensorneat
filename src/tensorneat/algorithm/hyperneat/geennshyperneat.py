from typing import Callable

import jax
from jax import vmap, numpy as jnp

from .substrate import *
from tensorneat.common import State, ACT, AGG
from tensorneat.algorithm import BaseAlgorithm, NEAT
from tensorneat.genome import BaseNode, BaseConn, RecurrentGenome


class geennsHyperNEAT(BaseAlgorithm):
    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        weight_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation: Callable = AGG.sum,
        activation: Callable = ACT.sigmoid,
        activate_time: int = 10,
        output_transform: Callable = ACT.sigmoid,
    ):
        assert (
            substrate.query_coors.shape[1] == neat.num_inputs
        ), "Query coors of Substrate should be equal to NEAT input size"
        
        self.substrate = substrate
        self.neat = neat
        self.weight_threshold = weight_threshold
        self.max_weight = max_weight
        self.hyper_genome = RecurrentGenome(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=HyperNEATNode(aggregation, activation),
            conn_gene=HyperNEATConn(),
            activate_time=activate_time,
            output_transform=output_transform,
        )
        self.pop_size = neat.pop_size

    def setup(self, state=State()):
        state = self.neat.setup(state)
        state = self.substrate.setup(state)
        return self.hyper_genome.setup(state)

    def ask(self, state):
        return self.neat.ask(state)

    def tell(self, state, fitness):
        state = self.neat.tell(state, fitness)
        return state

    def transform(self, state, individual):
        transformed = self.neat.transform(state, individual)
        
        # CPPN query: ensure correct shape after querying
        query_res = vmap(self.neat.forward, in_axes=(None, None, 0))(
            state, transformed, self.substrate.query_coors
        )

        # Ensure the query results have the correct shape
        if query_res.ndim > 1:
            query_res = jnp.ravel(query_res)

        # Process nodes and connections based on query results
        h_nodes, h_conns = self.substrate.make_nodes(query_res), self.substrate.make_conns(query_res)
        
        return self.hyper_genome.transform(state, h_nodes, h_conns)

    def forward(self, state, transformed, inputs):
        # Add bias to inputs
        inputs_with_bias = jnp.concatenate([inputs, jnp.array([1])])
        
        # Ensure inputs_with_bias is flattened if necessary
        if inputs_with_bias.ndim > 1:
            inputs_with_bias = jnp.ravel(inputs_with_bias)

        # Pass properly formatted inputs to the genome's forward method
        res = self.hyper_genome.forward(state, transformed, inputs_with_bias)
        return res
