import jax
from jax import vmap, numpy as jnp
from .utils import unflatten_conns

from .base import BaseGenome
from .gene import DefaultNode, DefaultConn
from .operations import DefaultMutation, DefaultCrossover, DefaultDistance
from .utils import unflatten_conns, extract_node_attrs, extract_conn_attrs

from tensorneat.common import attach_with_inf


class GeennsRecurrentGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""
    
    network_type = "recurrent"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        output_transform=None,
        input_transform=None,
        init_hidden_layers=(),
        activate_time=10,
    ):
        super().__init__(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            node_gene,
            conn_gene,
            mutation,
            crossover,
            distance,
            output_transform,
            input_transform,
            init_hidden_layers,
        )
        self.activate_time = activate_time

        # Set input and output indices
        self.input_idx = jnp.arange(num_inputs)  # First num_inputs nodes are inputs
        self.output_idx = jnp.arange(num_outputs) + (max_nodes - num_outputs)  # Last num_outputs nodes are outputs
        
        # Precompute total number of nodes statically
        self.total_nodes = max(num_inputs, num_outputs) + 1  # Static node count

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        
        # Extract connection attributes (e.g., weights) from the genome's connections
        conns_attrs = extract_conn_attrs(conns)
        
        # Extract node attributes (e.g., activation functions) from the genome's nodes
        nodes_attrs = extract_node_attrs(nodes)
        
        # Set the attributes for further use in the forward method
        self.conns_attrs = conns_attrs
        self.nodes_attrs = nodes_attrs
        
        return nodes, conns, u_conns


    def forward(self, state, attrs, inputs):
        """
        Forward pass through the recurrent genome.
        """
        print(f"attrs: {attrs}")
        weight = attrs[0]  # First element of the tuple is the weight

        # Optionally, if attrs contains more parameters (e.g., bias, delay), you can unpack them
        # bias = attrs[1]
        
        # Use the weight (and other parameters if applicable)
        return inputs * weight

        # Use the precomputed total number of nodes
        total_nodes = self.total_nodes

        # Initial values for all nodes (input, hidden, output)
        vals = jnp.zeros(total_nodes)

        # Function to loop through activation steps
        def body_func(_, values):
            # Flatten the input array if necessary
            flattened_inputs = jnp.ravel(inputs)
            
            # Ensure the number of inputs matches the number of input nodes
            if flattened_inputs.shape[0] > len(self.input_idx):
                trimmed_inputs = flattened_inputs[:len(self.input_idx)]
            else:
                trimmed_inputs = flattened_inputs

            # Set input values
            values = values.at[self.input_idx].set(trimmed_inputs)

            # Calculate connections using the connection genes
            node_ins = vmap(
                vmap(self.conn_gene.forward, in_axes=(None, 0, None)),
                in_axes=(None, 0, 0)
            )(state, self.conns_attrs[:len(values)], values)
            
            # Compute new values for nodes (activation of nodes based on connections)
            values = vmap(self.node_gene.forward, in_axes=(None, 0, 0, 0))(
                state, self.nodes_attrs, node_ins.T, self.is_output_nodes
            )
            
            return values

        # Loop through activation steps
        vals = jax.lax.fori_loop(0, self.activate_time, body_func, vals)

        # Apply output transform if specified, or return final output node values
        if self.output_transform is None:
            return vals[self.output_idx]
        else:
            return self.output_transform(vals[self.output_idx])


    def sympy_func(self, state, network, precision=3):
        raise ValueError("Sympy function is not supported for Recurrent Network!")

    def visualize(self, network):
        raise ValueError("Visualize function is not supported for Recurrent Network!")