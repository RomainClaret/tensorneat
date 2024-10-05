from jax import vmap
import numpy as np
import matplotlib.pyplot as plt

from .default import DefaultSubstrate
from tensorneat.genome.utils import set_conn_attrs

class GeennsHyperNeatSubstrate(DefaultSubstrate):
    def __init__(self, input_coordinates, output_coordinates, hidden_coordinates, dense_connections=False):
        """
        Initialize the substrate with an option for dense or empty connections.
        
        Parameters:
        - input_coordinates: List of coordinates for input nodes.
        - output_coordinates: List of coordinates for output nodes.
        - hidden_coordinates: List of lists, where each list represents a hidden layer's node coordinates.
        - dense_connections: If True, automatically generates a fully connected network.
        """
        self.input_coordinates = input_coordinates
        self.output_coordinates = output_coordinates
        self.hidden_coordinates = hidden_coordinates

        # Combine all coordinates (input + hidden layers + output)
        self.all_coordinates = np.array(input_coordinates + [coord for layer in hidden_coordinates for coord in layer] + output_coordinates)

        # Set up the node counts and coordinates
        self.inputs = len(self.input_coordinates)
        self.outputs = len(self.output_coordinates)
        self.coors = np.array(self.all_coordinates)
        self.nodes = np.array(np.arange(len(self.all_coordinates)))

        # Generate dense connections if requested
        if dense_connections:
            self.conns = self.generate_dense_connections()
        else:
            self.conns = np.array([])  # No connections if empty

    def generate_dense_connections(self):
        """
        Generates a fully connected network.
        Each input connects to all hidden nodes, and all hidden nodes connect to all output nodes.
        Fully connects between hidden layers if there are multiple layers.
        """
        conns = []
        
        # Get the indices of each layer
        hidden_start = self.num_inputs
        output_start = self.nodes_cnt - self.num_outputs
        
        hidden_layer_sizes = [len(layer) for layer in self.hidden_coordinates]
        hidden_layer_ranges = []
        layer_start = hidden_start
        
        for layer_size in hidden_layer_sizes:
            hidden_layer_ranges.append((layer_start, layer_start + layer_size))
            layer_start += layer_size
        
        # Fully connect inputs to the first hidden layer
        for input_idx in range(self.num_inputs):
            for hidden_idx in range(hidden_layer_ranges[0][0], hidden_layer_ranges[0][1]):
                conns.append([input_idx, hidden_idx])
        
        # Fully connect between hidden layers
        for i in range(len(hidden_layer_ranges) - 1):
            current_layer_start, current_layer_end = hidden_layer_ranges[i]
            next_layer_start, next_layer_end = hidden_layer_ranges[i + 1]
            for curr_hidden_idx in range(current_layer_start, current_layer_end):
                for next_hidden_idx in range(next_layer_start, next_layer_end):
                    conns.append([curr_hidden_idx, next_hidden_idx])
        
        # Fully connect last hidden layer to the output nodes
        for hidden_idx in range(hidden_layer_ranges[-1][0], hidden_layer_ranges[-1][1]):
            for output_idx in range(output_start, self.nodes_cnt):
                conns.append([hidden_idx, output_idx])
        
        return np.array(conns)


    def make_nodes(self, query_res):
        # Ensure the nodes are a 2D array where each node has its own row
        return self.nodes[:, np.newaxis] if len(self.nodes.shape) == 1 else self.nodes



    #def make_conns(self, query_res):
    #    # change weight of conns
    #    return vmap(set_conn_attrs)(self.conns, query_res)

    def make_conns(self, query_res):
        assert query_res.shape[0] == self.conns.shape[0], \
            f"Mismatch between number of connection queries {query_res.shape[0]} and substrate connections {self.conns.shape[0]}"
        
        # Adjust weights for the connections based on the query results
        return vmap(set_conn_attrs)(self.conns, query_res)


    @property
    def query_coors(self):
        """
        Generate all query coordinates for the CPPN.
        This will return the coordinate pairs needed to query the CPPN for weights
        between input-to-hidden, hidden-to-hidden, and hidden-to-output connections.
        Each query includes coordinates of the source and target nodes, and a bias input.
        """
        # Initialize list to store all coordinate pairs
        all_queries = []
        
        # Input to first hidden layer queries
        input_to_hidden = [(in_coor, h_coor) 
                        for in_coor in self.input_coordinates 
                        for h_coor in self.hidden_coordinates[0]]
        all_queries.extend(input_to_hidden)
        
        # Hidden to hidden queries (between consecutive hidden layers)
        for i in range(len(self.hidden_coordinates) - 1):
            hidden_to_hidden = [(h1_coor, h2_coor)
                                for h1_coor in self.hidden_coordinates[i]
                                for h2_coor in self.hidden_coordinates[i + 1]]
            all_queries.extend(hidden_to_hidden)
        
        # Last hidden layer to output layer queries
        hidden_to_output = [(h_coor, out_coor) 
                            for h_coor in self.hidden_coordinates[-1] 
                            for out_coor in self.output_coordinates]
        all_queries.extend(hidden_to_output)
        
        # Convert the list of tuples to a numpy array
        all_queries = np.array(all_queries)
        
        # Add bias to each query (bias is typically 1.0)
        bias = np.ones((all_queries.shape[0], 1))
        all_queries_with_bias = np.hstack([all_queries.reshape(-1, 4), bias])  # Reshape and add bias
        
        # Ensure the number of queries matches the number of connections in the substrate
        assert all_queries_with_bias.shape[1] == 5, \
            f"Query coordinates should have 5 elements (source_x, source_y, target_x, target_y, bias). Found {all_queries_with_bias.shape[1]} elements per query."
        assert all_queries_with_bias.shape[0] == self.conns.shape[0], \
            f"Mismatch between number of queries {all_queries_with_bias.shape[0]} and connections {self.conns.shape[0]}"
        
        return all_queries_with_bias



    @property
    def num_inputs(self):
        return self.inputs

    @property
    def num_outputs(self):
        return self.outputs

    @property
    def nodes_cnt(self):
        return self.nodes.shape[0]

    @property
    def conns_cnt(self):
        return self.conns.shape[0]

    def show_substrate(self):
        fig, ax = plt.subplots()

        # Scatter plot for the input nodes
        input_nodes = self.coors[:self.num_inputs]

        # For hidden layers, calculate the range for each hidden layer
        hidden_nodes = []
        hidden_layer_ranges = []
        current_index = self.num_inputs
        for layer in self.hidden_coordinates:
            hidden_nodes.append(self.coors[current_index:current_index + len(layer)])
            hidden_layer_ranges.append((current_index, current_index + len(layer)))
            current_index += len(layer)

        # Output nodes
        output_nodes = self.coors[self.nodes_cnt - self.num_outputs:]

        # Plot input nodes (blue)
        ax.scatter(input_nodes[:, 0], input_nodes[:, 1], color='blue', label='Input Nodes', s=100)

        # Colors for each hidden layer (cycle through if there are more hidden layers)
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
        
        # Plot hidden nodes (different color for each layer)
        for idx, layer_nodes in enumerate(hidden_nodes):
            color = colors[idx % len(colors)]  # Cycle through colors if necessary
            ax.scatter(layer_nodes[:, 0], layer_nodes[:, 1], label=f'Hidden Layer {idx + 1}', s=100, color=color)

        # Plot output nodes (red)
        ax.scatter(output_nodes[:, 0], output_nodes[:, 1], color='red', label='Output Nodes', s=100)

        # Plot connections as lines between nodes
        for conn in self.conns:
            src_idx, tgt_idx = conn[0], conn[1]
            src = self.coors[src_idx]
            tgt = self.coors[tgt_idx]
            ax.plot([src[0], tgt[0]], [src[1], tgt[1]], 'k-', lw=1)  # Black lines for connections

        # Label and show plot
        ax.legend()
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Substrate Visualization')
        plt.grid(True)
        plt.show()


