class ESSubstrate(BaseSubstrate):
    def __init__(self, num_inputs, num_outputs, coors, nodes, conns, band_threshold=0.1, variance_threshold=0.2):
        self.inputs = num_inputs
        self.outputs = num_outputs
        self.coors = np.array(coors)
        self.nodes = np.array(nodes)
        self.conns = np.array(conns)
        self.band_threshold = band_threshold  # Threshold for connection pruning
        self.variance_threshold = variance_threshold  # Threshold for recursive division

    def make_nodes(self, query_res):
        """
        Return nodes based on the current query results.
        """
        return self.nodes

    def make_conns(self, query_res):
        """
        Set connections based on CPPN output, applying band threshold pruning.
        """
        conns_with_attrs = vmap(set_conn_attrs)(self.conns, query_res)
        pruned_conns = self.prune_conns(conns_with_attrs)
        return pruned_conns

    def prune_conns(self, conns):
        """
        Prune connections based on the band threshold (ES-HyperNEAT feature).
        """
        # Prune connections with weights below the band threshold
        return conns[conns[:, 2] > self.band_threshold]  # Assuming weight is in column 2

    def recursive_division(self, cppn_output, region):
        """
        Recursively divide substrate region based on variance threshold (quadtree-like).
        """
        variance = np.var(cppn_output)
        if variance > self.variance_threshold:
            # Subdivide the region further
            # Example: Split into quadrants or more subdivisions based on needs
            subregions = self.divide_region(region)
            for subregion in subregions:
                self.recursive_division(cppn_output, subregion)

    def divide_region(self, region):
        """
        Divide the current region into smaller subregions (quadtree-like).
        """
        # Example: Divide a 2D region into 4 quadrants
        x_min, x_max, y_min, y_max = region
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2

        subregions = [
            (x_min, mid_x, y_min, mid_y),
            (mid_x, x_max, y_min, mid_y),
            (x_min, mid_x, mid_y, y_max),
            (mid_x, x_max, mid_y, y_max),
        ]
        return subregions

    @property
    def query_coors(self):
        return self.coors

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