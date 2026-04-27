from logging import critical
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow.keras as K

class BipartiteGraphConvolution(K.Model):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(self, emb_size, activation, initializer, right_to_left=False):
        super().__init__()
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left

        self.built = False

        # feature layers
        self.feature_module_left = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=True, kernel_initializer=self.initializer)
        ])
        self.feature_module_edge = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer)
        ])
        self.feature_module_right = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer)
        ])
        self.feature_module_final = K.Sequential([
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer)
        ])

        # output_layers
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ])

    def build(self, input_shapes):

        if not self.built:

            l_shape, ei_shape, ev_shape, r_shape = input_shapes

            self.feature_module_left.build(l_shape)
            self.feature_module_edge.build(ev_shape)
            self.feature_module_right.build(r_shape)
            self.feature_module_final.build([None, self.emb_size])
            self.output_module.build([None, self.emb_size + (l_shape[1] if self.right_to_left else r_shape[1])])
            self.built = True

    def call(self, inputs, training):
        """
        Perfoms a partial graph convolution on the given bipartite graph.

        Inputs
        ------
        left_features: 2D float tensor
            Features of the left-hand-side nodes in the bipartite graph
        edge_indices: 2D int tensor
            Edge indices in left-right order
        edge_features: 2D float tensor
            Features of the edges
        right_features: 2D float tensor
            Features of the right-hand-side nodes in the bipartite graph
        scatter_out_size: 1D int tensor
            Output size (left_features.shape[0] or right_features.shape[0], unknown at compile time)

        Other parameters
        ----------------
        training: boolean
            Training mode indicator
        """
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left:
            scatter_dim = 0
            prev_features = left_features
        else:
            scatter_dim = 1
            prev_features = right_features

        # compute joint features
        joint_features = self.feature_module_final(
            tf.gather(
                self.feature_module_left(left_features),
                axis=0,
                indices=edge_indices[0]
            ) +
            self.feature_module_edge(edge_features) +
            tf.gather(
                self.feature_module_right(right_features),
                axis=0,
                indices=edge_indices[1])
        )

        # perform convolution
        conv_output = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
            shape=[scatter_out_size, self.emb_size]
        )

        neighbour_count = tf.scatter_nd(
            updates=tf.ones(shape=[tf.shape(edge_indices)[1], 1], dtype=tf.float32),
            indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
            shape=[scatter_out_size, 1])
        neighbour_count = tf.where(
            tf.equal(neighbour_count, 0),
            tf.ones_like(neighbour_count),
            neighbour_count)  # NaN safety trick
        conv_output = conv_output / neighbour_count  # mean convolution

        # apply final module
        output = self.output_module(tf.concat([
            conv_output,
            prev_features,
        ], axis=1))

        return output


class BipartiteGraphConvolution_attention(K.Model):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(self, emb_size, activation, initializer, right_to_left=False):
        super().__init__()
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left

        self.built = False

        self.feature_module_edge = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer)
        ])

        self.feature_module_final = K.Sequential([
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer)
        ])
        self.attention = K.Sequential([
            # K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
            K.layers.Dense(units=3, activation=None, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer),
            K.layers.Activation(K.activations.sigmoid),
        ])

        # output_layers
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ])

    def build(self, input_shapes):

        if not self.built:

            l_shape, ei_shape, ev_shape, r_shape = input_shapes

            # self.feature_module_left.build(l_shape)
            self.feature_module_edge.build(ev_shape)
            # self.feature_module_right.build(r_shape)
            self.feature_module_final.build([None, self.emb_size])
            self.attention.build([None, 3 * self.emb_size])
            self.output_module.build([None, self.emb_size + (l_shape[1] if self.right_to_left else r_shape[1])])
            self.built = True

    def call(self, inputs, training):
        """
        Perfoms a partial graph convolution on the given bipartite graph.

        Inputs
        ------
        left_features: 2D float tensor
            Features of the left-hand-side nodes in the bipartite graph
        edge_indices: 2D int tensor
            Edge indices in left-right order
        edge_features: 2D float tensor
            Features of the edges
        right_features: 2D float tensor
            Features of the right-hand-side nodes in the bipartite graph
        scatter_out_size: 1D int tensor
            Output size (left_features.shape[0] or right_features.shape[0], unknown at compile time)

        Other parameters
        ----------------
        training: boolean
            Training mode indicator
        """
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left:
            scatter_dim = 0
            prev_features = left_features

            # compute joint features, 不考虑target feature
            joint_features = self.feature_module_final(
                self.feature_module_edge(edge_features) +
                tf.gather(
                    right_features,
                    axis=0,
                    indices=edge_indices[1])
            )
        else:
            scatter_dim = 1
            prev_features = right_features

            # compute joint features
            joint_features = self.feature_module_final(
                tf.gather(
                    left_features,
                    axis=0,
                    indices=edge_indices[0]
                ) +
                self.feature_module_edge(edge_features)
            )
        
        attention_input = tf.concat([tf.gather(left_features, axis=0, indices=edge_indices[0]),
            self.feature_module_edge(edge_features),
            tf.gather(right_features, axis=0, indices=edge_indices[1])], axis=1)

        attention_parameter = self.attention(attention_input)

        joint_features = joint_features * attention_parameter

        # perform convolution
        conv_output = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
            shape=[scatter_out_size, self.emb_size]
        )

        neighbour_count = tf.scatter_nd(
            updates=tf.ones(shape=[tf.shape(edge_indices)[1], 1], dtype=tf.float32),
            indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
            shape=[scatter_out_size, 1])
        neighbour_count = tf.where(
            tf.equal(neighbour_count, 0),
            tf.ones_like(neighbour_count),
            neighbour_count)  # NaN safety trick
        conv_output = conv_output / neighbour_count  # mean convolution

        # apply final module
        output = self.output_module(tf.concat([
            conv_output,
            prev_features,
        ], axis=1))

        return output


class GCNPolicy(K.Model):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self, emb_size=16):
        super().__init__()

        self.emb_size = emb_size
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 14

        self.v_o_nfeats = 1
        self.c_o_nfeats = 1
        self.obj_nfeats = 10

        self.activation = K.activations.relu
        self.initializer = K.initializers.Orthogonal()
        self.built = False

        # CONSTRAINT EMBEDDING
        self.cons_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # VARIABLE EMBEDDING
        self.var_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # OBJECTIVE EMBEDDING
        self.obj_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # GRAPH CONVOLUTIONS
        # self.conv_v_to_o = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)
        # self.conv_o_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
        # self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
        
        # self.conv_c_to_o = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)
        # self.conv_o_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
        # self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)


        # GRAPH CONVOLUTIONS Attention
        self.conv_v_to_o = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer)
        self.conv_o_to_c = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer, right_to_left=True)
        self.conv_v_to_c = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer, right_to_left=True)
        
        self.conv_c_to_o = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer)
        self.conv_o_to_v = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer, right_to_left=True)
        self.conv_c_to_v = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer)


        # OUTPUT
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer, use_bias=False),
        ])

        # build model right-away
        self.build([
            (None, self.cons_nfeats),
            (2, None),
            (None, self.edge_nfeats),
            (None, self.var_nfeats),
            (None, ),
            (None, ),
            (None, self.obj_nfeats),
            (None, self.v_o_nfeats),
            (None, self.c_o_nfeats),
            (2, None),
            (2, None)
        ])



    def build(self, input_shapes):
        c_shape, ei_shape, ev_shape, v_shape, nc_shape, nv_shape, obj_shape, v_o_shape, c_o_shape, v_o_ind, c_o_ind = input_shapes
        emb_shape = [None, self.emb_size]

        if not self.built:
            self.cons_embedding.build(c_shape)
            # self.edge_embedding.build(ev_shape)
            self.var_embedding.build(v_shape)
            self.obj_embedding.build(obj_shape)

            self.conv_v_to_o.build((emb_shape, v_o_ind, v_o_shape, emb_shape))
            self.conv_o_to_c.build((emb_shape, c_o_ind, c_o_shape, emb_shape))
            self.conv_v_to_c.build((emb_shape, ei_shape, ev_shape, emb_shape))

            self.conv_c_to_o.build((emb_shape, c_o_ind, c_o_shape, emb_shape))
            self.conv_o_to_v.build((emb_shape, v_o_ind, v_o_shape, emb_shape))
            self.conv_c_to_v.build((emb_shape, ei_shape, ev_shape, emb_shape))

            self.output_module.build(emb_shape)
            self.built = True

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = tf.reduce_max(n_vars_per_sample)

        output = tf.split(
            value=output,
            num_or_size_splits=n_vars_per_sample,
            axis=1,
        )

        output = tf.concat([
            tf.pad(
                x,
                paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
                mode='CONSTANT',
                constant_values=pad_value)
            for x in output
        ], axis=0)

        return output

    def call(self, inputs, training=True):
        """
        Accepts stacked mini-batches, i.e. several bipartite graphs aggregated
        as one. In that case the number of variables per samples has to be
        provided, and the output consists in a padded dense tensor.

        Parameters
        ----------
        inputs: list of tensors
            Model input as a bipartite graph. May be batched into a stacked graph.

        Inputs
        ------
        constraint_features: 2D float tensor
            Constraint node features (n_constraints x n_constraint_features)
        edge_indices: 2D int tensor
            Edge constraint and variable indices (2, n_edges)
        edge_features: 2D float tensor
            Edge features (n_edges, n_edge_features)
        variable_features: 2D float tensor
            Variable node features (n_variables, n_variable_features)
        n_cons_per_sample: 1D int tensor
            Number of constraints for each of the samples stacked in the batch.
        n_vars_per_sample: 1D int tensor
            Number of variables for each of the samples stacked in the batch.

        Other parameters
        ----------------
        training: boolean
            Training mode indicator
        """
        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample, variable_objective_features, variable_objective_indices, constraint_objective_features, constraint_objective_indices, objective_features = inputs
        n_cons_total = tf.reduce_sum(n_cons_per_sample)
        n_vars_total = tf.reduce_sum(n_vars_per_sample)
        n_objs_total = tf.shape(n_cons_per_sample)[0]

        # EMBEDDINGS
        constraint_features = self.cons_embedding(constraint_features)
        # edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        objective_features = self.obj_embedding(objective_features)

        # GRAPH CONVOLUTIONS

        objective_features = self.conv_v_to_o((variable_features, variable_objective_indices, variable_objective_features, objective_features, n_objs_total), training)
        objective_features = self.activation(objective_features)
        constraint_features = self.conv_o_to_c((constraint_features, constraint_objective_indices, constraint_objective_features, objective_features, n_cons_total), training)
        constraint_features = self.activation(constraint_features)
        constraint_features = self.conv_v_to_c((
            constraint_features, edge_indices, edge_features, variable_features, n_cons_total), training)
        constraint_features = self.activation(constraint_features)


        objective_features = self.conv_c_to_o((constraint_features, constraint_objective_indices, constraint_objective_features, objective_features, n_objs_total), training)
        objective_features = self.activation(objective_features)

        variable_features = self.conv_o_to_v((variable_features, variable_objective_indices, variable_objective_features, objective_features, n_vars_total), training)
        variable_features = self.activation(variable_features)
        variable_features = self.conv_c_to_v((
            constraint_features, edge_indices, edge_features, variable_features, n_vars_total), training)
        variable_features = self.activation(variable_features)

        # OUTPUT
        output = self.output_module(variable_features)
        output = tf.sigmoid(output)
        output = tf.reshape(output, [1, -1])

        if n_vars_per_sample.shape[0] > 1:
            output = self.pad_output(output, n_vars_per_sample)

        return output


class GCNPolicy_critic(K.Model):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self, reward_length, emb_size=16):
        super().__init__()

        self.reward_length = reward_length

        self.emb_size = emb_size
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 14

        self.v_o_nfeats = 1
        self.c_o_nfeats = 1
        self.obj_nfeats = 10

        self.activation = K.activations.relu
        self.initializer = K.initializers.Orthogonal()
        self.built = False

        # CONSTRAINT EMBEDDING
        self.cons_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # VARIABLE EMBEDDING
        self.var_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # OBJECTIVE EMBEDDING
        self.obj_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # GRAPH CONVOLUTIONS
        # self.conv_v_to_o = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)
        # self.conv_o_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
        # self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
        
        # self.conv_c_to_o = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)
        # self.conv_o_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
        # self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)

        # GRAPH CONVOLUTIONS Attention
        self.conv_v_to_o = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer)
        self.conv_o_to_c = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer, right_to_left=True)
        self.conv_v_to_c = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer, right_to_left=True)
        
        self.conv_c_to_o = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer)
        self.conv_o_to_v = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer, right_to_left=True)
        self.conv_c_to_v = BipartiteGraphConvolution_attention(self.emb_size, self.activation, self.initializer)


        self.critical_output = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.reward_length, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # OUTPUT
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer, use_bias=False),
        ])

        # build model right-away
        self.build([
            (None, self.cons_nfeats),
            (2, None),
            (None, self.edge_nfeats),
            (None, self.var_nfeats),
            (None, ),
            (None, ),
            (None, 2),  # concat action and variable output
            (None, self.obj_nfeats),
            (None, self.v_o_nfeats),
            (None, self.c_o_nfeats),
            (2, None),
            (2, None)
        ])


    def build(self, input_shapes):
        c_shape, ei_shape, ev_shape, v_shape, nc_shape, nv_shape, critical_shape, obj_shape, v_o_shape, c_o_shape, v_o_ind, c_o_ind = input_shapes
        emb_shape = [None, self.emb_size]

        if not self.built:
            self.cons_embedding.build(c_shape)
            self.var_embedding.build(v_shape)
            self.obj_embedding.build(obj_shape)

            self.conv_v_to_o.build((emb_shape, v_o_ind, v_o_shape, emb_shape))
            self.conv_o_to_c.build((emb_shape, c_o_ind, c_o_shape, emb_shape))
            self.conv_v_to_c.build((emb_shape, ei_shape, ev_shape, emb_shape))

            self.conv_c_to_o.build((emb_shape, c_o_ind, c_o_shape, emb_shape))
            self.conv_o_to_v.build((emb_shape, v_o_ind, v_o_shape, emb_shape))
            self.conv_c_to_v.build((emb_shape, ei_shape, ev_shape, emb_shape))

            self.output_module.build(emb_shape)
            self.critical_output.build(critical_shape)
            self.built = True

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = tf.reduce_max(n_vars_per_sample)

        output = tf.split(
            value=output,
            num_or_size_splits=n_vars_per_sample,
            axis=1,
        )
        output = tf.concat([
            tf.pad(
                x,
                paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
                mode='CONSTANT',
                constant_values=pad_value)
            for x in output
        ], axis=0)

        return output

    def call(self, inputs, actions, training=True):  # critic net需要增加action作为输入
        """
        Accepts stacked mini-batches, i.e. several bipartite graphs aggregated
        as one. In that case the number of variables per samples has to be
        provided, and the output consists in a padded dense tensor.

        Parameters
        ----------
        inputs: list of tensors
            Model input as a bipartite graph. May be batched into a stacked graph.

        Inputs
        ------
        constraint_features: 2D float tensor
            Constraint node features (n_constraints x n_constraint_features)
        edge_indices: 2D int tensor
            Edge constraint and variable indices (2, n_edges)
        edge_features: 2D float tensor
            Edge features (n_edges, n_edge_features)
        variable_features: 2D float tensor
            Variable node features (n_variables, n_variable_features)
        n_cons_per_sample: 1D int tensor
            Number of constraints for each of the samples stacked in the batch.
        n_vars_per_sample: 1D int tensor
            Number of variables for each of the samples stacked in the batch.

        Other parameters
        ----------------
        training: boolean
            Training mode indicator
        """
        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample, variable_objective_features, variable_objective_indices, constraint_objective_features, constraint_objective_indices, objective_features = inputs
        n_cons_total = tf.reduce_sum(n_cons_per_sample)
        n_vars_total = tf.reduce_sum(n_vars_per_sample)
        n_objs_total = tf.shape(n_cons_per_sample)[0]

        # EMBEDDINGS
        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)
        objective_features = self.obj_embedding(objective_features)

        # GRAPH CONVOLUTIONS

        objective_features = self.conv_v_to_o((variable_features, variable_objective_indices, variable_objective_features, objective_features, n_objs_total), training)
        objective_features = self.activation(objective_features)
        constraint_features = self.conv_o_to_c((constraint_features, constraint_objective_indices, constraint_objective_features, objective_features, n_cons_total), training)
        constraint_features = self.activation(constraint_features)
        constraint_features = self.conv_v_to_c((
            constraint_features, edge_indices, edge_features, variable_features, n_cons_total), training)
        constraint_features = self.activation(constraint_features)


        objective_features = self.conv_c_to_o((constraint_features, constraint_objective_indices, constraint_objective_features, objective_features, n_objs_total), training)
        objective_features = self.activation(objective_features)

        variable_features = self.conv_o_to_v((variable_features, variable_objective_indices, variable_objective_features, objective_features, n_vars_total), training)
        variable_features = self.activation(variable_features)
        variable_features = self.conv_c_to_v((
            constraint_features, edge_indices, edge_features, variable_features, n_vars_total), training)
        variable_features = self.activation(variable_features)

        # OUTPUT
        output = self.output_module(variable_features)
        output = tf.reshape(output, [-1, 1])
        actions = tf.reshape(actions, [-1, 1])

        if n_vars_per_sample.shape[0] > 1:
            output = self.pad_output(output, n_vars_per_sample)
        
        output = tf.concat((output, actions), axis=1)  # concat variable output and actions
        critical_output = self.critical_output(output)
        critical_output = tf.tanh(critical_output)
        
        output = tf.reduce_mean(critical_output, axis=0)
        output = tf.reshape(output, [-1, 1])

        return output



class MLPPolicy(K.Model):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self, emb_size=16):
        super().__init__()

        self.emb_size = emb_size
        self.var_nfeats = 20

        self.activation = K.activations.relu
        self.initializer = K.initializers.Orthogonal()
        self.built = False

        # VARIABLE EMBEDDING
        self.var_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])
        
        # OUTPUT
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer, use_bias=False),
        ])

        # build model right-away
        self.build([
            (None, self.var_nfeats)
        ])


    def build(self, input_shapes):
        v_shape = input_shapes[0]
        emb_shape = [None, self.emb_size]

        if not self.built:
            self.var_embedding.build(v_shape)

            self.output_module.build(emb_shape)
            self.built = True


    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = tf.reduce_max(n_vars_per_sample)

        output = tf.split(
            value=output,
            num_or_size_splits=n_vars_per_sample,
            axis=1,
        )

        output = tf.concat([
            tf.pad(
                x,
                paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
                mode='CONSTANT',
                constant_values=pad_value)
            for x in output
        ], axis=0)

        return output

    def call(self, inputs, training=True):

        variable_features, n_vars_per_sample = inputs
        n_vars_total = tf.reduce_sum(n_vars_per_sample)

        # EMBEDDINGS
        variable_features = self.var_embedding(variable_features)

        # OUTPUT
        output = self.output_module(variable_features)
        output = tf.sigmoid(output)
        output = tf.reshape(output, [1, -1])

        # if n_vars_per_sample.shape[0] > 1:
        #     output = self.pad_output(output, n_vars_per_sample)

        return output


class MLPPolicy_critic(K.Model):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self, reward_length, emb_size=16):
        super().__init__()

        self.reward_length = reward_length

        self.emb_size = emb_size
        self.var_nfeats = 20

        self.activation = K.activations.relu
        self.initializer = K.initializers.Orthogonal()
        self.built = False

        # VARIABLE EMBEDDING
        self.var_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        self.critical_output = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.reward_length, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # OUTPUT
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer, use_bias=False),
        ])

        # build model right-away
        self.build([
            (None, self.var_nfeats),
            (None, 2),  # concat action and variable output
        ])


    def build(self, input_shapes):
        v_shape, critical_shape = input_shapes
        emb_shape = [None, self.emb_size]

        if not self.built:
            self.var_embedding.build(v_shape)

            self.output_module.build(emb_shape)
            self.critical_output.build(critical_shape)
            self.built = True

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = tf.reduce_max(n_vars_per_sample)

        output = tf.split(
            value=output,
            num_or_size_splits=n_vars_per_sample,
            axis=1,
        )
        output = tf.concat([
            tf.pad(
                x,
                paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
                mode='CONSTANT',
                constant_values=pad_value)
            for x in output
        ], axis=0)

        return output

    def call(self, inputs, actions, training=True):  # critic net需要增加action作为输入
        """
        Accepts stacked mini-batches, i.e. several bipartite graphs aggregated
        as one. In that case the number of variables per samples has to be
        provided, and the output consists in a padded dense tensor.

        Parameters
        ----------
        inputs: list of tensors
            Model input as a bipartite graph. May be batched into a stacked graph.

        Inputs
        ------
        constraint_features: 2D float tensor
            Constraint node features (n_constraints x n_constraint_features)
        edge_indices: 2D int tensor
            Edge constraint and variable indices (2, n_edges)
        edge_features: 2D float tensor
            Edge features (n_edges, n_edge_features)
        variable_features: 2D float tensor
            Variable node features (n_variables, n_variable_features)
        n_cons_per_sample: 1D int tensor
            Number of constraints for each of the samples stacked in the batch.
        n_vars_per_sample: 1D int tensor
            Number of variables for each of the samples stacked in the batch.

        Other parameters
        ----------------
        training: boolean
            Training mode indicator
        """
        variable_features, n_vars_per_sample = inputs
        n_vars_total = tf.reduce_sum(n_vars_per_sample)

        # EMBEDDINGS
        variable_features = self.var_embedding(variable_features)

        # OUTPUT
        output = self.output_module(variable_features)
        output = tf.reshape(output, [-1, 1])
        actions = tf.reshape(actions, [-1, 1])

        # if n_vars_per_sample.shape[0] > 1:
        #     output = self.pad_output(output, n_vars_per_sample)
        
        output = tf.concat((output, actions), axis=1)  # concat variable output and actions
        critical_output = self.critical_output(output)
        critical_output = tf.tanh(critical_output)
        
        output = tf.reduce_mean(critical_output, axis=0)
        output = tf.reshape(output, [-1, 1])

        return output