import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import layers

"Fi-GNN"

class CTR_ggnn(object):
    def __init__(self, config, device, loader, mode):
        self.config = config
        self.mode = mode
        if mode == "Train":
            self.is_training = False
            self.batch_size = self.config.train_batch_size
            self.maxstep_size = self.config.train_step_size
            reuse = None
        elif mode == "Valid":
            self.is_training = False
            self.batch_size = self.config.valid_batch_size
            self.maxstep_size = self.config.valid_step_size
            reuse = True
        else:
            self.is_training = False
            self.batch_size = self.config.test_batch_size
            self.maxstep_size = self.config.test_step_size
            reuse = True

        self.hidden_size = hidden_size = config.hidden_size  # hidden_size = 16
        self.GNN_step = GNN_step = config.GNN_step
        self.node_num = node_num = config.node_num

        opt = config.sgd_opt
        beta = config.beta
        batch_size = self.batch_size

        hidden_stdv = np.sqrt(1. / (hidden_size))
        # embedding initial
        with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):
            feature_embedding_list = []
            for idx, each_feature_num in enumerate(config.feature_num): # # config.feature_num= [3000,3000,3000....3000] 22个特征
                feature_embedding_list.append(  # 循环22次[matrix_id,   matrix_click]， matrix_id= 3000 *16

                    tf.get_variable(name='feature_embedding_' + str(idx), shape=[each_feature_num, hidden_size], # 每个特征的样本个数 * 16 # [[16维],.... [16维]]一共有3000个
                                    initializer=tf.random_normal_initializer(hidden_stdv))
                )

            w_attention = tf.get_variable(
                name='w_attetion',
                shape=[hidden_size * len(config.feature_num), len(config.feature_num)],
                initializer=tf.random_normal_initializer(stddev=0.2)
            )
            w_score2 = tf.get_variable(
                name='w_score_2', shape=[hidden_size, 1],
                initializer=tf.random_normal_initializer(hidden_stdv)
            )

        # #------------feed-----------------##
        # config.feature_num= [3000,3000,3000, ....3000] 22个特征
        self.input_x = input_x = tf.placeholder(tf.int32, [batch_size, len(config.feature_num)]) # 占位符input的shape是20(batch_size) * 2
        self.input_y = input_y = tf.placeholder(tf.int32, [batch_size, 1])

        input_x_unstack = tf.unstack(tf.transpose(input_x, (1, 0)), axis=0) # 将
        feature_embedding_input = []
        # translate into embedding (lookup)
        for idx in range(len(input_x_unstack)): # range(2)
            feature_embedding_input.append(
                tf.nn.embedding_lookup(feature_embedding_list[idx], input_x_unstack[idx])
            )

        self.feature_input = tf.transpose(tf.stack(feature_embedding_input, axis=0), (1, 0 ,2)) # input_feature

        with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):

            self.w_A = self.weights('a_value', hidden_size, 0) # 生成weight参数 32*1, shape=[2*hidden_size, 1],
            self.b_A = self.biases('a_value', hidden_size, 0) # bias 1

            graph = self.init_graph(config, self.feature_input)

            final_state, test1 = self.GNN(
                self.feature_input, batch_size, hidden_size, GNN_step, len(config.feature_num), graph
            )

            atten_pos = self.attention_layer(
                final_state, w_attention, batch_size, hidden_size, len(config.feature_num)
            )
            score_pos = tf.matmul(tf.reshape(final_state, [-1, hidden_size]), w_score2)
            score_pos = tf.maximum(0.01 * score_pos, score_pos)
            score_pos = tf.reshape(score_pos, [batch_size, len(config.feature_num)])
            s_pos = tf.reshape(tf.reduce_sum(score_pos * atten_pos, axis=1), [batch_size, 1])


            self.predict = predict = tf.sigmoid(s_pos)
        # -------------evaluation--------------
        self.auc_result, self.auc_opt = tf.metrics.auc(
            labels=self.input_y,
            predictions=predict
        )
        self.s_pos = s_pos
        # -------------cost ---------------
        cost_parameter = 0.
        num_parameter = 0.

        score_mean = tf.losses.log_loss(
            labels=self.input_y,
            predictions=predict
        )
        self.cost = cost = score_mean + cost_parameter

        # ---------------optimizer---------------#
        self.no_opt = tf.no_op()
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False)

        if mode == 'Train':
            self.auc_opt = tf.no_op()
            self.auc_result = tf.no_op()
            if opt == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
            if opt == 'Momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(cost)
            if opt == 'RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)
            if opt == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cost)

        else:
            self.optimizer = tf.no_op()

    def weights(self, name, hidden_size, i):
        image_stdv = np.sqrt(1. / (2048))
        hidden_stdv = np.sqrt(1. / (hidden_size))
        if name == 'in_image':
            w = tf.get_variable(name='w/in_image_'+ str(i),
                                shape=[2048, hidden_size],
                                initializer=tf.random_normal_initializer(stddev=image_stdv))
        if name == 'out_image':
            w = tf.get_variable(name='w/out_image_' + str(i),
                                shape=[hidden_size, 2048],
                                initializer=tf.random_normal_initializer(stddev=image_stdv))
        if name == 'hidden_state':
            if i > 0:
                with tf.variable_scope("w", reuse=True): # 变量共享
                    w = tf.get_variable(name='hidden_state',
                                        shape=[hidden_size, hidden_size],
                                        )
            else:
                with tf.variable_scope("w"):
                    w = tf.get_variable(name='hidden_state',
                                        shape=[hidden_size, hidden_size],
                                        )
        if name == 'hidden_state_in':
            w = tf.get_variable(
                name='w/hidden_state_in_' + str(i),
                shape=[hidden_size, hidden_size],
            )

        if name == 'hidden_state_out':
            w = tf.get_variable(
                name='w/hidden_state_out_' + str(i),
                shape=[hidden_size, hidden_size],
            )

        if name == 'a_value':
            w = tf.get_variable(
                name='w/a_value_' + str(i),
                shape=[2*hidden_size, 1], # [32, 1]
            )

        return w

    def biases(self, name, hidden_size, i):
        image_stdv = np.sqrt(1. / (2048))
        hidden_stdv = np.sqrt(1. / (hidden_size))

        if name == 'hidden_state':
            if i > 0:
                with tf.variable_scope("b", reuse=True):
                    b = tf.get_variable(name='hidden_state', shape=[hidden_size],
                                        initializer=tf.random_normal_initializer(stddev=hidden_stdv)
                                        )

            else:
                with tf.variable_scope("b"):
                    b = tf.get_variable(name='hidden_state', shape=[hidden_size],
                                        initializer=tf.random_normal_initializer(stddev=hidden_stdv)
                                        )
        if name == 'hidden_state_in':
            b = tf.get_variable(
                name='b/hidden_state_in_' + str(i),
                shape=(hidden_size, ),
            )

        if name == 'hidden_state_out':
            b = tf.get_variable(
                name='b/hidden_state_out_' + str(i),
                shape=(hidden_size, ),
            )

        if name == 'a_value':
            b = tf.get_variable(
                name='b/a_value_' + str(i),
                shape=(1, ),
            )

        return b

    def message_pass(self, x, hidden_size, batch_size, num_category, graph):  # x = [batch_size, feature_num, hidden_size]

        w_hidden_state = self.weights('hidden_state_out', hidden_size, 0)  # 16 * 16， 0表示当前是第0层，不reuse
        b_hidden_state = self.biases('hidden_state_out', hidden_size, 0)   # (16,)
        # 1
        x_all = tf.reshape(tf.matmul(     # x_all是第1个特征 * weight + 之后的结果，batch_size * 16
                                    tf.reshape(x[:, 0, :], [batch_size, hidden_size]), # batch中第一个特征：batch_size * 16
                                    w_hidden_state)
                            + b_hidden_state,
                [batch_size, hidden_size])
        # x_all = W_{out} * h
        for idx_feature in range(1, num_category): # num_category = feature_num = 22
            w_hidden_state = self.weights('hidden_state_out', hidden_size, idx_feature)
            b_hidden_state = self.biases('hidden_state_out', hidden_size, idx_feature)
            x_all_idx = tf.reshape(tf.matmul(
                                            tf.reshape(x[:, idx_feature, :], [batch_size, hidden_size]),
                                            w_hidden_state)
                               + b_hidden_state,
                    [batch_size, hidden_size])
            x_all = tf.concat([x_all, x_all_idx], 1) # cat之前shape：batch_size * 16, 将新的特征按列拼接在原来的后面
        #  x_all = W_{out} * h * A
        x_all = tf.reshape(x_all, [batch_size, num_category, hidden_size]) # batch_size * 22 * 16
        x_all = tf.transpose(x_all, (0, 2, 1))  # [batch_size, hidden_size, num_category]

        # 2. x_ = [hidden_size, num_category]
        x = x_all[0]       # batch_size的第一条数据 16 * 22
        graph_ = graph[0]   # 22 * 22
        x = tf.matmul(x, graph_) # 16 * 22 拿到了带有weight的
        for idx_sample in range(1, batch_size):
            x_i = x_all[idx_sample]
            graph_ = graph[idx_sample]
            x_i = tf.matmul(x_i, graph_)
            x = tf.concat([x, x_i], 0)
        x = tf.reshape(x, [batch_size, hidden_size, num_category])  # x = [batch_size, hidden_size, num_category]
        x = tf.transpose(x, (0, 2, 1))   # [batch_size, num_category, hidden_size]

        # 3.
        b_hidden_state = self.biases('hidden_state_in', hidden_size, 0)
        x = tf.reshape( tf.matmul(x[:, 0, :], self.weights('hidden_state_in', hidden_size, 0)) + b_hidden_state, # 拿第一个feature， weights = 16 * 16
            [batch_size, hidden_size]
        )
        for idx_feature in range(1, num_category):
            b_hidden_state = self.biases('hidden_state_in', hidden_size, idx_feature)
            i_x = tf.reshape(
                tf.matmul(x[:, idx_feature, :], self.weights('hidden_state_in', hidden_size, idx_feature)) + b_hidden_state,
                [batch_size, hidden_size]
            )
            x = tf.concat([x, i_x], 1)
        x = tf.reshape(x, [batch_size, num_category, hidden_size])

        return x

    def GNN(self, feature_embedding_input, batch_size, hidden_size, n_steps, feature_num, graph): # hidden_size是每个特征的维度
        h0 = feature_embedding_input

        h0 = tf.reshape(h0, [batch_size, feature_num, hidden_size])
  
        state = h0

        with tf.variable_scope("gnn"):
            for step in range(n_steps):  # GNN的层数3， n_steps == 3
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
    
                x = self.message_pass(state, hidden_size, batch_size, feature_num, graph)

                state_new = x[0] + state[0]
                state_new = tf.transpose(state_new, (1,0))

                for i in range(1, batch_size):
                    state_ = x[i] + state[i]  ##input of GRUCell must be 2 rank, not 3 rank
                    state_ = tf.transpose(state_, (1,0))
                    state_new = tf.concat([state_new, state_], 0)
                #x = tf.reshape(x, [batch_size, num_category, hidden_size])
                state_new = tf.reshape(state_new, [batch_size, feature_num, hidden_size])  ##restore: 2 rank to 3 rank

                #Residual
                state = h0 + state_new # 数值相加

        return state, h0






    def attention_layer(self, state, w_attention, batch_size, hidden_size, num_category):
        flat_state = tf.reshape(state, shape=(batch_size, hidden_size * num_category))
        return tf.sigmoid(tf.matmul(flat_state, w_attention))

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.learning_rate, learning_rate))

    def dense_layer(self, state, batch_size, hidden_size, num_category):

        flat_state = tf.reshape(state, shape=(batch_size, hidden_size * num_category))

        output = tf.contrib.layers.fully_connected(
            inputs=flat_state,
            num_outputs=400
        )
        output = tf.contrib.layers.fully_connected(
            inputs=output,
            num_outputs=1,
            activation_fn=None
        )
        return output

    def init_graph(self, config, x):
        node_num = len(config.feature_num) # 节点数24 == 特征数
        hidden_size = self.hidden_size # 16
        graph = []

        for i in range(self.batch_size):   # x = [batch_size, feature_num, hidden_size]
            a = tf.tile(x[i], [node_num, 1])  # x[i] = [feature_num, hidden_size], a = 484 * 16
            b = tf.tile(x[i], [1, node_num])  # x[i] = [feature_num, hidden_size], b = 22 * 352                     (352 = 16 * 22)

            a = tf.reshape(a, [node_num, node_num, hidden_size])
            b = tf.reshape(b, [node_num, node_num, hidden_size])

            c = tf.concat([a,b], 2)
            c = tf.reshape(c, [node_num * node_num, hidden_size * 2]) # 484 * 32

            value = tf.reshape(tf.matmul(c, self.w_A) + self.b_A, [node_num, node_num]) # 484 * 32 , 32 * 1 == 484 * 1
            value = value * (tf.ones((node_num, node_num)) - tf.eye(node_num)) # 22 * 22的matrix， * 对角线为0，其他值为1的矩阵

            value = tf.maximum(0.01 * value, value)  # leaky relu,

            value = tf.nn.softmax(value, 0) # 在第0个维度上做softmax # 22 * 22的tensor
            graph.append(value)

        graph = tf.reshape(graph, [self.batch_size, node_num, node_num]) # 每一条数据做一个，22个node的全连接图

        return graph



 