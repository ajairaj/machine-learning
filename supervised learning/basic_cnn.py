import tensorflow as tf

def basic_cnn(inputs, targets):
    with tf.variable_scope('conv_net', reuse=False):
        conv_a = tf.layers.conv2d(inputs = inputs,
                                  filters = 16, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_a')
        conv_b = tf.layers.conv2d(inputs = conv_a,
                                  filters = 32, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_b')
        conv_c = tf.layers.conv2d(inputs = conv_b,
                                  filters = 64, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_c')        
        x = conv_c.shape[1].value
        y = conv_c.shape[2].value
        c = conv_c.shape[3].value
        flattened = tf.reshape(conv_c, [-1, x * y * c], name='flatten')
        dense_1 = tf.layers.dense(inputs = flattened, 
                                   name = 'dense_1', 
                                   units = 64, 
                                   activation = tf.nn.leaky_relu,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer())
        bnorm_1 = tf.layers.batch_normalization(dense_1, name = "bnorm_1")
        
        logits = tf.layers.dense(inputs = bnorm_1, 
                                 name = 'logits', 
                                 units = targets.shape[1], 
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())
        softmax = tf.nn.softmax(logits, name = 'softmax')

    with tf.variable_scope('cost', reuse = False):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = targets, logits = logits)
        cost = tf.reduce_mean(loss)
        
    return softmax, cost

def basic_cnn_2(inputs, targets):
    with tf.variable_scope('conv_net', reuse=False):
        conv_a = tf.layers.conv2d(inputs = inputs,
                                  filters = 16, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_a')
        drop_a = tf.nn.dropout(conv_a, name='drop_a')
        conv_b = tf.layers.conv2d(inputs = bnorm_a,
                                  filters = 32, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_b')
        drop_b = tf.nn.dropout(conv_b, name='drop_b')
        conv_c = tf.layers.conv2d(inputs = bnorm_b,
                                  filters = 64, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_c')
        drop_c = tf.nn.dropout(conv_c, name='drop_c')
        
        x = drop_c.shape[1].value
        y = drop_c.shape[2].value
        c = drop_c.shape[3].value
        flattened = tf.reshape(drop_c, [-1, x * y * c], name='flatten')
        dense_1 = tf.layers.dense(inputs = flattened, 
                                   name = 'dense_1', 
                                   units = 64, 
                                   activation = tf.nn.leaky_relu,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer())
        drop_1 = tf.layers.batch_normalization(dense_1, name = "drop_1")
        
        logits = tf.layers.dense(inputs = drop_1, 
                                 name = 'logits', 
                                 units = targets.shape[1], 
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())
        softmax = tf.nn.softmax(logits, name = 'softmax')

    with tf.variable_scope('cost', reuse = False):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = targets, logits = logits)
        cost = tf.reduce_mean(loss)
    
    return softmax, cost

def basic_cnn_3(inputs, targets):
    with tf.variable_scope('conv_net', reuse=False):
        conv_a = tf.layers.conv2d(inputs = inputs,
                                  filters = 16, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_a')
        bnorm_a = tf.layers.batch_normalization(conv_a, name='bnorm_a')
        conv_b = tf.layers.conv2d(inputs = bnorm_a,
                                  filters = 32, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_b')
        bnorm_b = tf.layers.batch_normalization(conv_b, name='bnorm_b')
        conv_c = tf.layers.conv2d(inputs = bnorm_b,
                                  filters = 64, 
                                  kernel_size = (3,3), 
                                  padding = 'same', 
                                  activation = tf.nn.leaky_relu,
                                  strides = 2,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name = 'conv_c')
        bnorm_c = tf.layers.batch_normalization(conv_c, name='bnorm_c')
        
        x = bnorm_c.shape[1].value
        y = bnorm_c.shape[2].value
        c = bnorm_c.shape[3].value
        flattened = tf.reshape(bnorm_c, [-1, x * y * c], name='flatten')
        dense_1 = tf.layers.dense(inputs = flattened, 
                                   name = 'dense_1', 
                                   units = 64, 
                                   activation = tf.nn.leaky_relu,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer())
        bnorm_1 = tf.layers.batch_normalization(dense_1, name = "bnorm_1")
        
        logits = tf.layers.dense(inputs = bnorm_1, 
                                 name = 'logits', 
                                 units = targets.shape[1], 
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())
        softmax = tf.nn.softmax(logits, name = 'softmax')

    with tf.variable_scope('cost', reuse = False):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = targets, logits = logits)
        cost = tf.reduce_mean(loss)
    
    return softmax, cost