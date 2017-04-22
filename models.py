def make_conv2d_params(kernel_size, input_channel, output_channel, mu=0, sigma=0.1):
    weights = tf.Variable(
        tf.truncated_normal(shape=(kernel_size, kernel_size, input_channel, output_channel),
                            mean=mu, stddev=sigma))
    biases = tf.Variable(tf.zeros(output_channel))
    return weights, biases

def conv2d(x, weights, biases, stride, padding='SAME'):
    conv1 = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding=padding) + biases
    conv1 = tf.nn.relu(conv1)
    return conv1

def fc(inputs, num_input, num_output, mu=0, sigma=0.1):
    w = tf.Variable(tf.truncated_normal(shape=(num_input, num_output), mean=mu, stddev=sigma))
    b = tf.Variable(tf.zeros(num_output))
    fc1 = tf.matmul(inputs, w) + b

    return tf.nn.relu(fc1)

def squeeze(inputs, input_channel, output_channel):
    w, b = make_conv2d_params(1, input_channel, output_channel)
    return conv2d(inputs, w, b, stride=1)

def expand(inputs, input_channel, output_channel):
    w1x1, b1x1 = make_conv2d_params(1, input_channel, output_channel//2)
    w3x3, b3x3 = make_conv2d_params(3, input_channel, output_channel//2)
    e1x1 = conv2d(inputs, w1x1, b1x1, stride=1)
    e3x3 = conv2d(inputs, w3x3, b3x3, stride=1)
    return tf.concat([e1x1, e3x3], 3)

def MiniSqueeze(x, drop_rate):
    c1_w, c1_b = make_conv2d_params(1, 3, 30)
    c1 = conv2d(x, c1_w, c1_b, 1)

    s1 = squeeze(c1, 30, 10)
    e1 = expand(s1, 10, 40)

    s2 = squeeze(e1, 40, 10)
    e2 = expand(s2, 10, 40)

    p1 = tf.nn.max_pool(e2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    s3 = squeeze(p1, 40, 5)
    e3 = expand(s3, 5, 40)

    s4 = squeeze(e3, 40, 5)
    e4 = expand(s4, 5, 40)

    p2 = tf.nn.max_pool(e4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    c2_w, c2_b = make_conv2d_params(1, 40, 43)
    c2 = conv2d(p2, c2_w, c2_b, 2)

    d1 = tf.nn.dropout(c2, drop_rate)

    c3_w, c3_b = make_conv2d_params(4, 43, 43)
    c3 = conv2d(d1, c3_w, c3_b, 4)
    return tf.contrib.layers.flatten(c3), "mini_squeeze"

def MiniSqueezeWithFCv2(x):
    c1_w, c1_b = make_conv2d_params(5, 3, 30)
    c1 = conv2d(x, c1_w, c1_b, 1)

    s1 = squeeze(c1, 30, 10)
    e1 = expand(s1, 10, 40)

    #s2 = squeeze(e1, 40, 10)
    #e2 = expand(s2, 10, 40)

    p1 = tf.nn.max_pool(e1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    s3 = squeeze(p1, 40, 5)
    e3 = expand(s3, 5, 20)

    #s4 = squeeze(e3, 40, 5)
    #e4 = expand(s4, 5, 40)

    p2 = tf.nn.max_pool(e3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    f1 = tf.contrib.layers.flatten(p2)

    fc1 = fc(f1, 1280, 256)
    fc2 = fc(fc1, 256, 43)
    return fc2, "mini_squeeze_fc_v2"

def MultiScaleLeNet(x):
    mu = 0
    sigma = 0.1

    c1_w, c1_b = make_conv2d_params(5, 3, 6)
    c1 = conv2d(x, c1_w, c1_b, 1, padding='VALID')

    p1 = tf.nn.max_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    c2_w, c2_b = make_conv2d_params(5, 6, 16)
    c2 = conv2d(p1, c2_w, c2_b, 1, padding='VALID')

    p2 = tf.nn.max_pool(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    f1 = tf.contrib.layers.flatten(p1)
    f2 = tf.contrib.layers.flatten(p2)

    fc0 = tf.concat([f1, f2], axis=1)

    fc1_W = tf.Variable(tf.truncated_normal(shape=(1576, 168), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(168))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(168, 86), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(86))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    fc3_W = tf.Variable(tf.truncated_normal(shape=(86, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits, "MsLeNet"

def LeNet(x):
    mu = 0
    sigma = 0.1

    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0 = tf.contrib.layers.flatten(conv2)

    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits, "LeNet"

import tensorflow as tf

def incept_a(inputs):
    in1_w, in1_b = make_conv2d_params(1, 192, 64)
    in1 = conv2d(inputs, in1_w, in1_b, 1)

    in2r_w, in2r_b = make_conv2d_params(1, 192, 96)
    in2r = conv2d(inputs, in2r_w, in2r_b, 1)

    in2_w, in2_b = make_conv2d_params(3, 96, 128)
    in2 = conv2d(in2r, in2_w, in2_b, 1)

    in3r_w, in3r_b = make_conv2d_params(1, 192, 16)
    in3r = conv2d(inputs, in3r_w, in3r_b, 1)

    in3_w, in3_b = make_conv2d_params(1, 16, 32)
    in3 = conv2d(in3r, in3_w, in3_b, 1)

    inpr = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')

    inp_w, inp_b = make_conv2d_params(1, 192, 32)
    inp = conv2d(inpr, inp_w, inp_b, 1)

    return tf.concat([in1, in2, in3, inp], axis=3)

def incept_b(inputs):
    in1_w, in1_b = make_conv2d_params(1, 256, 128)
    in1 = conv2d(inputs, in1_w, in1_b, 1)

    in2r_w, in2r_b = make_conv2d_params(1, 256, 128)
    in2r = conv2d(inputs, in2r_w, in2r_b, 1)

    in2_w, in2_b = make_conv2d_params(3, 128, 192)
    in2 = conv2d(in2r, in2_w, in2_b, 1)

    in3r_w, in3r_b = make_conv2d_params(1, 256, 32)
    in3r = conv2d(inputs, in3r_w, in3r_b, 1)

    in3_w, in3_b = make_conv2d_params(1, 32, 96)
    in3 = conv2d(in3r, in3_w, in3_b, 1)

    inpr = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')

    inp_w, inp_b = make_conv2d_params(1, 256, 64)
    inp = conv2d(inpr, inp_w, inp_b, 1)

    return tf.concat([in1, in2, in3, inp], axis=3)

def Inception(x, channels=3):
    c1_w, c1_b = make_conv2d_params(7, channels, 64)
    c1 = conv2d(x, c1_w, c1_b, 1)

    p1 = tf.nn.max_pool(c1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    n1 = tf.nn.local_response_normalization(p1, depth_radius=5, alpha=0.0001, beta=0.75)

    c2_w, c2_b = make_conv2d_params(1, 64, 64)
    c2 = conv2d(n1, c2_w, c2_b, 1)

    c3_w, c3_b = make_conv2d_params(3, 64, 192)
    c3 = conv2d(c2, c3_w, c3_b, 1)

    n2 = tf.nn.local_response_normalization(c3, depth_radius=5, alpha=0.0001, beta=0.75)

    p2 = tf.nn.max_pool(n2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    incept1 = incept_a(p2)

    incept2 = incept_b(incept1)

    p3 = tf.nn.max_pool(incept2, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')
    print(incept1, incept2, p3)
    fl1 = tf.contrib.layers.flatten(p3)

    fc_W = tf.Variable(tf.truncated_normal(shape=(1920, 43), mean=0, stddev=0.1))

    logits = tf.matmul(fl1, fc_W)
    return logits, "Inception_"+str(channels)

if __name__ == '__main__':
    ''' Test models '''
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    drop_rate = tf.placeholder(tf.float32)

    logits, modelname = Inception(x, 3)
    print(logits)
