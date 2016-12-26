"""
A sample client used to launch jobs on a Distributed TensorFlow cluster.
In this program, we want to find the sum of traces of 5 random matrices.
Each matrix is generated on a different process in the cluster and its traces
are added together.
"""

import tensorflow as tf
import os
# create a empty graph

def get_block_name(i, j, k):
    return "sub-matrix-"+str(i)+"-"+str(j)+"-"+str(k)


def get_intermediate_trace_name(i, j, k):
    return "inter-"+str(i)+"-"+str(j)+"-"+str(k)

g = tf.Graph()


# make the graph we created as the default graph. All the variables and
# operators we create will be added to this graph
with g.as_default():

    tf.logging.set_verbosity(tf.logging.DEBUG)

    # this sets the random seed for operations on the current graph
    tf.set_random_seed(1024)
    N = 100000 # dimension of the matrix
    R = 5
    d = 100 # number of splits along one dimension. Thus, we will have 100 blocks
    M = int(N / d)
    matrices = {} # a container to hold the operators we just created
    for i in range(0, R):
        with tf.device("/job:worker/task:%d" % i):
            for j in range(i*d/5, i*d/5 + 20):
                for k in range(0, d):
                    matrix_name = get_block_name(i, j, k)
                    matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)
		    print i
		    print j
		    print k
    # container to hold operators that calculate the traces of individual
    # matrices.
    intermediate_traces = {}
    for i in range(0, R):
        with tf.device("/job:worker/task:%d" % i):
            for j in range(i*d/5, i*d/5 + 20):
                for k in range(0, d):
                    A = matrices[get_block_name(i, j, k)]
                    B = matrices[get_block_name(i, j, k)]
                    intermediate_traces[get_intermediate_trace_name(i, j, k)] = tf.trace(tf.matmul(A, B))
		   
    # sum all the traces
    with tf.device("/job:worker/task:0"):
        retval = tf.add_n(intermediate_traces.values())

    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://vm-34-2:2222", config=config) as sess:
        result = sess.run(retval)
        tf.train.SummaryWriter("%s/example_single" % (os.environ.get("TF_LOG_DIR")), sess.graph)
        sess.close()
        print "Trace of the big matrix is = ", result
