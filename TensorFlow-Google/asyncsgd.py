import tensorflow as tf
import os
tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS
num_features = 33762578
indix = []
g = tf.Graph()
with g.as_default():
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones([num_features,1],dtype=tf.float32))
	error = tf.Variable(0) #to calculate the error
#Reading and Parsing
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
            filename_queue = tf.train.string_input_producer([
		"/home/ubuntu/uploaded-scripts/data/tfrecords00",
		"/home/ubuntu/uploaded-scripts/data/tfrecords01",
		"/home/ubuntu/uploaded-scripts/data/tfrecords02",
		"/home/ubuntu/uploaded-scripts/data/tfrecords03",
		"/home/ubuntu/uploaded-scripts/data/tfrecords04"
	    ], num_epochs=None)
	    filename_queue1 = tf.train.string_input_producer([
                "/home/ubuntu/uploaded-scripts/data/tfrecords00",
                "/home/ubuntu/uploaded-scripts/data/tfrecords01",
	    ], num_epochs=None)
	    filename_queue2 = tf.train.string_input_producer([
                "/home/ubuntu/uploaded-scripts/datacheck/tfrecords22",
	    ], num_epochs=None)
	    reader = tf.TFRecordReader(name="operator_%d" % FLAGS.task_index)
	    if FLAGS.task_index!=4:
           	 _, serialized_example = reader.read(filename_queue)
            if FLAGS.task_index==4:
	    	 _, serialized_example = reader.read(filename_queue1)
	    features = tf.parse_single_example(serialized_example,
                                    features={
                                    'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                    'index' : tf.VarLenFeature(dtype=tf.int64),
                                    'value' : tf.VarLenFeature(dtype=tf.float32),
                                    })
	    label = tf.cast(features['label'],tf.float32)
    	    index = features['index']
	    indix = index
    	    value = features['value']
#Calculating local_gradient in optimised manner using gather()
	    w_local = tf.gather(w, index.values)
            local_gradient = tf.scalar_mul(-0.1,tf.mul(tf.mul(label,tf.sigmoid(tf.mul(label,tf.matmul(
		tf.transpose(w_local),tf.expand_dims(value.values,1)))-1)),tf.expand_dims(value.values,1)))
#Reading record data for error checking at vm-0
    with tf.device("/job:worker/task:0"):
                _, serialized_example2 = reader.read(filename_queue2)
                features2 = tf.parse_single_example(serialized_example2,
                                    features={
                                    'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                    'index' : tf.VarLenFeature(dtype=tf.int64),
                                    'value' : tf.VarLenFeature(dtype=tf.float32),
                                    })
                label2 = features2['label']
                index2 = features2['index']
                value2 = features2['value']
                x2 = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index2),
                                   [num_features,],tf.sparse_tensor_to_dense(value2))
                x2 = tf.expand_dims(x2,1)
#Calculating gradient value from the local gradients to be updated to w             	    
    with tf.device("/job:worker/task:0"):
                assign_op = tf.scatter_add(w,indix.values,local_gradient)
		error_op =  error.assign_add(tf.reshape(tf.cast(tf.not_equal(tf.sign(label2),tf.sign(
                        tf.cast(tf.matmul(tf.transpose(w),x2),tf.int64))),tf.int32),[]))
    with tf.Session("grpc://vm-34-%d:2222" % (FLAGS.task_index+1)) as sess:
        if FLAGS.task_index == 0:
#		if False ==  tf.is_variable_initialized(w).eval():
		sess.run(tf.initialize_all_variables())
	coord = tf.train.Coordinator()
        coord = tf.train.start_queue_runners(sess=sess,coord=coord)
	er =0
	for j in range(0, 20):
	    for k in range(0, 100):    	
		weight=sess.run(assign_op)
		#print "iteration %d"% (j+1)
	    	if FLAGS.task_index == 0: 
			err=sess.run(error_op)
	    print "Error in Iteration %d : %d" % (j+1,err-er)
	    print weight
	    er=err
        coord.request_stop()
  	coord.join(threads, stop_grace_period_secs=5)
	sess.close()
