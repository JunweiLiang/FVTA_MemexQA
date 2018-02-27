# coding=utf-8
# trainer class, given the model (model has the property loss)



import tensorflow as tf


class Trainer():
	def __init__(self,model,config):
		self.config = config
		self.model = model # this is an model instance		

		self.global_step = model.global_step # 

		self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
		#self.opt = tf.train.AdamOptimizer(config.init_lr)

		self.loss = model.loss # get the loss funcion

		self.summary = model.summary # nothing yet

		self.grads = self.opt.compute_gradients(self.loss) # will train all trainable in Graph
		#config.clip_gradient_norm = 1
		#self.grads = [(tf.clip_by_value(grad, -1*config.clip_gradient_norm, config.clip_gradient_norm), var) for grad, var in self.grads]
		# process gradients?
		self.train_op = self.opt.apply_gradients(self.grads,global_step=self.global_step)


	def step(self,sess,batch,get_summary=False): 
		assert isinstance(sess,tf.Session)
		# idxs is a tuple (23,123,33..) index for sample
		batchIdx,batch_data = batch
		feed_dict = self.model.get_feed_dict(batch_data,is_train=True)
		if get_summary:
			loss, summary, train_op = sess.run([self.loss,self.summary,self.train_op],feed_dict=feed_dict)
		else:
			loss, train_op = sess.run([self.loss,self.train_op],feed_dict=feed_dict)
			summary = None
		return loss, summary, train_op

