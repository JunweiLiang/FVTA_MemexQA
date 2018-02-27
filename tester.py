# coding=utf-8
# tester, given the config with model path


import tensorflow as tf
import numpy as np

class Tester():
	def __init__(self,model,config,sess=None):
		self.config = config
		self.model = model

		self.yp = self.model.yp # the output of the model # [N,M,JX]


	def step(self,sess,batch):
		# give one batch of Dataset, use model to get the result,
		assert isinstance(sess,tf.Session)
		batchIdxs,batch_data =  batch
		feed_dict = self.model.get_feed_dict(batch_data,is_train=False)
		yp, = sess.run([self.yp],feed_dict=feed_dict)
		# clip the output
		# yp should be [N,4]
		yp = yp[:batch_data.num_examples]
		return yp

	def trim(self,input_s,num):
		return [ one[:num] if type(one) == type(np.array(0)) else -1 for one in input_s ]
	# get all the value needed for visualization
	def step_vis(self,sess,batch):
		# give one batch of Dataset, use model to get the result,
		assert isinstance(sess,tf.Session)
		batchIdxs,batch_data =  batch
		feed_dict = self.model.get_feed_dict(batch_data,is_train=False)

		yp,C,C_win,att_logits,q_att_logits,at_mask,ad_mask,when_mask,where_mask,pts_mask,pis_mask,q_mask,hat_len,had_len,hwhen_len,hwhere_len,hpts_len,hpis_len,JXP,warp_h,h,at,ad,when,where,pts,pis,q = sess.run([self.yp,self.model.C,self.model.C_win,self.model.att_logits,self.model.q_att_logits,self.model.at_mask,self.model.ad_mask,self.model.when_mask,self.model.where_mask,self.model.pts_mask,self.model.pis_mask,self.model.q_mask,self.model.hat_len,self.model.had_len,self.model.hwhen_len,self.model.hwhere_len,self.model.hpts_len,self.model.hpis_len,self.model.JXP,self.model.warp_h,self.model.hall,self.model.at,self.model.ad,self.model.when,self.model.where,self.model.pts,self.model.pis,self.model.q],feed_dict=feed_dict)

		# clip the output
		# yp should be [N,4]
		yp = yp[:batch_data.num_examples] # this is needed for the last batch
		# beware some will became -1 after trim
		C,C_win,att_logits,q_att_logits,at_mask,pts_mask,pis_mask,q_mask = self.trim([C,C_win,att_logits,q_att_logits,at_mask,pts_mask,pis_mask,q_mask],batch_data.num_examples)
		return yp,C,C_win,att_logits,q_att_logits,at_mask,ad_mask,when_mask,where_mask,pts_mask,pis_mask,q_mask,hat_len,had_len,hwhen_len,hwhere_len,hpts_len,hpis_len,JXP,warp_h,h,at,ad,when,where,pts,pis,q