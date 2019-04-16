# coding=utf-8
# tensorflow model graph 


import tensorflow as tf
from utils import flatten, reconstruct, Dataset, exp_mask
import numpy as np
import random, sys

VERY_NEGATIVE_NUMBER = -1e30

def get_model(config):
	# implement a multi gpu model?
	with tf.name_scope(config.modelname),  tf.device("/gpu:0"):
		model = Model(config, "model_%s"%config.modelname)

	return model


from copy import deepcopy # for C[i].insert(Y[i])

# a flatten and reconstruct version of softmax
def softmax(logits, scope=None):
	with tf.name_scope(scope or "softmax"): # noted here is name_scope not variable
		flat_logits = flatten(logits, 1)
		flat_out = tf.nn.softmax(flat_logits)
		out = reconstruct(flat_out, logits, 1)
		return out

# softmax selection?
# return target * softmax(logits)
# target: [ ...,  J,  d]
# logits: [ ...,  J]

# so [N, M, dim] * [N, M] -> [N, dim],  so [N, M] is the attention for each M

# return: [ ...,  d] # so the target vector is attended with logits' softmax
# [N, M, JX, JQ, 2d] * [N, M, JX, JQ] (each context to query's mapping) -> [N, M, JX, 2d] # attened the JQ dimension
def softsel(target, logits, hard=False, hardK=None, scope=None):
	with tf.variable_scope(scope or "softsel"): # there is no variable to be learn here
		
		
		a = softmax(logits) # shape is the same

		
		target_rank = len(target.get_shape().as_list())
		# [N, M, JX, JQ, 2d] elem* [N, M, JX, JQ, 1]
		return tf.reduce_sum(tf.expand_dims(a, -1)*target, target_rank-2) # second last dim


# x -> [Num, JX, W, embedding dim] # conv2d requires an input of 4d [batch,  in_height,  in_width,  in_channels]
def conv1d(x, filter_size, height, keep_prob, is_train=None, wd=None, scope=None):
	with tf.variable_scope(scope or "conv1d"):
		num_channels = x.get_shape()[-1] # embedding dim[8]
		filter_var = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype="float")
		bias = tf.get_variable('bias', shape=[filter_size], dtype='float')
		strides = [1, 1, 1, 1]
		# add dropout to input
		
		d = tf.nn.dropout(x, keep_prob=keep_prob)
		outd = tf.cond(is_train, lambda:d, lambda:x)
		#conv
		xc = tf.nn.relu(tf.nn.conv2d(outd, filter_var, strides, padding='VALID')+bias)
		# simple max pooling?
		out = tf.reduce_max(xc, 2) # [-1, JX, num_channel]

		if wd is not None:
			add_wd(wd)

		return out

# fully-connected layer
# simple linear layer,  without activatation # remember to add it
# [N, M, JX, JQ, 2d] => x[N*M*JX*JQ, 2d] * W[2d, output_size] -> 
def linear(x, output_size, scope, add_tanh=False, wd=None):
	with tf.variable_scope(scope):
		# since the input here is not two rank,  we flat the input while keeping the last dims
		keep = 1
		#print x.get_shape().as_list()
		flat_x = flatten(x, keep) # keeping the last one dim # [N, M, JX, JQ, 2d] => [N*M*JX*JQ, 2d]
		#print flat_x.get_shape() # (?,  200) # wd+cwd
		bias_start = 0.0
		if not (type(output_size) == type(1)): # need to be get_shape()[k].value
			output_size = output_size.value

		#print [flat_x.get_shape()[-1], output_size]

		W = tf.get_variable("W", dtype="float", initializer=tf.truncated_normal([flat_x.get_shape()[-1].value, output_size], stddev=0.1))
		bias = tf.get_variable("b", dtype="float", initializer=tf.constant(bias_start, shape=[output_size]))
		flat_out = tf.matmul(flat_x, W)+bias

		if add_tanh:
			flat_out = tf.tanh(flat_out, name="tanh")


		if wd is not None:
			add_wd(wd)

		out = reconstruct(flat_out, x, keep)
		return out


def batch_norm(x, is_train=True, epsilon=1e-5, decay=0.9, scope=None):
	scope = scope or "batch_norm"
	# what about tf.nn.batch_normalization
	return tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon, scale=True, is_training=is_train, scope=scope)



# hinfo * att(hinfo, hq) -> attended_hinfo / [hinfo;attened_hinfo]
# hq -> [N, JQ, w]
# hinfo -> [N, M, J*, w] / [N, M, J*, JX, w]
# h_info_mask -> [N, M, J*]
# hinfo[N, M, J1, w]/[N, M, J1, J2, w] -> h_a[N, w]
# input: hinfo [N,  ...,  2d],  hq [N, JQ, 2d] -> output: [N, 2d]
# simiMatrix: what type of similarity matrix we use for the linear transform
# 	1: A,  B ,  A*B
# 	2:(A-B)^2,  A*B
# 	3: A,  B,  (A-B)^2,  A*B

# attention(hq,  tf.expand_dims(g1_all, 1),  self.q_mask,  add_tanh=config.add_tanh, simiMatrix=config.simiMatrix,  wd=config.wd, bidirect=config.use_bidirection,  scope="question_att")
# hq -> [N, JQ, 2d]
			# g1_all -> [N, 1, 2d]
			# gp -> [N, 2d]
def attention(hinfo, hq, hinfo_mask=None, hq_mask=None, simiMatrix=1, wd=None, add_tanh=False, bidirect=False, scope=None):
	with tf.variable_scope(scope or "attention_2vector"):
		N = hinfo.get_shape().as_list()[0]
		w = hinfo.get_shape().as_list()[-1]
		
		M = tf.shape(hinfo)[1]
		JQ = tf.shape(hq)[1]

		#hinfo_rank = tf.rank(hinfo)

		hinfo = tf.reshape(hinfo, [N, -1, w]) # M*J / M*J*JX # [N, JQ, 2d]
		#print hinfo.get_shape().as_list(), hq.get_shape().as_list()
		if hinfo_mask is not None:
			hinfo_mask = tf.reshape(hinfo_mask, [N, -1]) # [N, JQ]
		#vector length for hinfo
		V = tf.shape(hinfo)[1]
		# so hinfo -> [N, V, w],  hinfo_mask -> [N, V]
		# change two matrix to be the same [N, V, JQ, 2d]
		h_aug = tf.tile(tf.expand_dims(hinfo, 2), [1, 1, JQ, 1])
		q_aug = tf.tile(tf.expand_dims(hq, 1), [1, V, 1, 1])

		if (hinfo_mask is not None) and (hq_mask is not None):
			# change the mask too
			h_mask_aug = tf.tile(tf.expand_dims(hinfo_mask, 2), [1, 1, JQ])
			q_mask_aug = tf.tile(tf.expand_dims(hq_mask, 1), [1, V, 1])

			mask = h_mask_aug & q_mask_aug # [N, V, JQ]

		# get [N, V, JQ]
		if simiMatrix == 1:
			a_logits = linear(tf.concat([h_aug, q_aug, h_aug*q_aug], 3), output_size=1, add_tanh=add_tanh, scope="att_logits")
			a_logits = tf.squeeze(a_logits, 3)
		elif simiMatrix == 2:
			a_logits = linear(tf.concat([h_aug*q_aug, (h_aug-q_aug)*(h_aug-q_aug)], 3), add_tanh=add_tanh, output_size=1, scope="att_logits")
			a_logits = tf.squeeze(a_logits, 3)
		elif simiMatrix == 3:
			a_logits = linear(tf.concat([h_aug, q_aug, (h_aug-q_aug)*(h_aug-q_aug), h_aug*q_aug], 3), output_size=1, add_tanh=add_tanh, scope="att_logits")
			a_logits = tf.squeeze(a_logits, 3)
		elif simiMatrix == 4: 
			# cosine simi,  [N, V, JQ, 2d] -> [N, V, JQ]
			h_aug_norm = tf.nn.l2_normalize(h_aug, -1)
			q_aug_norm = tf.nn.l2_normalize(q_aug, -1)
			a_logits = tf.reduce_sum(tf.multiply(h_aug_norm, q_aug_norm), 3)
		else:
			print "similarity matrix not implemented"
			sys.exit()

		

		# apply mask
		if (hinfo_mask is not None) and (hq_mask is not None):
			a_logits = exp_mask(a_logits, mask)

		# hinfo -> [N, V, w],  * max([N, V, JQ]) [N, V] [so each info "word" 's max prob to the whole question]
		# h_a -> [N, w]
		#h_a = softsel(hinfo, tf.reduce_max(a_logits, 2), hard=True, hardK=3)
		h_a = softsel(hinfo, tf.reduce_max(a_logits, 2), hard=False)

		# add a reversed-directional vector here
		if bidirect:
			# q [N, JQ, w] -> q_aug : [N, V, JQ, w]  * [N, V, JQ] -> [N, V, w]
			q_a = softsel(q_aug, a_logits, hard=False) # each V attended with query

			# here we simply average them
			q_a = tf.reduce_mean(q_a, 1) # [N, w]

			# concat two direction attended vector
			h_a = tf.concat([h_a, q_a], 1) #[N, 2w]

			# need output to be [N, 2w]
			#h_a = linear(h_a, output_size=w, scope="combine_bidirect")


		if wd is not None:
			add_wd(wd)

		return h_a, a_logits

# hinfo -> [N, K, M, JMAX, 2d],  K is the modality
# hq -> [N, JQ, w]
# h_info_mask -> [N, K, M, JMAX]
# simiMatrix: what type of similarity matrix we use for the linear transform
# 	1: A,  B ,  A*B
# 	2:(A-B)^2,  A*B
# 	3: A,  B,  (A-B)^2,  A*B
def attention_3d(hinfo, hq, hinfo_mask=None, hq_mask=None, simiMatrix=1, wd=None, add_tanh=False, time_warp_att=False, C=None, bidirect=False, scope=None):
	with tf.variable_scope(scope or "attention_2vector"):
		# static shape that we now before runtime
		N = hinfo.get_shape().as_list()[0]
		w = hinfo.get_shape().as_list()[-1]
		K = hinfo.get_shape().as_list()[1]

		M = tf.shape(hinfo)[2]
		JQ = tf.shape(hq)[1]

		#hinfo_rank = tf.rank(hinfo)

		hinfo = tf.reshape(hinfo, [N, K, -1, w]) # M*JMAX
		#print hinfo.get_shape().as_list(), hq.get_shape().as_list()
		if hinfo_mask is not None:
			hinfo_mask = tf.reshape(hinfo_mask, [N, K, -1])
		#vector length for hinfo
		T = tf.shape(hinfo)[2]
		# so hinfo -> [N, K, T, w],  hinfo_mask -> [N, K, T]
		# change two matrix to be the same
		h_aug = tf.tile(tf.expand_dims(hinfo, 3), [1, 1, 1, JQ, 1])
		q_aug = tf.tile(tf.expand_dims(tf.expand_dims(hq, 1), 1), [1, K, T, 1, 1])

		if (hinfo_mask is not None) and (hq_mask is not None):
			# change the mask too
			h_mask_aug = tf.tile(tf.expand_dims(hinfo_mask, 3), [1, 1, 1, JQ])
			q_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(hq_mask, 1), 1), [1, K, T, 1])

			mask = h_mask_aug & q_mask_aug # [N, K, T, JQ]

		# [N, K, T, JQ, 2d] -> [N, K, T, JQ, 1]
		if simiMatrix == 1:
			a_logits = linear(tf.concat([h_aug, q_aug, h_aug*q_aug], 4), output_size=1, add_tanh=add_tanh, scope="att_logits")
			a_logits = tf.squeeze(a_logits, 4)
		elif simiMatrix == 2:
			a_logits = linear(tf.concat([h_aug*q_aug, (h_aug-q_aug)*(h_aug-q_aug)], 4), output_size=1, add_tanh=add_tanh, scope="att_logits")
			a_logits = tf.squeeze(a_logits, 4)
		elif simiMatrix == 3:
			a_logits = linear(tf.concat([h_aug, q_aug, (h_aug-q_aug)*(h_aug-q_aug), h_aug*q_aug], 4), output_size=1, add_tanh=add_tanh, scope="att_logits")
			a_logits = tf.squeeze(a_logits, 4)
		elif simiMatrix == 4: 
			# cosine simi,  [N, K, T, JQ, 2d] -> [N, K, T, JQ]
			h_aug_norm = tf.nn.l2_normalize(h_aug, -1)
			q_aug_norm = tf.nn.l2_normalize(q_aug, -1)
			a_logits = tf.reduce_sum(tf.multiply(h_aug_norm, q_aug_norm), 4)
		else:
			print "similarity matrix not implemented"
			sys.exit()

		# [N, K, T, JQ]
		

		# apply mask
		if (hinfo_mask is not None) and (hq_mask is not None):
			a_logits = exp_mask(a_logits, mask)

		# attend on T (timestep),  then K (modality)
		# hinfo -> [N, K, T, w] -> [N, K, w] -> [N, w] h_a
		a_logits_maxed = tf.reduce_max(a_logits, 3) # [N, K, T]
		if time_warp_att:
			T = tf.shape(C)[-1]
			# time_warp with matrix C
			# C [N, T, T]
			a_temp = tf.tile(tf.expand_dims(a_logits_maxed, 3), [1, 1, 1, T]) #[N, K, T, T]
			C_ex = tf.tile(tf.expand_dims(C, 1), [1, K, 1, 1])# [N, K, T, T]
			a_logits_maxed = tf.reduce_sum(a_temp*C_ex, -1) #[N, K, T]
			

		h_a = softsel(softsel(hinfo, a_logits_maxed), tf.reduce_max(a_logits, [3, 2]))#, hard=False, hardK=3)

		# add a reversed-directional vector here
		if bidirect:
			# q [N, JQ, w] -> q_aug : [N, V, JQ, w]  * [N, V, JQ] -> [N, V, w]
			q_a = softsel(q_aug, a_logits, hard=False) # each V attended with query

			# here we simply average them
			q_a = tf.reduce_mean(q_a, 1) # [N, w]

			# concat two direction attended vector
			h_a = tf.concat([h_a, q_a], 1) #[N, 2w]

			# need output to be [N, 2w]
			#h_a = linear(h_a, output_size=w, scope="combine_bidirect")


		if wd is not None:
			add_wd(wd)

		return h_a, a_logits

# time correlation times the indication function,  given [N, T, T]  get a new [N, T, T]
def time_indication_func(C, warp_type=1, scope=None):
	with tf.variable_scope(scope or "time_warp"):
		if warp_type==1: # all Time 
			return C, None
		else:
			T = tf.shape(C)[-1]
			N = C.get_shape().as_list()[0]

			one = tf.diag(tf.ones([T], dtype=tf.float32)) # [T, T]
			one = tf.tile(tf.expand_dims(one, 0), [N, 1, 1])

			if warp_type == 2: # current time??
				#one = tf.eye(T, dtype=tf.float32, batch_shape=N)
				return one*C, None
			elif warp_type == 3: # past
				# need T to be constant
				#a = np.zeros((T, T), dtype="float")
				#a[np.tril_indices(T, 0)] = 1.0
				#past = tf.constant(a)
				past = tf.matrix_band_part(tf.ones([T, T], dtype=tf.float32), -1, 0) # lower triangular
				past = tf.tile(tf.expand_dims(past, 0), [N, 1, 1])

				return past*C, None
			elif warp_type == 4: # future
				#a = np.zeros((T, T), dtype="float")
				#a[np.triu_indices(T, 0)] = 1.0
				#future = tf.constant(a)
				future = tf.matrix_band_part(tf.ones([T, T], dtype=tf.float32), 0, -1) # upper triangular
				future = tf.tile(tf.expand_dims(future, 0), [N, 1, 1])
				return future*C, None

			elif warp_type == 5: # past-future with trainable window
				window_t_init=3
				window_t = tf.get_variable('time_warp_window_t', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(window_t_init), trainable=True)
				window_t_int = tf.cast(tf.ceil(window_t), tf.int64)

				windowed_C = tf.matrix_band_part(tf.ones([T, T], dtype=tf.float32), window_t_int, window_t_int)
				windowed_C = tf.tile(tf.expand_dims(windowed_C, 0), [N, 1, 1])
				return windowed_C*C, window_t
			else:
				raise Exception("time warping type not implemented")




# add current scope's variable's l2 loss to loss collection
def add_wd(wd, scope=None):
	if wd != 0.0:
		scope = scope or tf.get_variable_scope().name
		vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope=scope)
		with tf.variable_scope("weight_decay"):
			for var in vars_:
				weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="%s/wd"%(var.op.name))
				tf.add_to_collection("losses", weight_decay)


def get_initializer(matrix):
	def _initializer(shape,  dtype=None,  partition_info=None,  **kwargs): return matrix
	return _initializer

class Model():
	def __init__(self, config, scope):
		self.scope = scope
		self.config = config
		# a step var to keep track of current training process
		self.global_step = tf.get_variable('global_step', shape=[], dtype='int32', initializer=tf.constant_initializer(0), trainable=False) # a counter

		# get all the dimension here
		N = self.N = config.batch_size
		
		VW = self.VW = config.word_vocab_size
		VC = self.VC = config.char_vocab_size
		W = self.W = config.max_word_size

		# embedding dim
		self.cd, self.wd, self.cwd = config.char_emb_size, config.word_emb_size, config.char_out_size

		# image dimension
		self.idim = config.image_feat_dim

		self.num_choice = config.num_choice

		# these could be used for visualization
		self.C = tf.constant(-1) # the time correlation matrix
		self.C_win = tf.constant(-1)
		self.att_logits = tf.constant(-1) # the 3d attention logits
		self.q_att_logits = tf.constant(-1) # the question attention logits if there is 
		self.JXP = tf.constant(-1)

		# for version 1 1-dim attention
		self.hat_len = tf.constant(-1)
		self.had_len = tf.constant(-1)
		self.hwhen_len = tf.constant(-1)
		self.hwhere_len = tf.constant(-1)
		self.hpis_len = tf.constant(-1)
		self.hpts_len = tf.constant(-1)

		#for showing the h vectors
		self.warp_h = tf.constant(-1)
		

		# step limits
		# M -> album max num
		# -----JX -> title max words (album title, photo title)
		# JXA -> album title max words
		# JXP -> photo title max words
		# JD -> album description max word
		# JT -> album when max word
		# JG -> album where max word

		# JI -> album max photo

		# JA -> max answer (choice) length
		# JQ -> max question length

		# all the inputs

		# album title
		# [N, M, JXA]
		self.at = tf.placeholder('int32', [N, None, None], name="at")
		self.at_c = tf.placeholder("int32", [N, None, None, W], name="at_c")
		self.at_mask = tf.placeholder("bool", [N, None, None], name="at_mask") # to get the sequence length

		# album description
		# [N, M, JD]
		self.ad = tf.placeholder('int32', [N, None, None], name="ad")
		self.ad_c = tf.placeholder("int32", [N, None, None, W], name="ad_c")
		self.ad_mask = tf.placeholder("bool", [N, None, None], name="ad_mask")

		# album when,  where
		# [N, M, JT/JG]
		self.when = tf.placeholder("int32", [N, None, None], name="when")
		self.when_c = tf.placeholder("int32", [N, None, None, W], name="when_c")
		self.when_mask = tf.placeholder("bool", [N, None, None], name="when_mask")
		self.where = tf.placeholder("int32", [N, None, None], name="where")
		self.where_c = tf.placeholder("int32", [N, None, None, W], name="where_c")
		self.where_mask = tf.placeholder("bool", [N, None, None], name="where_mask")

		# photo titles
		# [N, M, JI, JXP]
		self.pts = tf.placeholder('int32', [N, None, None, None], name="pts")
		self.pts_c = tf.placeholder("int32", [N, None, None, None, W], name="pts_c")
		self.pts_mask = tf.placeholder("bool", [N, None, None, None], name="pts_mask")

		# for vis
		self.JXP = tf.shape(self.pts)[3]


		# photo
		# [N, M, JI] # each is a photo index
		self.pis = tf.placeholder('int32', [N, None, None], name="pis")
		self.pis_mask = tf.placeholder("bool", [N, None, None], name="pis_mask")

		# question
		self.q = tf.placeholder('int32', [N, None], name="q")
		self.q_c = tf.placeholder('int32',  [N,  None,  W],  name='q_c')
		self.q_mask = tf.placeholder("bool", [N, None], name="q_mask")

		# answer + choice words
		# [N, 4, JA]
		self.choices = tf.placeholder("int32", [N, self.num_choice, None], name="choices")
		self.choices_c = tf.placeholder("int32", [N, self.num_choice, None, W], name="choices_c")
		self.choices_mask = tf.placeholder("bool", [N, self.num_choice, None], name="choices_mask")

		# 4 choice classification
		self.y = tf.placeholder('bool',  [N,  self.num_choice],  name='y')
		

		# feed in the pretrain word vectors for all batch
		self.existing_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name="pre_emb_mat")

		# feed in the image feature for this batch
		# [photoNumForThisBatch, image_dim]
		self.image_emb_mat = tf.placeholder("float", [None, config.image_feat_dim], name="image_emb_mat")

		# used for drop out switch
		self.is_train = tf.placeholder('bool',  [],  name='is_train')

		# forward output
		# the following will be added in build_forward and build_loss()
		self.logits = None

		self.yp = None # prob

		self.loss = None

		self.build_forward()
		self.build_loss()

		self.summary = tf.summary.merge_all() # for visualize and stuff? # not used now


	def build_forward(self):
		config = self.config
		VW = self.VW
		VC = self.VC
		W = self.W
		N = self.N
		# dynamic decide some step,  for sequence length
		M = tf.shape(self.pis)[1] # photo num
		JXA = tf.shape(self.at)[2] # for album title,  photo title
		JD = tf.shape(self.ad)[2] # description length
		JT = tf.shape(self.when)[2]
		JG = tf.shape(self.where)[2]

		JI = tf.shape(self.pis)[2] # used for photo_title,  photo
		JXP = tf.shape(self.pts)[3]


		JQ = tf.shape(self.q)[1]
		JA = tf.shape(self.choices)[2]

		# embeding size
		cdim, wdim, cwdim = self.cd, self.wd, self.cwd #cwd: char -> word output dimension
		# image feature dim
		idim = self.idim # image_feat dimension

		# all input:
		#	at,  ad,  when,  where,  
		#	pts,  pis
		#	q,  choices

		# embedding
		with tf.variable_scope('emb'):
			# char stuff
			if config.use_char:
			#with tf.variable_scope("char"):
				# [char_vocab_size, char_emb_dim]
				with tf.variable_scope("var"): #,  tf.device("/cpu:0"): 
					char_emb = tf.get_variable("char_emb", shape=[VC, cdim], dtype="float")

				# the embedding for each of character 
				# [N, M, JXA, W]
				Aat_c = tf.nn.embedding_lookup(char_emb, self.at_c)
				# [N, M, JD, W]
				Aad_c = tf.nn.embedding_lookup(char_emb, self.ad_c)
				# [N, M, JT, W]
				Awhen_c = tf.nn.embedding_lookup(char_emb, self.when_c)
				# [N, M, JG, W]
				Awhere_c = tf.nn.embedding_lookup(char_emb, self.where_c)
				# [N, M, JI, JXP, W] -> [N, M, JI, JXP, W, cdim]
				Apts_c = tf.nn.embedding_lookup(char_emb, self.pts_c)

				# [N, JQ, W]
				Aq_c = tf.nn.embedding_lookup(char_emb, self.q_c)
				Achoices_c = tf.nn.embedding_lookup(char_emb, self.choices_c)

				# flatten for conv2d input like images
				Aat_c = tf.reshape(Aat_c, [-1, JXA, W, cdim])
				Aad_c = tf.reshape(Aad_c, [-1, JD, W, cdim])
				Awhen_c = tf.reshape(Awhen_c, [-1, JT, W, cdim])
				Awhere_c = tf.reshape(Awhere_c, [-1, JG, W, cdim])
				# [N*M*JI, JXP, W, cdim]
				Apts_c = tf.reshape(Apts_c, [-1, JXP, W, cdim])

				Aq_c = tf.reshape(Aq_c, [-1, JQ, W, cdim])
				# [N*4, ]
				Achoices_c = tf.reshape(Achoices_c, [-1, JA, W, cdim])

				#char CNN
				filter_size = cwdim # output size for each word
				filter_height = 5
				with tf.variable_scope("conv"):
					xat = conv1d(Aat_c, filter_size, filter_height, config.keep_prob, self.is_train, wd=config.wd, scope="conv1d")
					tf.get_variable_scope().reuse_variables()
					xad = conv1d(Aad_c, filter_size, filter_height, config.keep_prob, self.is_train, wd=config.wd, scope="conv1d")
					xwhen = conv1d(Awhen_c, filter_size, filter_height, config.keep_prob, self.is_train, wd=config.wd, scope="conv1d")
					xwhere = conv1d(Awhere_c, filter_size, filter_height, config.keep_prob, self.is_train, wd=config.wd, scope="conv1d")
					xpts = conv1d(Apts_c, filter_size, filter_height, config.keep_prob, self.is_train, wd=config.wd, scope="conv1d")
					qq = conv1d(Aq_c, filter_size, filter_height, config.keep_prob, self.is_train, wd=config.wd, scope="conv1d")
					qchoices = conv1d(Achoices_c, filter_size, filter_height, config.keep_prob, self.is_train, wd=config.wd, scope="conv1d")

					# reshape them back
					xat = tf.reshape(xat, [-1, M, JXA, cwdim])
					xad = tf.reshape(xad, [-1, M, JD, cwdim])
					xwhen = tf.reshape(xwhen, [-1, M, JT, cwdim])
					xwhere = tf.reshape(xwhere, [-1, M, JG, cwdim])
					xpts = tf.reshape(xpts, [-1, M, JI, JXP, cwdim])

					qq = tf.reshape(qq, [-1, JQ, cwdim])
					# [N, num_choice, JA, cwdim]
					qchoices = tf.reshape(qchoices, [-1, self.num_choice, JA, cwdim])

			# word stuff
			with tf.variable_scope('word'):
				with tf.variable_scope("var"):
					# get the word embedding for new words
					if config.is_train:
						# for new word
						word_emb_mat = tf.get_variable("word_emb_mat", dtype="float", shape=[VW, wdim], initializer=get_initializer(config.emb_mat)) # it's just random initialized
					else: # save time for loading the emb during test
						word_emb_mat = tf.get_variable("word_emb_mat", dtype="float", shape=[VW, wdim])
					# concat with pretrain vector
					# so 0 - VW-1 index for new words,  the rest for pretrain vector
					# and the pretrain vector is fixed
					word_emb_mat = tf.concat([word_emb_mat, self.existing_emb_mat], 0)

				#[N, M, JXA] -> [N, M, JXA, wdim]
				Aat = tf.nn.embedding_lookup(word_emb_mat, self.at)
				Aad = tf.nn.embedding_lookup(word_emb_mat, self.ad)
				Awhen = tf.nn.embedding_lookup(word_emb_mat, self.when)
				Awhere = tf.nn.embedding_lookup(word_emb_mat, self.where)
				Apts = tf.nn.embedding_lookup(word_emb_mat, self.pts)

				Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)
				Achoices = tf.nn.embedding_lookup(word_emb_mat, self.choices)

			# concat char and word
			if config.use_char:

				xat = tf.concat([xat, Aat], 3)
				xad = tf.concat([xad, Aad], 3)
				xwhen = tf.concat([xwhen, Awhen], 3)
				xwhere = tf.concat([xwhere, Awhere], 3)
				# [N, M, JI, JX, wdim+cwdim]
				xpts = tf.concat([xpts, Apts], 4)

				# [N, JQ, wdim+cwdim]
				qq = tf.concat([qq, Aq], 2)
				qchoices = tf.concat([qchoices, Achoices], 3)

			else:
				xat = Aat
				xad = Aad
				xwhen = Awhen
				xwhere = Awhere
				xpts = Apts

				qq = Aq
				qchoices = Achoices
				# all the above last dim is the same [wdim+cwdim] or just [wdim]

			# get the image feature
			with tf.variable_scope("image"):

				# [N, M, JI] -> [N, M, JI, idim]
				xpis = tf.nn.embedding_lookup(self.image_emb_mat, self.pis)

				# use image trans,  then linearly transform it to lower dim
				# TODO: CNN transform?
				if config.use_image_trans:
					
					with tf.variable_scope("image_transform"):
						#[N, M, JI, idim] -> [N, M, JI, newdim]
						xpis = linear(xpis, add_tanh=config.add_tanh, output_size=config.image_trans_dim, wd=config.wd, scope="image_trans_linear")

			

		d = config.hidden_size

		# LSTM / GRU?
		cell_text = tf.nn.rnn_cell.BasicLSTMCell(d, state_is_tuple=True)
		cell_img = tf.nn.rnn_cell.BasicLSTMCell(d, state_is_tuple=True)
		#cell_text = tf.nn.rnn_cell.GRUCell(d)
		#cell_img = tf.nn.rnn_cell.GRUCell(d)
		# add dropout
		keep_prob = tf.cond(self.is_train, lambda:tf.constant(config.keep_prob), lambda:tf.constant(1.0))

		cell_text = tf.nn.rnn_cell.DropoutWrapper(cell_text, keep_prob)

		cell_img = tf.nn.rnn_cell.DropoutWrapper(cell_img, keep_prob)

		
		# it is important to think about which LSTM shared with which?

		# sequence length for each
		at_len = tf.reduce_sum(tf.cast(self.at_mask, "int32"), 2) # [N, M] # each album's title length
		ad_len = tf.reduce_sum(tf.cast(self.ad_mask, "int32"), 2)
		when_len = tf.reduce_sum(tf.cast(self.when_mask, "int32"), 2)
		where_len = tf.reduce_sum(tf.cast(self.where_mask, "int32"), 2) # [N, M]

		pis_len = tf.reduce_sum(tf.cast(self.pis_mask, "int32"), 2) #[N, M, JI] #[N, M]

		pts_len = tf.reduce_sum(tf.cast(self.pts_mask, "int32"), 3) # [N, M, JI, JXP] -> [N, M, JI]

		q_len = tf.reduce_sum(tf.cast(self.q_mask, "int32"), 1) # [N] # each question 's length

		choices_len = tf.reduce_sum(tf.cast(self.choices_mask, "int32"), 2) # [N, 4]

		# xat -> [N, M, JXA, wdim+cwdim]
		# xad -> [N, M, JD, wdim+cwdim]
		# xwhen/xwhere -> [N, M, JT/JG, wdim+cwdim]
		# xpts -> [N, M, JI, JXP, wdim+cwdim]

		# xpis -> [N, M, JI, idim]

		# qq -> [N, JQ, wdim+cwdim]
		# qchoices -> [N, 4, JA, wdim+cwdim]

		# roll the sentence into lstm for context and question
		# from [N, M, JI, JX] -> [N, M, 2d]
		with tf.variable_scope("reader"):
			with tf.variable_scope("text"):
				(fw_hq, bw_hq), (fw_lq, bw_lq) = tf.nn.bidirectional_dynamic_rnn(cell_text, cell_text, qq, sequence_length=q_len, dtype="float", scope="utext")
				# concat the fw and backward lstm output
				hq = tf.concat([fw_hq, bw_hq], 2)
				lq = tf.concat([fw_lq.h, bw_lq.h], 1) #LSTM CELL
				#lq = tf.concat([fw_lq, bw_lq], 1) # GRU

				tf.get_variable_scope().reuse_variables()

				# flat all
				# choices
				flat_qchoices = flatten(qchoices, 2) # [N, 4, JA, dim] -> [N*4, JA, dim]
				# album title
				flat_xat = flatten(xat, 2) #[N, M, JXA, dim] -> [N*M, JXA, dim]
				flat_xad = flatten(xad, 2)
				flat_xwhen = flatten(xwhen, 2)
				flat_xwhere = flatten(xwhere, 2)
				
				#print "flat_xpis shape:%s"%(flat_xpis.get_shape())

				# photo tiles
				flat_xpts = flatten(xpts, 2) # [N, M, JI, JXP, dim] -> [N*M*JI, JXP, dim]
				#print "flat_xpts shape:%s"%(flat_xpts.get_shape())

				# get the sequence length,  all one dim
				flat_qchoices_len = flatten(choices_len, 0) # [N*4]
				flat_xat_len = flatten(at_len, 0) # [N*M]
				flat_xad_len = flatten(ad_len, 0) # [N*M]
				flat_xwhen_len = flatten(when_len, 0) # [N*M]
				flat_xwhere_len = flatten(where_len, 0) # [N*M]
				
				flat_xpts_len = flatten(pts_len, 0) # [N*M*JI]

				
				# put all through LSTM
				# uncomment to use ALL LSTM output or LAST LSTM output

				# album title
				# [N*M, JXA, d]
				(fw_hat_flat, bw_hat_flat), (fw_lat_flat, bw_lat_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text, cell_text, flat_xat, sequence_length=flat_xat_len, dtype="float", scope="utext")
				fw_hat = reconstruct(fw_hat_flat, xat, 2) # 
				bw_hat = reconstruct(bw_hat_flat, xat, 2)
				hat = tf.concat([fw_hat, bw_hat], 3) # [N, M, JXA, 2d]
				# lstm
				fw_lat = tf.reshape(fw_lat_flat.h, [N, M, d]) # [N*M, d] -> [N, M, d]
				bw_lat = tf.reshape(bw_lat_flat.h, [N, M, d])
				# GRU
				#fw_lat = tf.reshape(fw_lat_flat, [N, M, d]) # [N*M, d] -> [N, M, d]
				#bw_lat = tf.reshape(bw_lat_flat, [N, M, d])
				lat = tf.concat([fw_lat, bw_lat], 2) # [N, M, 2d]


				# album desciption
				# [N*M, JD, d]
				(fw_had_flat, bw_had_flat), (fw_lad_flat, bw_lad_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text, cell_text, flat_xad, sequence_length=flat_xad_len, dtype="float", scope="utext")
				fw_had = reconstruct(fw_had_flat, xad, 2) # 
				bw_had = reconstruct(bw_had_flat, xad, 2)
				had = tf.concat([fw_had, bw_had], 3) # [N, M, JD, 2d]
				# LSTM
				fw_lad = tf.reshape(fw_lad_flat.h, [N, M, d]) # [N*M, d] -> [N, M, d]
				bw_lad = tf.reshape(bw_lad_flat.h, [N, M, d])
				# GRU
				#fw_lad = tf.reshape(fw_lad_flat, [N, M, d]) # [N*M, d] -> [N, M, d]
				#bw_lad = tf.reshape(bw_lad_flat, [N, M, d])
				lad = tf.concat([fw_lad, bw_lad], 2) # [N, M, 2d]

				# when
				(fw_hwhen_flat, bw_hwhen_flat), (fw_lwhen_flat, bw_lwhen_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text, cell_text, flat_xwhen, sequence_length=flat_xwhen_len, dtype="float", scope="utext")
				fw_hwhen = reconstruct(fw_hwhen_flat, xwhen, 2) # 
				bw_hwhen = reconstruct(bw_hwhen_flat, xwhen, 2)
				hwhen = tf.concat([fw_hwhen, bw_hwhen], 3) # [N, M, JT, 2d]
				# LSTM
				fw_lwhen = tf.reshape(fw_lwhen_flat.h, [N, M, d]) # [N*M, d] -> [N, M, d]
				bw_lwhen = tf.reshape(bw_lwhen_flat.h, [N, M, d])
				# GRU
				#fw_lwhen = tf.reshape(fw_lwhen_flat, [N, M, d]) # [N*M, d] -> [N, M, d]
				#bw_lwhen = tf.reshape(bw_lwhen_flat, [N, M, d])
				lwhen = tf.concat([fw_lwhen, bw_lwhen], 2) # [N, M, 2d]


				# where
				(fw_hwhere_flat, bw_hwhere_flat), (fw_lwhere_flat, bw_lwhere_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text, cell_text, flat_xwhere, sequence_length=flat_xwhere_len, dtype="float", scope="utext")
				fw_hwhere = reconstruct(fw_hwhere_flat, xwhere, 2) # 
				bw_hwhere = reconstruct(bw_hwhere_flat, xwhere, 2)
				hwhere = tf.concat([fw_hwhere, bw_hwhere], 3) # [N, M, JG, 2d]
				# LSTM
				fw_lwhere = tf.reshape(fw_lwhere_flat.h, [N, M, d]) # [N*M, d] -> [N, M, d]
				bw_lwhere = tf.reshape(bw_lwhere_flat.h, [N, M, d])
				# GRU
				#fw_lwhere = tf.reshape(fw_lwhere_flat, [N, M, d]) # [N*M, d] -> [N, M, d]
				#bw_lwhere = tf.reshape(bw_lwhere_flat, [N, M, d])
				lwhere = tf.concat([fw_lwhere, bw_lwhere], 2) # [N, M, 2d]


				# photo title
				# [N*M*JI, JXP, d]
				(fw_hpts_flat, bw_hpts_flat), (fw_lpts_flat, bw_lpts_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text, cell_text, flat_xpts, sequence_length=flat_xpts_len, dtype="float", scope="utext")
				fw_hpts = reconstruct(fw_hpts_flat, xpts, 2) # 
				bw_hpts = reconstruct(bw_hpts_flat, xpts, 2) # [N, M, JI, JXP, d]
				hpts = tf.concat([fw_hpts, bw_hpts], 4) # [N, M, JI, JXP, 2d]
				# LSTM
				fw_lpts = tf.reshape(fw_lpts_flat.h, [N, M, JI, d]) # [N*M*JI, d] -> [N, M, JI, d]
				bw_lpts = tf.reshape(bw_lpts_flat.h, [N, M, JI, d])
				# GRU
				#fw_lpts = tf.reshape(fw_lpts_flat, [N, M, JI, d]) # [N*M*JI, d] -> [N, M, JI, d]
				#bw_lpts = tf.reshape(bw_lpts_flat, [N, M, JI, d])
				lpts = tf.concat([fw_lpts, bw_lpts], 3) # [N, M, JI, 2d]	

				# choices
				(fw_hchoices_flat, bw_hchoices_flat), (fw_lchoices_flat, bw_lchoices_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text, cell_text, flat_qchoices, sequence_length=flat_qchoices_len, dtype="float", scope="utext")
				fw_hchoices = reconstruct(fw_hchoices_flat, qchoices, 2) # 
				bw_hchoices = reconstruct(bw_hchoices_flat, qchoices, 2)
				hchoices = tf.concat([fw_hchoices, bw_hchoices], 3) # [N, 4, JA, 2d]
				# LSTM
				fw_lchoices = tf.reshape(fw_lchoices_flat.h, [N, -1, d]) # [N*4, d] -> [N, 4, d]
				bw_lchoices = tf.reshape(bw_lchoices_flat.h, [N, -1, d])
				# GRU
				#fw_lchoices = tf.reshape(fw_lchoices_flat, [N, -1, d]) # [N*4, d] -> [N, 4, d]
				#bw_lchoices = tf.reshape(bw_lchoices_flat, [N, -1, d])
				lchoices = tf.concat([fw_lchoices, bw_lchoices], 2) # [N, 4, 2d]


			with tf.variable_scope("image"):
				# photos
				flat_xpis = flatten(xpis, 2) # [N, M, JI, idim] -> [N*M, JI, idim]
				flat_xpis_len = flatten(pis_len, 0) # [N*M]


				# photo # use different LSTM
				# [N*M, JXP, d]
				(fw_hpis_flat, bw_hpis_flat), (fw_lpis_flat, bw_lpis_flat) = tf.nn.bidirectional_dynamic_rnn(cell_img, cell_img, flat_xpis, sequence_length=flat_xpis_len, dtype="float", scope="uimage")
				fw_hpis = reconstruct(fw_hpis_flat, xpis, 2) # 
				bw_hpis = reconstruct(bw_hpis_flat, xpis, 2) # [N, M, JI, JXP, d]
				hpis = tf.concat([fw_hpis, bw_hpis], 3) # [N, M, JI, 2d]
				# LSTM
				fw_lpis = tf.reshape(fw_lpis_flat.h, [N, M, d]) # [N*M, d] -> [N, M, d]
				bw_lpis = tf.reshape(bw_lpis_flat.h, [N, M, d])
				# GRU
				#fw_lpis = tf.reshape(fw_lpis_flat, [N, M, d]) # [N*M, d] -> [N, M, d]
				#bw_lpis = tf.reshape(bw_lpis_flat, [N, M, d])
				lpis = tf.concat([fw_lpis, bw_lpis], 2) # [N, M, 2d]

			if config.wd is not None: # l2 weight decay for the reader
				add_wd(config.wd)

		# all rnn output
		# hq -> [N, JQ, 2d]

		# hat -> [N, M, JXA, 2d]
		# had -> [N, M, JD, 2d]
		# hwhen -> [N, M, JT, 2d]
		# hwhere -> [N, M, JG, 2d]
		# hpts -> [N, M, JI, JXP, 2d]
		# hpis -> [N, M, JI, 2d]

		# hchoices -> [N, 4, JA, 2d]

		# last states:
		# lq -> [N, 2d]

		# lat -> [N, M, 2d]
		# lad -> [N, M, 2d]
		# lwhen -> [N, M, 2d]
		# lwhere -> [N, M, 2d]
		# lpts -> [N, M, JI, 2d]
		# lpis -> [N, M, 2d]

		# lchoices -> [N, 4, 2d]

		# padding to become one context tensor
		K = 6  # modality
		if config.no_photo:
			K=5
		with tf.variable_scope("context_tensor"):
			# first,  pad all sequence to JMAX, 
			# this is different for each batch
			JMAX = tf.reduce_max([JXA, JD, JT, JG, JI*JXP, JI])
			#print JMAX

			# pad all # paddins is shape (n, 2),  where n is the rank of x

			hat_pad = tf.pad(hat, paddings=[[0, 0], [0, 0], [0, JMAX-JXA], [0, 0]], mode="CONSTANT", name="hat_pad") # , constant_values=0 # tf 1.3
			
			hat_pad = tf.reshape(hat_pad, [N, M, -1, 2*d]) # need to do this to let tf know the shape?

			had_pad = tf.pad(had, paddings=[[0, 0], [0, 0], [0, JMAX-JD], [0, 0]], mode="CONSTANT", name="had_pad")

			hwhen_pad = tf.pad(hwhen, paddings=[[0, 0], [0, 0], [0, JMAX-JT], [0, 0]], mode="CONSTANT", name="hwhen_pad")

			hwhere_pad = tf.pad(hwhere, paddings=[[0, 0], [0, 0], [0, JMAX-JG], [0, 0]], mode="CONSTANT", name="hwhere_pad")

			hpis_pad = tf.pad(hpis, paddings=[[0, 0], [0, 0], [0, JMAX-JI], [0, 0]], mode="CONSTANT", name="hpis_pad")

			hpts_re = tf.reshape(hpts, [N, M, -1, 2*d])

			hpts_pad = tf.pad(hpts_re, paddings=[[0, 0], [0, 0], [0, JMAX-JI*JXP], [0, 0]], mode="CONSTANT", name="hpts_pad")

			at_mask_pad = tf.pad(self.at_mask, paddings=[[0, 0], [0, 0], [0, JMAX-JXA]], mode="CONSTANT", name="at_mask_pad") # , constant_values=0 # tf 1.3

			ad_mask_pad = tf.pad(self.ad_mask, paddings=[[0, 0], [0, 0], [0, JMAX-JD]], mode="CONSTANT", name="ad_mask_pad")

			when_mask_pad = tf.pad(self.when_mask, paddings=[[0, 0], [0, 0], [0, JMAX-JT]], mode="CONSTANT", name="when_mask_pad")
			where_mask_pad = tf.pad(self.where_mask, paddings=[[0, 0], [0, 0], [0, JMAX-JG]], mode="CONSTANT", name="where_mask_pad")

			pis_mask_pad = tf.pad(self.pis_mask, paddings=[[0, 0], [0, 0], [0, JMAX-JI]], mode="CONSTANT", name="pis_mask_pad")

			pts_mask_re = tf.reshape(self.pts_mask, [N, M, -1])

			pts_mask_pad = tf.pad(pts_mask_re, paddings=[[0, 0], [0, 0], [0, JMAX-JI*JXP]], mode="CONSTANT", name="pts_mask_pad")

			# now they should all be [N, M, JMAX, 2d]
			if config.no_photo:
				hall = tf.stack([hat_pad, had_pad, hwhen_pad, hwhere_pad, hpts_pad], axis=1) # [N, K, M, JMAX, 2d]

				hall_mask = tf.stack([at_mask_pad, ad_mask_pad, when_mask_pad, where_mask_pad, pts_mask_pad], axis=1) # [N, K, M, JMAX]
			else:
			# stack!
				hall = tf.stack([hat_pad, had_pad, hwhen_pad, hwhere_pad, hpts_pad, hpis_pad], axis=1) # [N, K, M, JMAX, 2d]

				hall_mask = tf.stack([at_mask_pad, ad_mask_pad, when_mask_pad, where_mask_pad, pts_mask_pad, pis_mask_pad], axis=1) # [N, K, M, JMAX]

			self.hall = hall # the full context tensor



		# time correlation c
		C = None
		"""
		# this cause too much memory
		if config.use_time_warp:
			with tf.variable_scope("time_correlation_c"):
				# hq [N, JQ, 2d]
				# hall [N, K, M, JMAX, 2d]

				hall_no_k = tf.reduce_mean(hall, 1) # [N, M, JMAX, 2d] # smaller tensor
				hall_t = tf.reshape(hall_no_k, [N, -1, 2*d]) # [N, T, 2d]
				T = tf.shape(hall_t)[-2]

				hall_t1 = tf.tile(tf.expand_dims(tf.expand_dims(hall_t, 2), 2), [1, 1, T, JQ, 1])
				hall_t2 = tf.tile(tf.expand_dims(tf.expand_dims(hall_t, 2), 2), [1, 1, T, JQ, 1])

				hq_t = tf.tile(tf.expand_dims(tf.expand_dims(hq, 1), 1), [1, T, T, 1, 1])

				#[N, T, T, JQ, 1]
				C_logits = linear(tf.concat([hall_t1*hall_t2, (hall_t1 - hall_t2)*(hall_t1- hall_t2), hall_t1*hq_t, hall_t2*hq_t, (hall_t1 - hq_t)*(hall_t1 - hq_t), (hall_t2 - hq_t)*(hall_t2 - hq_t)], 4), output_size=1, add_tanh=config.add_tanh, scope="C_logits")
				C_logits = tf.squeeze(C_logits, 4)
				#[N, T, T]
				C_logits = tf.reduce_max(C_logits, 3)

				# apply time warping function
				C = time_warp(C_logits, warp_type=config.warp_type, scope="time_warp_C")

				# get new context
				hall_tt = tf.tile(tf.expand_dims(tf.reshape(hall, [N, K, -1, 2*d]), 3), [1, 1, 1, T, 1]) #[N, K, T, T, 2d]
				C = tf.expand_dims(tf.tile(tf.expand_dims(C, 1), [1, K, 1, 1]), -1)# [N, K, T, T, 1]
				h_warped = tf.reduce_sum(hall_tt*C, -2) #[N, K, T, 2d]

				hall = tf.reshape(h_warped, [N, K, M, JMAX, 2*d])
		"""
		# smaller version
		if config.use_time_warp:
			with tf.variable_scope("time_warp"): 

				# get question vector first
				# hq [N, JQ, 2d]
				# hall [N, K, T, 2d] 
				c_hidden_size = 2*d

				# transpose Q 
				# [N, JQ, 2d] -> [N, 2d, JQ]
				#hq_t = tf.transpose(hq, [0, 2, 1])
				#print hq_t.get_shape() # need to know JQ

				# [N, 2d, JQ] -> [N, 2d, 1]
				#WQ = linear(hq_t, output_size=1, add_tanh=False, scope="WQ")
				#WQ = tf.squeeze(WQ, 2) #[N, 2d]

				WQ = lq

				#WQ = attention(hq,  tf.reshape(hall, [N, -1, 2*d]),  self.q_mask,  simiMatrix=config.simiMatrix,  wd=config.wd, add_tanh=False,  scope="question_att")

				hall_t = tf.reshape(hall, [N, K, -1, 2*d]) # [N, K, T, 2d]

				T = tf.shape(hall_t)[-2]

				# [N, K, T1, T2, 2d]
				hall_t1 = tf.tile(tf.expand_dims(hall_t, 3), [1, 1, 1, T, 1])
				hall_t2 = tf.tile(tf.expand_dims(hall_t, 3), [1, 1, 1, T, 1])

				# WQ [N, 2d] -> [N, K, T1, T2, 2d]
				WQ_t = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(WQ, 1), 1), 1), [1, K, T, T, 1])

				#w[h, h],  [N, K, T, T, 2d]
				WH = linear(tf.concat([hall_t1*hall_t2, (hall_t1-hall_t2)*(hall_t1-hall_t2)], 4), output_size=c_hidden_size, add_tanh=False, scope="WH")

				# [N, K, T, T, 1]
				C_logits = linear(WH+WQ_t, output_size=1, add_tanh=False, scope="WC")
				C_logits = tf.tanh(tf.reduce_sum(tf.squeeze(C_logits, 4), 1)) # [N, T, T]

				# for batch_size 1
				C_logits = tf.reshape(C_logits, [N, T, T])

				self.C = C_logits # correlation 

				# times different indication function
				C, win = time_indication_func(C_logits, warp_type=config.warp_type, scope="time_warp_C") # [N, T, T]
				self.C_win = win # the window size variable if the warp_type==5 past_future


				# get new context
				hall_tt = tf.tile(tf.expand_dims(hall_t, 3), [1, 1, 1, T, 1]) #[N, K, T, T, 2d]
				C_ex = tf.expand_dims(tf.tile(tf.expand_dims(C, 1), [1, K, 1, 1]), -1)# [N, K, T, T, 1]
				h_warped = tf.reduce_sum(hall_tt*C_ex, -2) #[N, K, T, 2d]

				hall = tf.reshape(h_warped, [N, K, M, JMAX, 2*d]) # warped context tensor
				#hall = exp_mask(hall, hall_mask)
				self.warp_h = hall




		# now from [N, M, 2d] -> [N, 2d]
		# attention layer
		with tf.variable_scope("attention"): # for baseline here has no attention


			# [N, 2d]
			g1_all, att_logits = attention_3d(hall, hq, hall_mask, self.q_mask, simiMatrix=config.simiMatrix, wd=config.wd, add_tanh=config.add_tanh, bidirect=config.use_bidirection, time_warp_att=config.use_time_warp_att, C=C, scope="all")

			self.att_logits = att_logits

			self.g1_all = g1_all

		


		with tf.variable_scope("choices_emb"):
			# embed the choices
			# hchoices -> [N, 4, JA, 2d]
			# gchoices -> [N, 4, 2d]
			# maybe attend something in the future
			#gchoices = tf.reduce_mean(hchoices, 2)# [N, 4, 2d]
			# hq -> [N, JQ, 2d]

			gchoices = lchoices #[N, 4, 2d] # last LSTM state for each choice

		with tf.variable_scope("question_emb"):
			# hq -> [N, JQ, 2d]
			# g1_all -> [N, 1, 2d]
			# gp -> [N, 2d]
			if config.use_question_att:
				gq, att_logits = attention(hq,  tf.expand_dims(g1_all, 1),  self.q_mask, tf.ones([N, 1], dtype=tf.bool),  add_tanh=config.add_tanh, simiMatrix=config.simiMatrix,  wd=config.wd, bidirect=config.use_bidirection,  scope="question_att")
				self.q_att_logits = att_logits
				if config.use_bidirection:
					gq = linear(gq, output_size=2*d, scope="bidrection_squash")
			else:
				gq = lq # this is the last hidden state of each question # [N, 2d]
			self.gq = gq

		# the modeling layer
		with tf.variable_scope("output"):
			
			# g1_a [N, 2d] # this could be viewed as an answer representation
			# together with the choices_emb and question_emb,  
			# we do a single layer multi class classification

			# tile g1_a [N, 2d] -> [N, 1, 2d] to concat with gchoices
			# [N, 4, 2d]
			g1_a_t = tf.tile(tf.expand_dims(g1_all, 1), [1, self.num_choice, 1])

			# tile gq for all choices
			gq = tf.tile(tf.expand_dims(gq, 1), [1, self.num_choice, 1]) # [N, 4, 2d]


			# [N, 4, 2d] -> [N*4, 1]
			# TODO: consider different similarity matrix
			# just g1 and gchoices
			#logits = linear(tf.concat([g1_a_t, gchoices, g1_a_t*gchoices], 2), output_size=1, wd=config.wd, scope="choicelogits")
			if config.use_eu_output:
				# eu similarity
				logits = linear(tf.concat([gq, g1_a_t, gchoices, g1_a_t*gchoices, gq*gchoices, (g1_a_t-gchoices)*(g1_a_t - gchoices), (gq-gchoices)*(gq-gchoices)], 2), add_tanh=config.add_tanh, output_size=1, scope="choicelogits")
			else:
				logits = linear(tf.concat([gq, g1_a_t, gchoices, g1_a_t*gchoices, gq*gchoices], 2), output_size=1, scope="choicelogits")
			

			logits = tf.squeeze(logits, 2) # [N, 4, 1] -> [N, 4]
			yp = tf.nn.softmax(logits) # [N, 4]

			# for loss and forward
			self.logits = logits
			self.yp = yp

	def build_loss(self):
		# logits -> [N, 4]
		# y -> [N, 4]
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, "float")) # [N] # softmax cross entropy loss.
		#
		losses = tf.reduce_mean(losses) # scalar,  avg loss of the whole batch

		tf.add_to_collection("losses", losses)

		# there might be l2 weight loss in some layer
		self.loss = tf.add_n(tf.get_collection("losses"), name="total_losses")
		#tf.summary.scalar(self.loss.op.name,  self.loss)

	# givng a batch of data,  construct the feed dict
	def get_feed_dict(self, batch, is_train=False):
		assert isinstance(batch, Dataset)
		# get the cap for each kind of step first
		config = self.config
		N = config.batch_size
		#N = len(batch.data['q'])
		if config.showspecs:
			N = 2
		M = config.max_num_albums
		#JX = config.max_sent_title_size
		JXA = config.max_sent_album_title_size
		JXP = config.max_sent_photo_title_size
		JD = config.max_sent_des_size
		JQ = config.max_question_size
		JI = config.max_num_photos
		JT = config.max_when_size
		JG = config.max_where_size
		JA = config.max_answer_size

		VW = config.word_vocab_size
		VC = config.char_vocab_size
		d = config.hidden_size
		W = config.max_word_size

		# This could make training faster
		# so each minibatch 's max length is different
		
		new_JXA = max(len(title) for sample in batch.data['album_title'] for title in sample)
		new_JXP = max([len(title) for sample in batch.data['photo_titles'] for album in sample for title in album]+[0])
		if new_JXA == 0: # empty??
			new_JXA = 1
		if new_JXP == 0: # empty??
			new_JXP = 1
		#JX = min(JX, new_JX) # so JX should be the longest sentence  in the batch,  but may not be the longest in the whole dataset
		JXA = min(JXA, new_JXA)
		JXP = min(JXP, new_JXP)

		new_JD = max(len(des) for sample in batch.data['album_description'] for des in sample)
		if new_JD == 0: # empty??
			new_JD = 1
		JD = min(JD, new_JD)

		new_JG = max(len(where) for sample in batch.data['where'] for where in sample)
		if new_JG == 0: # could be empty
			new_JG = 1
		JG = min(JG, new_JG)

		new_JT = max(len(when) for sample in batch.data['when'] for when in sample)
		if new_JT == 0: # empty??
			new_JT = 1
		JT = min(JT, new_JT)

		new_JI = max(len(album) for sample in batch.data['photo_ids'] for album in sample)
		if new_JI == 0: # empty??
			new_JI = 1
		JI = min(JI, new_JI)

		new_JQ = max(len(ques) for ques in batch.data['q'])
		if(new_JQ == 0):
			new_JQ = 1
		JQ = min(JQ, new_JQ)


		new_M = max(len(onesample) for onesample in batch.data['album_title'])
		if(new_M == 0):
			new_M = 1
		M = min(M, new_M)

		feed_dict = {}

		# initial all the placeholder
		# all words initial is 0 ,  means -NULL- token
		at = np.zeros([N, M, JXA], dtype='int32')
		at_c = np.zeros([N, M, JXA, W], dtype="int32")
		at_mask = np.zeros([N, M, JXA], dtype="bool")

		ad = np.zeros([N, M, JD], dtype='int32')
		ad_c = np.zeros([N, M, JD, W], dtype="int32")
		ad_mask = np.zeros([N, M, JD], dtype="bool")

		when = np.zeros([N, M, JT], dtype='int32')
		when_c = np.zeros([N, M, JT, W], dtype="int32")
		when_mask = np.zeros([N, M, JT], dtype="bool")

		where = np.zeros([N, M, JG], dtype='int32')
		where_c = np.zeros([N, M, JG, W], dtype="int32")
		where_mask = np.zeros([N, M, JG], dtype="bool")

		pts = np.zeros([N, M, JI, JXP], dtype="int32")
		pts_c = np.zeros([N, M, JI, JXP, W], dtype="int32")
		pts_mask = np.zeros([N, M, JI, JXP], dtype="bool")

		pis = np.zeros([N, M, JI], dtype='int32')
		pis_mask = np.zeros([N, M, JI], dtype="bool")

		q = np.zeros([N, JQ], dtype='int32')
		q_c = np.zeros([N, JQ, W], dtype="int32")
		q_mask = np.zeros([N, JQ], dtype="bool")

		choices = np.zeros([N, self.num_choice, JA], dtype='int32')
		choices_c = np.zeros([N, self.num_choice, JA, W], dtype="int32")
		choices_mask = np.zeros([N, self.num_choice, JA], dtype="bool")


		# link the feed_dict
		feed_dict[self.at] = at
		feed_dict[self.at_c] = at_c
		feed_dict[self.at_mask] = at_mask

		feed_dict[self.ad] = ad
		feed_dict[self.ad_c] = ad_c
		feed_dict[self.ad_mask] = ad_mask

		feed_dict[self.when] = when
		feed_dict[self.when_c] = when_c
		feed_dict[self.when_mask] = when_mask

		feed_dict[self.where] = where
		feed_dict[self.where_c] = where_c
		feed_dict[self.where_mask] = where_mask

		feed_dict[self.pts] = pts
		feed_dict[self.pts_c] = pts_c
		feed_dict[self.pts_mask] = pts_mask

		feed_dict[self.pis] = pis
		feed_dict[self.pis_mask] = pis_mask

		feed_dict[self.q] = q
		feed_dict[self.q_c] = q_c
		feed_dict[self.q_mask] = q_mask

		feed_dict[self.choices] = choices
		feed_dict[self.choices_c] = choices_c
		feed_dict[self.choices_mask] = choices_mask

		feed_dict[self.is_train] = is_train

		# image feat mat and word mat
		feed_dict[self.image_emb_mat] = batch.data['pidx2feat']
		feed_dict[self.existing_emb_mat] = batch.shared['existing_emb_mat']


		# question and choices
		Q = batch.data['q']
		Q_c = batch.data['cq'] 

		C = deepcopy(batch.data['cs']) # for the choice,  since we will add correct answer into it,  we copy so it won't affect other batch
		C_c = deepcopy(batch.data['ccs'])

		# data
		AT = batch.data['album_title']
		AT_c = batch.data['album_title_c']
		AD = batch.data['album_description']
		AD_c = batch.data['album_description_c']
		WHERE = batch.data['where']
		WHERE_c = batch.data['where_c']
		WHEN = batch.data['when']
		WHEN_c = batch.data['when_c']

		PT = batch.data['photo_titles']
		PT_c = batch.data['photo_titles_c']

		PI = batch.data['photo_idxs']

		# for training,  one of the y will be in the choices
		# only training feed the y
		if is_train:
			Y = batch.data['y']
			Y_c = batch.data['cy']

			y = np.zeros([N, self.num_choice], dtype="bool")
			feed_dict[self.y] = y

			# decide the index of correct choice first,  we randomly decide it
			correctIndex = np.random.choice(self.num_choice, N) # get a array of size [N]


			#for i in xrange(N): # some batch will be smaller
			for i in xrange(len(batch.data['y'])):
				y[i, correctIndex[i]] = True
				# put the answer into the choices
				assert len(C[i]) == (self.num_choice - 1)
				C[i].insert(correctIndex[i], Y[i])
				C_c[i].insert(correctIndex[i], Y_c[i])
				assert len(batch.data['cs'][i]) == (self.num_choice - 1)

			# for debug
			if config.showspecs:
				print "first two batch's answer:%s ,  char:%s,  correctIdx:%s"%(Y[:2], Y_c[:2], y[:2])
				print "first two batch's choices:%s ,  char:%s"%(C[:2], C_c[:2])

		else:
			# for testing,  put the answer into the original idx if there is any
			if(batch.data.has_key("y") and batch.data.has_key("cy") and batch.data.has_key('yidx')):
				Y = batch.data['y']
				Y_c = batch.data['cy']
				Y_idx = batch.data['yidx']
				#for i in xrange(N): # some batch will be smaller
				for i in xrange(len(batch.data['y'])):
					#print i, len(C[i])
					assert len(C[i]) == (self.num_choice - 1),  ("C[i] len:%s, %s, Y:%s"%(len(C[i]), C[i], Y[i]))
					C[i].insert(Y_idx[i], Y[i])
					C_c[i].insert(Y_idx[i], Y_c[i])
					

			# will check choice num in the end

		# the photo idx is simple

		for i, pi in enumerate(PI):
			# one batch
			for j, pij in enumerate(pi):
				# one album
				if j == config.max_num_albums:
					break
				for k, pijk in enumerate(pij):
					if k == config.max_num_photos:
						break
					#print pijk
					assert isinstance(pijk, int)
					pis[i, j, k] = pijk
					pis_mask[i, j, k] = True



		def get_word(word):
			d = batch.shared['word2idx'] # this is for the word not in glove
			for each in (word,  word.lower(),  word.capitalize(),  word.upper()):
				if each in d:
					return d[each]
			# the word in glove
			
			d2 = batch.shared['existing_word2idx']
			for each in (word,  word.lower(),  word.capitalize(),  word.upper()):
				if each in d2:
					return d2[each] + len(d) # all idx + len(the word to train)
			return 1 # 1 is the -UNK-

		def get_char(char):
			d = batch.shared['char2idx']
			if char in d:
				return d[char]
			return 1

		# for all the text,  get each word's index.
		# album title
		for i,  ati in enumerate(AT): # batch_sizes
			# one batch
			for j, atij in enumerate(ati):
				# one album
				if j == config.max_num_albums:
					break
				for k, atijk in enumerate(atij):
					# each word
					if k == config.max_sent_album_title_size:
						break
					wordIdx = get_word(atijk)
					at[i, j, k] = wordIdx
					at_mask[i, j, k] = True

		for i,  cati in enumerate(AT_c):
			# one batch
			for j,  catij in enumerate(cati):
				if j == config.max_num_albums:
					break
				for k,  catijk in enumerate(catij):
					# each word
					if k == config.max_sent_album_title_size:
						break
					for l, catijkl in enumerate(catijk):
						if l == config.max_word_size:
							break
						at_c[i, j, k, l] = get_char(catijkl)



		# album description
		for i,  adi in enumerate(AD): # batch_sizes
			# one batch
			for j, adij in enumerate(adi):
				# one album
				if j == config.max_num_albums:
					break
				for k, adijk in enumerate(adij):
					# each word
					if k == config.max_sent_des_size:
						break
					wordIdx = get_word(adijk)
					ad[i, j, k] = wordIdx
					ad_mask[i, j, k] = True

		for i,  cadi in enumerate(AD_c):
			# one batch
			for j,  cadij in enumerate(cadi):
				if j == config.max_num_albums:
					break
				for k,  cadijk in enumerate(cadij):
					# each word
					if k == config.max_sent_des_size:
						break
					for l, cadijkl in enumerate(cadijk):
						if l == config.max_word_size:
							break
						ad_c[i, j, k, l] = get_char(cadijkl)




		# album when
		for i,  wi in enumerate(WHEN): # batch_sizes
			# one batch
			for j, wij in enumerate(wi):
				# one album
				if j == config.max_num_albums:
					break
				for k, wijk in enumerate(wij):
					# each word
					if k == config.max_when_size:
						break
					wordIdx = get_word(wijk)
					when[i, j, k] = wordIdx
					when_mask[i, j, k] = True

		for i,  cwi in enumerate(WHEN_c):
			# one batch
			for j,  cwij in enumerate(cwi):
				if j == config.max_num_albums:
					break
				for k,  cwijk in enumerate(cwij):
					# each word
					if k == config.max_when_size:
						break
					for l, cwijkl in enumerate(cwijk):
						if l == config.max_word_size:
							break
						when_c[i, j, k, l] = get_char(cwijkl)

		# album where
		for i,  wi in enumerate(WHERE): # batch_sizes
			# one batch
			for j, wij in enumerate(wi):
				# one album
				if j == config.max_num_albums:
					break
				for k, wijk in enumerate(wij):
					# each word
					if k == config.max_where_size:
						break
					wordIdx = get_word(wijk)
					where[i, j, k] = wordIdx
					where_mask[i, j, k] = True

		for i,  cwi in enumerate(WHERE_c):
			# one batch
			for j,  cwij in enumerate(cwi):
				if j == config.max_num_albums:
					break
				for k,  cwijk in enumerate(cwij):
					# each word
					if k == config.max_where_size:
						break
					for l, cwijkl in enumerate(cwijk):
						if l == config.max_word_size:
							break
						where_c[i, j, k, l] = get_char(cwijkl)


		# photo title
		for i,  pti in enumerate(PT): # batch_sizes
			# one batch
			for j, ptij in enumerate(pti):
				# one album
				if j == config.max_num_albums:
					break
				for k, ptijk in enumerate(ptij):
					# each photo
					if k == config.max_num_photos:
						break
					for l, ptijkl in enumerate(ptijk):
						if l == config.max_sent_photo_title_size:
							break
						# each word
						wordIdx = get_word(ptijkl)
						pts[i, j, k, l] = wordIdx
						pts_mask[i, j, k, l] = True
		for i,  pti in enumerate(PT_c): # batch_sizes
			# one batch
			for j, ptij in enumerate(pti):
				# one album
				if j == config.max_num_albums:
					break
				for k, ptijk in enumerate(ptij):
					# each photo
					if k == config.max_num_photos:
						break
					for l, ptijkl in enumerate(ptijk):
						if l == config.max_sent_photo_title_size:
							break
						# each word
						for o,  ptijklo in enumerate(ptijkl):
							# each char
							if o == config.max_word_size:
								break
							pts_c[i, j, k, l, o] = get_char(ptijklo)



		# answer choices

		for i, ci in enumerate(C):
			# one batch
			assert len(ci) == self.num_choice
			for j, cij in enumerate(ci):
				# one answer
				for k, cijk in enumerate(cij):
					# one word
					if k == config.max_answer_size:
						break
					wordIdx = get_word(cijk)
					choices[i, j, k] = wordIdx
					choices_mask[i, j, k] = True
		for i, ci in enumerate(C_c):
			# one batch
			assert len(ci) == self.num_choice,  (len(ci))
			for j, cij in enumerate(ci):
				# one answer
				for k, cijk in enumerate(cij):
					# one word
					if k == config.max_answer_size:
						break
					for l, cijkl in enumerate(cijk):
						if l == config.max_word_size:
							break
						choices_c[i, j, k, l] = get_char(cijkl)


		# loa the question
		# no limiting on the question word length
		for i,  qi in enumerate(Q):
			# one batch
			for j,  qij in enumerate(qi):
				q[i,  j] = get_word(qij)
				q_mask[i,  j] = True

		# load the question char
		for i,  cqi in enumerate(Q_c):
			for j,  cqij in enumerate(cqi):
				for k,  cqijk in enumerate(cqij):
					
					if k == config.max_word_size:
						break
					q_c[i,  j,  k] = get_char(cqijk)
		"""
		print q
		print choices
		print when
		print where
		#print pts_c
		print pis
		print N, M, JI, JXP, JXA, JD, JT, JD, W, JA
		sys.exit()
		"""
		

		#print feed_dict
		return feed_dict


