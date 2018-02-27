# coding=utf-8
# tensorflow model graph 


import tensorflow as tf
from utils import flatten,reconstruct,Dataset,exp_mask
import numpy as np
import random,sys
from attention_gru_cell import AttentionGRUCell

VERY_NEGATIVE_NUMBER = -1e30

def get_model(config):
	# implement a multi gpu model?
	with tf.name_scope(config.modelname), tf.device("/gpu:0"):
		model = Model(config,"model_%s"%config.modelname)

	return model


from copy import deepcopy # for C[i].insert(Y[i])


# a flatten and reconstruct version of softmax
def softmax(logits,scope=None):
	with tf.name_scope(scope or "softmax"): # noted here is name_scope not variable
		flat_logits = flatten(logits,1)
		flat_out = tf.nn.softmax(flat_logits)
		out = reconstruct(flat_out,logits,1)
		return out



# add current scope's variable's l2 loss to loss collection
def add_wd(wd,scope=None):
	if wd != 0.0:
		scope = scope or tf.get_variable_scope().name
		vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		with tf.variable_scope("weight_decay"):
			for var in vars_:
				weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="%s/wd"%(var.op.name))
				tf.add_to_collection("losses",weight_decay)

# modified from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
	"""Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
	
	encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
	ls = sentence_size+1
	le = embedding_size+1
	for i in range(1, le):
		for j in range(1, ls):
			encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
	encoding = 1 + 4 * encoding / embedding_size / sentence_size
	return np.transpose(encoding)


# fully-connected layer
# simple linear layer, without activatation # remember to add it
# [N,M,JX,JQ,2d] => x[N*M*JX*JQ,2d] * W[2d,output_size] -> 
def linear(x,output_size,scope,add_tanh=False,wd=None):
	with tf.variable_scope(scope):
		# since the input here is not two rank, we flat the input while keeping the last dims
		keep = 1
		#print x.get_shape().as_list()
		flat_x = flatten(x,keep) # keeping the last one dim # [N,M,JX,JQ,2d] => [N*M*JX*JQ,2d]
		#print flat_x.get_shape() # (?, 200) # wd+cwd
		bias_start = 0.0
		if not (type(output_size) == type(1)): # need to be get_shape()[k].value
			output_size = output_size.value

		#print [flat_x.get_shape()[-1],output_size]

		W = tf.get_variable("W",dtype="float",initializer=tf.truncated_normal([flat_x.get_shape()[-1].value,output_size],stddev=0.1))
		bias = tf.get_variable("b",dtype="float",initializer=tf.constant(bias_start,shape=[output_size]))
		flat_out = tf.matmul(flat_x,W)+bias

		if add_tanh:
			flat_out = tf.tanh(flat_out,name="tanh")


		if wd is not None:
			add_wd(wd)

		out = reconstruct(flat_out,x,keep)
		return out

# from https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow
def _get_attention(q_vec, prev_memory, fact_vec, reuse,hidden_size):
	"""Use question vector and previous memory to create scalar attention for current fact"""
	with tf.variable_scope("attention", reuse=reuse):

		features = [fact_vec*q_vec,
					fact_vec*prev_memory,
					tf.abs(fact_vec - q_vec),
					tf.abs(fact_vec - prev_memory)]

		feature_vec = tf.concat(features, 1)

		attention = tf.contrib.layers.fully_connected(feature_vec,
						#self.config.embed_size,
						hidden_size,
						activation_fn=tf.nn.tanh,
						reuse=reuse, scope="fc1")
	
		attention = tf.contrib.layers.fully_connected(attention,
						1,
						activation_fn=None,
						reuse=reuse, scope="fc2")
		
	return attention
# from https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow
def _generate_episode(memory, q_vec, fact_vecs, fact_vecs_length,hop_index,hidden_size):
	"""Generate episode by applying attention to current fact vectors through a modified GRU"""

	attentions = [tf.squeeze(
		_get_attention(q_vec, memory, fv, bool(hop_index) or bool(i),hidden_size), axis=1)
		for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

	attentions = tf.transpose(tf.stack(attentions))
	attentions = tf.nn.softmax(attentions)
	attentions = tf.expand_dims(attentions, axis=-1)

	reuse = True if hop_index > 0 else False
	
	# concatenate fact vectors and attentions for input into attGRU
	gru_inputs = tf.concat([fact_vecs, attentions], 2)

	with tf.variable_scope('attention_gru', reuse=reuse):
		_, episode = tf.nn.dynamic_rnn(AttentionGRUCell(hidden_size),
				gru_inputs,
				dtype=np.float32,
				sequence_length=fact_vecs_length
		)

	return episode

def get_initializer(matrix):
	def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
	return _initializer

class Model():
	def __init__(self,config,scope):
		self.scope = scope
		self.config = config
		# a step var to keep track of current training process
		self.global_step = tf.get_variable('global_step',shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False) # a counter

		# get all the dimension here
		N = self.N = config.batch_size
		
		VW = self.VW = config.word_vocab_size
		VC = self.VC = config.char_vocab_size
		W = self.W = config.max_word_size

		# embedding dim
		self.cd,self.wd,self.cwd = config.char_emb_size,config.word_emb_size,config.char_out_size

		# image dimension
		self.idim = config.image_feat_dim

		self.num_choice = 4

		# step limits
		# M -> album max num
		# -----JX -> title max words (album title,photo title)
		# JXA -> album title max words
		# JXP -> photo title max words
		# JD -> album description max word
		# JT -> album when max word
		# JG -> album where max word

		# JI -> album max photo

		# JA -> max answer (choice) length
		# JQ -> max question length

		# all the inputs
		M = config.max_num_albums

		# album title
		# [N,M,JXA]
		self.at = tf.placeholder('int32',[N,M,None],name="at")
		self.at_c = tf.placeholder("int32",[N,M,None,W],name="at_c")
		self.at_mask = tf.placeholder("bool",[N,M,None],name="at_mask") # to get the sequence length

		# album description
		# [N,M,JD]
		self.ad = tf.placeholder('int32',[N,M,None],name="ad")
		self.ad_c = tf.placeholder("int32",[N,M,None,W],name="ad_c")
		self.ad_mask = tf.placeholder("bool",[N,M,None],name="ad_mask")

		# album when, where
		# [N,M,JT/JG]
		self.when = tf.placeholder("int32",[N,M,None],name="when")
		self.when_c = tf.placeholder("int32",[N,M,None,W],name="when_c")
		self.when_mask = tf.placeholder("bool",[N,M,None],name="when_mask")
		self.where = tf.placeholder("int32",[N,M,None],name="where")
		self.where_c = tf.placeholder("int32",[N,M,None,W],name="where_c")
		self.where_mask = tf.placeholder("bool",[N,M,None],name="where_mask")

		# photo titles
		# [N,M,JI,JXP]
		self.pts = tf.placeholder('int32',[N,M,None,None],name="pts")
		self.pts_c = tf.placeholder("int32",[N,M,None,None,W],name="pts_c")
		self.pts_mask = tf.placeholder("bool",[N,M,None,None],name="pts_mask")

		# photo
		# [N,M,JI] # each is a photo index
		self.pis = tf.placeholder('int32',[N,M,None],name="pis")
		self.pis_mask = tf.placeholder("bool",[N,M,None],name="pis_mask")

		# question
		self.q = tf.placeholder('int32',[N,None],name="q")
		self.q_c = tf.placeholder('int32', [N, None, W], name='q_c')
		self.q_mask = tf.placeholder("bool",[N,None],name="q_mask")

		# answer + choice words
		# [N,4,JA]
		self.choices = tf.placeholder("int32",[N,self.num_choice,None],name="choices")
		self.choices_c = tf.placeholder("int32",[N,self.num_choice,None,W],name="choices_c")
		self.choices_mask = tf.placeholder("bool",[N,self.num_choice,None],name="choices_mask")

		# 4 choice classification
		self.y = tf.placeholder('bool', [N, self.num_choice], name='y')
		

		# feed in the pretrain word vectors for all batch
		self.existing_emb_mat = tf.placeholder('float',[None,config.word_emb_size],name="pre_emb_mat")

		# feed in the image feature for this batch
		# [photoNumForThisBatch,image_dim]
		self.image_emb_mat = tf.placeholder("float",[None,config.image_feat_dim],name="image_emb_mat")

		# used for drop out switch
		self.is_train = tf.placeholder('bool', [], name='is_train')

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

		# dynamic decide some step, for sequence length
		#M = tf.shape(self.at)[1] # photo num
		M = config.max_num_albums
		JI = config.max_num_photos

		JXA = tf.shape(self.at)[2] # for album title, photo title
		JD = tf.shape(self.ad)[2] # description length
		JT = tf.shape(self.when)[2]
		JG = tf.shape(self.where)[2]

		#JI = tf.shape(self.pis)[2] # used for photo_title, photo
		JXP = tf.shape(self.pts)[3]

		JQ = tf.shape(self.q)[1]
		JA = tf.shape(self.choices)[2]

		# embeding size
		cdim,wdim,cwdim = self.cd,self.wd,self.cwd #cwd: char -> word output dimension
		# image feature dim
		idim = self.idim # image_feat dimension

		# all input:
		#	at, ad, when, where, 
		#	pts, pis
		#	q, choices

		# embedding
		with tf.variable_scope('emb'):
			
			# word stuff
			with tf.variable_scope('word'):
				with tf.variable_scope("var"):
					# get the word embedding for new words
					if config.is_train:
						# for new word
						word_emb_mat = tf.get_variable("word_emb_mat",dtype="float",shape=[VW,wdim],initializer=get_initializer(config.emb_mat)) # it's just random initialized
					else: # save time for loading the emb during test
						word_emb_mat = tf.get_variable("word_emb_mat",dtype="float",shape=[VW,wdim])
					# concat with pretrain vector
					# so 0 - VW-1 index for new words, the rest for pretrain vector
					# and the pretrain vector is fixed
					word_emb_mat = tf.concat([word_emb_mat,self.existing_emb_mat],0)

				#[N,M,JXA] -> [N,M,JXA,wdim]
				Aat = tf.nn.embedding_lookup(word_emb_mat,self.at)
				Aad = tf.nn.embedding_lookup(word_emb_mat,self.ad)
				Awhen = tf.nn.embedding_lookup(word_emb_mat,self.when)
				Awhere = tf.nn.embedding_lookup(word_emb_mat,self.where)
				Apts = tf.nn.embedding_lookup(word_emb_mat,self.pts)

				Aq = tf.nn.embedding_lookup(word_emb_mat,self.q)
				Achoices = tf.nn.embedding_lookup(word_emb_mat,self.choices)

		
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

				# [N,M,JI] -> [N,M,JI,idim]
				xpis = tf.nn.embedding_lookup(self.image_emb_mat,self.pis)

				# use image trans, then linearly transform it to lower dim
				# TODO: CNN transform?
				if config.use_image_trans:
					
					with tf.variable_scope("image_transform"):
						#[N,M,JI,idim] -> [N,M,JI,newdim]
						xpis = linear(xpis,add_tanh=config.add_tanh,output_size=config.image_trans_dim,wd=config.wd,scope="image_trans_linear")
						#xpis = tf.nn.relu(xpis)



		d = config.hidden_size

		# LSTM / GRU?
		#cell_img = tf.nn.rnn_cell.BasicLSTMCell(d,state_is_tuple=True)
		cell_text = tf.nn.rnn_cell.GRUCell(d)
		cell_img = tf.nn.rnn_cell.GRUCell(d)

		# sequence length for each
		at_len = tf.reduce_sum(tf.cast(self.at_mask,"int32"),2) # [N,M] # each album's title length
		ad_len = tf.reduce_sum(tf.cast(self.ad_mask,"int32"),2)
		when_len = tf.reduce_sum(tf.cast(self.when_mask,"int32"),2)
		where_len = tf.reduce_sum(tf.cast(self.where_mask,"int32"),2) # [N,M]

		pis_len = tf.reduce_sum(tf.cast(self.pis_mask,"int32"),2) #[N,M,JI] #[N,M]

		pts_len = tf.reduce_sum(tf.cast(self.pts_mask,"int32"),3) # [N,M,JI,JXP] -> [N,M,JI]

		q_len = tf.reduce_sum(tf.cast(self.q_mask,"int32"),1) # [N] # each question 's length

		choices_len = tf.reduce_sum(tf.cast(self.choices_mask,"int32"),2) # [N,4]

		# xat -> [N,M,JXA,wdim]
		# xad -> [N,M,JD,wdim]
		# xwhen/xwhere -> [N,M,JT/JG,wdim]
		# xpts -> [N,M,JI,JXP,wdim]

		# xpis -> [N,M,JI,idim]

		# qq -> [N,JQ,wdim]
		# qchoices -> [N,4,JA,wdim]

		# use positional encoder to get sentence representation
		# from [N,M,JI,JX] -> [N,M,2d]
		with tf.variable_scope("reader"):
			with tf.variable_scope("text"):
				# question use a GRU
				
				_,lq = tf.nn.dynamic_rnn(cell_text,qq,sequence_length=q_len,dtype="float",scope="utext")

				tf.get_variable_scope().reuse_variables()

				# position encoding is not working
				# use GRU

				# GRU input
				# flat all
				# choices
				flat_qchoices = flatten(qchoices,2) # [N,4,JA,dim] -> [N*4,JA,dim]
				# album title
				flat_xat = flatten(xat,2) #[N,M,JXA,dim] -> [N*M,JXA,dim]
				flat_xad = flatten(xad,2)
				flat_xwhen = flatten(xwhen,2)
				flat_xwhere = flatten(xwhere,2)
				
				#print "flat_xpis shape:%s"%(flat_xpis.get_shape())

				# photo tiles
				flat_xpts = flatten(xpts,2) # [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]
				#print "flat_xpts shape:%s"%(flat_xpts.get_shape())

				# get the sequence length, all one dim
				flat_qchoices_len = flatten(choices_len,0) # [N*4]
				flat_xat_len = flatten(at_len,0) # [N*M]
				flat_xad_len = flatten(ad_len,0) # [N*M]
				flat_xwhen_len = flatten(when_len,0) # [N*M]
				flat_xwhere_len = flatten(where_len,0) # [N*M]
				
				flat_xpts_len = flatten(pts_len,0) # [N*M*JI]

				# album title
				_,lat_flat = tf.nn.dynamic_rnn(cell_text,flat_xat,sequence_length=flat_xat_len,dtype="float",scope="utext")
				lat = tf.reshape(lat_flat,[N,M,d])
				# description
				_,lad_flat = tf.nn.dynamic_rnn(cell_text,flat_xad,sequence_length=flat_xad_len,dtype="float",scope="utext")
				lad = tf.reshape(lad_flat,[N,M,d])
				# when
				_,lwhen_flat = tf.nn.dynamic_rnn(cell_text,flat_xwhen,sequence_length=flat_xwhen_len,dtype="float",scope="utext")
				lwhen = tf.reshape(lwhen_flat,[N,M,d])
				#where
				_,lwhere_flat = tf.nn.dynamic_rnn(cell_text,flat_xwhere,sequence_length=flat_xwhere_len,dtype="float",scope="utext")
				lwhere = tf.reshape(lwhere_flat,[N,M,d])
				
				# photo title
				_,lpts_flat = tf.nn.dynamic_rnn(cell_text,flat_xpts,sequence_length=flat_xpts_len,dtype="float",scope="uimage")
				#lpts = tf.reduce_mean(tf.reshape(lpts_flat,[N,M,JI,d]),2)
				#lpts = tf.reshape(lpts_flat,[N,M,JI,d])
				lpts = tf.reshape(lpts_flat,[N,M,-1,d])

				#choices
				_,lchoices_flat = tf.nn.dynamic_rnn(cell_text,flat_qchoices,sequence_length=flat_qchoices_len,dtype="float",scope="uimage")
				lchoices = tf.reshape(lchoices_flat,[N,-1,d]) # [N,4,d]
				

			with tf.variable_scope("image"):
				# first, transform image into text space with tanh activation
				xpis = linear(xpis,add_tanh=True,output_size=wdim,wd=config.wd,scope="image_trans_linear") # [N,M,JI,wdim]

				flat_xpis = flatten(xpis,2) # [N,M,JI,wdim] -> [N*M,JI,wdim]
				flat_xpis_len = flatten(pis_len,0) # [N*M]

				hpis_flat,lpis_flat = tf.nn.dynamic_rnn(cell_img,flat_xpis,sequence_length=flat_xpis_len,dtype="float",scope="uimage")
				#hpis = tf.reshape(hpis_flat,[N,M,JI,d])
				hpis = tf.reshape(hpis_flat,[N,M,-1,d])
				lpis = tf.reshape(lpis_flat,[N,M,d])
				

		# all rnn output

		# encoded:
		# lq -> [N,wdim]

		# lat -> [N,M,wdim]
		# lad -> [N,M,wdim]
		# lwhen -> [N,M,wdim]
		# lwhere -> [N,M,wdim]
		# lpts -> [N,M,JI,d]
		# lpis -> [N,M,wdim]
		# hpis -> [N,M,JI,d]

		# lchoices -> [N,4,d]
		with tf.variable_scope("input_facts"):
			# stack them
			K = 4
			#f_in = tf.stack([lat,lad,lwhen,lwhere,lpts,lpis],axis=2) # [N,M,K,wdim]
			f_in = tf.stack([lat,lad,lwhen,lwhere],axis=2) # [N,M,K,d]
			f_in = tf.concat([f_in,lpts,hpis],2) # [N,M,K+2*JI,d]
			
			f_in = tf.reshape(f_in,[N,M*(K+2*JI),d]) # need JI to be know in generate_eposide
			#f_in = tf.reshape(f_in,[N,-1,d])

			cell_facts_fw = tf.nn.rnn_cell.GRUCell(d)
			cell_facts_bw = tf.nn.rnn_cell.GRUCell(d)
			
			dynamic_M = tf.shape(self.pis)[1]
			dynamic_JI = tf.shape(self.pis)[2]
			facts_length = tf.tile(tf.expand_dims(dynamic_M*(K+2*dynamic_JI),0),[N])

			# f_in -> [N,M*K,d]
			facts, _ = tf.nn.bidirectional_dynamic_rnn(
				cell_facts_fw,
				cell_facts_bw,
				f_in,
				dtype=np.float32,
				sequence_length=facts_length
			)
			# add f_fw and f_bw
			facts = tf.reduce_sum(tf.stack(facts), axis=0) 

			# add dropout
			keep_prob = tf.cond(self.is_train,lambda:tf.constant(config.keep_prob),lambda:tf.constant(1.0))

			facts = tf.nn.dropout(facts,keep_prob)


		with tf.variable_scope("question_emb"):
			gq = lq # this is the last hidden state of each question # [N,d]

		with tf.variable_scope("choices_emb"):

			gchoices = lchoices #[N,4,d] # last LSTM state for each choice

		# gq -> [N,d]
		# gchoices -> [N,4,d]
		# facts -> [N,M*K,d]
		# from https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow
		with tf.variable_scope("memory"):
			prev_memory = gq
			for i in xrange(self.config.dmnplus_num_hops):
				episode = _generate_episode(prev_memory,gq,facts,facts_length,i,d) # [N,d]

				# update memory
				with tf.variable_scope("hop_%d" % i):
					prev_memory = tf.layers.dense(
							tf.concat([prev_memory, episode, gq], 1),
							d,
							activation=tf.nn.relu) #[N,d]
			output = prev_memory

			# add dropout again
			keep_prob = tf.cond(self.is_train,lambda:tf.constant(config.keep_prob),lambda:tf.constant(1.0))

			output = tf.nn.dropout(output,keep_prob)

		# the modeling layer
		with tf.variable_scope("output"):
			
			
			# [N,4,d]
			c_output = tf.tile(tf.expand_dims(output,1),[1,self.num_choice,1])

			# tile gq for all choices
			c_gq = tf.tile(tf.expand_dims(gq,1),[1,self.num_choice,1]) # [N,4,2d]


			logits = linear(tf.concat([c_gq,c_output,gchoices],2),output_size=1,add_tanh=False,scope="choicelogits")
			

			logits = tf.squeeze(logits,2) # [N,4,1] -> [N,4]
			yp = tf.nn.softmax(logits) # [N,4]

			# for loss and forward
			self.logits = logits
			self.yp = yp

	def build_loss(self):
		# logits -> [N,4]
		# y -> [N,4]
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=tf.cast(self.y,"float")) # [N] # softmax cross entropy loss.
		#
		losses = tf.reduce_mean(losses) # scalar, avg loss of the whole batch

		tf.add_to_collection("losses",losses)

		# add l2 regularization for all variables except biases
		l2_loss = 0.0
		if self.config.wd is not None:
			for v in tf.trainable_variables():
				if not 'bias' in v.name.lower():
					l2_loss += self.config.wd*tf.nn.l2_loss(v)

		tf.add_to_collection("losses",l2_loss)	

		# there might be l2 weight loss in some layer
		self.loss = tf.add_n(tf.get_collection("losses"),name="total_losses")
		#tf.summary.scalar(self.loss.op.name, self.loss)


	# givng a batch of data, construct the feed dict
	def get_feed_dict(self,batch,is_train=False):
		assert isinstance(batch,Dataset)
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
		#JX = min(JX,new_JX) # so JX should be the longest sentence  in the batch, but may not be the longest in the whole dataset
		JXA = min(JXA,new_JXA)
		JXP = min(JXP,new_JXP)

		new_JD = max(len(des) for sample in batch.data['album_description'] for des in sample)
		if new_JD == 0: # empty??
			new_JD = 1
		JD = min(JD,new_JD)

		new_JG = max(len(where) for sample in batch.data['where'] for where in sample)
		if new_JG == 0: # could be empty
			new_JG = 1
		JG = min(JG,new_JG)

		new_JT = max(len(when) for sample in batch.data['when'] for when in sample)
		if new_JT == 0: # empty??
			new_JT = 1
		JT = min(JT,new_JT)

		

		new_JQ = max(len(ques) for ques in batch.data['q'])
		if(new_JQ == 0):
			new_JQ = 1
		JQ = min(JQ,new_JQ)

		
		
		feed_dict = {}

		# initial all the placeholder
		# all words initial is 0 , means -NULL- token
		at = np.zeros([N,M,JXA],dtype='int32')
		at_c = np.zeros([N,M,JXA,W],dtype="int32")
		at_mask = np.zeros([N,M,JXA],dtype="bool")

		ad = np.zeros([N,M,JD],dtype='int32')
		ad_c = np.zeros([N,M,JD,W],dtype="int32")
		ad_mask = np.zeros([N,M,JD],dtype="bool")

		when = np.zeros([N,M,JT],dtype='int32')
		when_c = np.zeros([N,M,JT,W],dtype="int32")
		when_mask = np.zeros([N,M,JT],dtype="bool")

		where = np.zeros([N,M,JG],dtype='int32')
		where_c = np.zeros([N,M,JG,W],dtype="int32")
		where_mask = np.zeros([N,M,JG],dtype="bool")

		pts = np.zeros([N,M,JI,JXP],dtype="int32")
		pts_c = np.zeros([N,M,JI,JXP,W],dtype="int32")
		pts_mask = np.zeros([N,M,JI,JXP],dtype="bool")

		pis = np.zeros([N,M,JI],dtype='int32')
		pis_mask = np.zeros([N,M,JI],dtype="bool")

		q = np.zeros([N,JQ],dtype='int32')
		q_c = np.zeros([N,JQ,W],dtype="int32")
		q_mask = np.zeros([N,JQ],dtype="bool")

		choices = np.zeros([N,self.num_choice,JA],dtype='int32')
		choices_c = np.zeros([N,self.num_choice,JA,W],dtype="int32")
		choices_mask = np.zeros([N,self.num_choice,JA],dtype="bool")


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

		C = deepcopy(batch.data['cs']) # for the choice, since we will add correct answer into it, we copy so it won't affect other batch
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

		# for training, one of the y will be in the choices
		# only training feed the y
		if is_train:
			Y = batch.data['y']
			Y_c = batch.data['cy']

			y = np.zeros([N,self.num_choice],dtype="bool")
			feed_dict[self.y] = y

			# decide the index of correct choice first, we randomly decide it
			correctIndex = np.random.choice(self.num_choice,N) # get a array of size [N]


			#for i in xrange(N): # some batch will be smaller
			for i in xrange(len(batch.data['y'])):
				y[i,correctIndex[i]] = True
				# put the answer into the choices
				assert len(C[i]) == (self.num_choice - 1)
				C[i].insert(correctIndex[i],Y[i])
				C_c[i].insert(correctIndex[i],Y_c[i])
				assert len(batch.data['cs'][i]) == (self.num_choice - 1)

			# for debug
			if config.showspecs:
				print "first two batch's answer:%s , char:%s, correctIdx:%s"%(Y[:2],Y_c[:2],y[:2])
				print "first two batch's choices:%s , char:%s"%(C[:2],C_c[:2])

		else:
			# for testing, put the answer into the original idx if there is any
			if(batch.data.has_key("y") and batch.data.has_key("cy") and batch.data.has_key('yidx')):
				Y = batch.data['y']
				Y_c = batch.data['cy']
				Y_idx = batch.data['yidx']
				#for i in xrange(N): # some batch will be smaller
				for i in xrange(len(batch.data['y'])):
					#print i,len(C[i])
					assert len(C[i]) == (self.num_choice - 1), ("C[i] len:%s,%s,Y:%s"%(len(C[i]),C[i],Y[i]))
					C[i].insert(Y_idx[i],Y[i])
					C_c[i].insert(Y_idx[i],Y_c[i])
					

			# will check choice num in the end

		# the photo idx is simple

		for i,pi in enumerate(PI):
			# one batch
			for j,pij in enumerate(pi):
				# one album
				if j == config.max_num_albums:
					break
				for k,pijk in enumerate(pij):
					if k == config.max_num_photos:
						break
					#print pijk
					assert isinstance(pijk,int)
					pis[i,j,k] = pijk
					pis_mask[i,j,k] = True



		def get_word(word):
			d = batch.shared['word2idx'] # this is for the word not in glove
			for each in (word, word.lower(), word.capitalize(), word.upper()):
				if each in d:
					return d[each]
			# the word in glove
			
			d2 = batch.shared['existing_word2idx']
			for each in (word, word.lower(), word.capitalize(), word.upper()):
				if each in d2:
					return d2[each] + len(d) # all idx + len(the word to train)
			return 1 # 1 is the -UNK-

		def get_char(char):
			d = batch.shared['char2idx']
			if char in d:
				return d[char]
			return 1

		# for all the text, get each word's index.
		# album title
		for i, ati in enumerate(AT): # batch_sizes
			# one batch
			for j,atij in enumerate(ati):
				# one album
				if j == config.max_num_albums:
					break
				for k,atijk in enumerate(atij):
					# each word
					if k == config.max_sent_album_title_size:
						break
					wordIdx = get_word(atijk)
					at[i,j,k] = wordIdx
					at_mask[i,j,k] = True

		for i, cati in enumerate(AT_c):
			# one batch
			for j, catij in enumerate(cati):
				if j == config.max_num_albums:
					break
				for k, catijk in enumerate(catij):
					# each word
					if k == config.max_sent_album_title_size:
						break
					for l,catijkl in enumerate(catijk):
						if l == config.max_word_size:
							break
						at_c[i,j,k,l] = get_char(catijkl)



		# album description
		for i, adi in enumerate(AD): # batch_sizes
			# one batch
			for j,adij in enumerate(adi):
				# one album
				if j == config.max_num_albums:
					break
				for k,adijk in enumerate(adij):
					# each word
					if k == config.max_sent_des_size:
						break
					wordIdx = get_word(adijk)
					ad[i,j,k] = wordIdx
					ad_mask[i,j,k] = True

		for i, cadi in enumerate(AD_c):
			# one batch
			for j, cadij in enumerate(cadi):
				if j == config.max_num_albums:
					break
				for k, cadijk in enumerate(cadij):
					# each word
					if k == config.max_sent_des_size:
						break
					for l,cadijkl in enumerate(cadijk):
						if l == config.max_word_size:
							break
						ad_c[i,j,k,l] = get_char(cadijkl)




		# album when
		for i, wi in enumerate(WHEN): # batch_sizes
			# one batch
			for j,wij in enumerate(wi):
				# one album
				if j == config.max_num_albums:
					break
				for k,wijk in enumerate(wij):
					# each word
					if k == config.max_when_size:
						break
					wordIdx = get_word(wijk)
					when[i,j,k] = wordIdx
					when_mask[i,j,k] = True

		for i, cwi in enumerate(WHEN_c):
			# one batch
			for j, cwij in enumerate(cwi):
				if j == config.max_num_albums:
					break
				for k, cwijk in enumerate(cwij):
					# each word
					if k == config.max_when_size:
						break
					for l,cwijkl in enumerate(cwijk):
						if l == config.max_word_size:
							break
						when_c[i,j,k,l] = get_char(cwijkl)

		# album where
		for i, wi in enumerate(WHERE): # batch_sizes
			# one batch
			for j,wij in enumerate(wi):
				# one album
				if j == config.max_num_albums:
					break
				for k,wijk in enumerate(wij):
					# each word
					if k == config.max_where_size:
						break
					wordIdx = get_word(wijk)
					where[i,j,k] = wordIdx
					where_mask[i,j,k] = True

		for i, cwi in enumerate(WHERE_c):
			# one batch
			for j, cwij in enumerate(cwi):
				if j == config.max_num_albums:
					break
				for k, cwijk in enumerate(cwij):
					# each word
					if k == config.max_where_size:
						break
					for l,cwijkl in enumerate(cwijk):
						if l == config.max_word_size:
							break
						where_c[i,j,k,l] = get_char(cwijkl)


		# photo title
		for i, pti in enumerate(PT): # batch_sizes
			# one batch
			for j,ptij in enumerate(pti):
				# one album
				if j == config.max_num_albums:
					break
				for k,ptijk in enumerate(ptij):
					# each photo
					if k == config.max_num_photos:
						break
					for l,ptijkl in enumerate(ptijk):
						if l == config.max_sent_photo_title_size:
							break
						# each word
						wordIdx = get_word(ptijkl)
						pts[i,j,k,l] = wordIdx
						pts_mask[i,j,k,l] = True
		for i, pti in enumerate(PT_c): # batch_sizes
			# one batch
			for j,ptij in enumerate(pti):
				# one album
				if j == config.max_num_albums:
					break
				for k,ptijk in enumerate(ptij):
					# each photo
					if k == config.max_num_photos:
						break
					for l,ptijkl in enumerate(ptijk):
						if l == config.max_sent_photo_title_size:
							break
						# each word
						for o, ptijklo in enumerate(ptijkl):
							# each char
							if o == config.max_word_size:
								break
							pts_c[i,j,k,l,o] = get_char(ptijklo)



		# answer choices

		for i,ci in enumerate(C):
			# one batch
			assert len(ci) == self.num_choice
			for j,cij in enumerate(ci):
				# one answer
				for k,cijk in enumerate(cij):
					# one word
					if k == config.max_answer_size:
						break
					wordIdx = get_word(cijk)
					choices[i,j,k] = wordIdx
					choices_mask[i,j,k] = True
		for i,ci in enumerate(C_c):
			# one batch
			assert len(ci) == self.num_choice, (len(ci))
			for j,cij in enumerate(ci):
				# one answer
				for k,cijk in enumerate(cij):
					# one word
					if k == config.max_answer_size:
						break
					for l,cijkl in enumerate(cijk):
						if l == config.max_word_size:
							break
						choices_c[i,j,k,l] = get_char(cijkl)


		# loa the question
		# no limiting on the question word length
		for i, qi in enumerate(Q):
			# one batch
			for j, qij in enumerate(qi):
				q[i, j] = get_word(qij)
				q_mask[i, j] = True

		# load the question char
		for i, cqi in enumerate(Q_c):
			for j, cqij in enumerate(cqi):
				for k, cqijk in enumerate(cqij):
					if k == config.max_word_size:
						break
					q_c[i, j, k] = get_char(cqijk)
		#print feed_dict
		return feed_dict


