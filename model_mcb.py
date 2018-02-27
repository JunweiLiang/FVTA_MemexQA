# coding=utf-8
# tensorflow model graph 


import tensorflow as tf
from utils import flatten,reconstruct,Dataset,exp_mask
import numpy as np
import random,sys


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

# softmax selection?
# return target * softmax(logits)
# target: [ ..., J, d]
# logits: [ ..., J]

# so [N,M,dim] * [N,M] -> [N,dim], so [N,M] is the attention for each M

# return: [ ..., d] # so the target vector is attended with logits' softmax
# [N,M,JX,JQ,2d] * [N,M,JX,JQ] (each context to query's mapping) -> [N,M,JX,2d] # attened the JQ dimension
def softsel(target,logits,hard=False,hardK=None,scope=None):
	with tf.variable_scope(scope or "softsel"): # there is no variable to be learn here
				
		a = softmax(logits) # shape is the same

		
		target_rank = len(target.get_shape().as_list())
		# [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
		return tf.reduce_sum(tf.expand_dims(a,-1)*target,target_rank-2) # second last dim


# x -> [Num,JX,W,embedding dim] # conv2d requires an input of 4d [batch, in_height, in_width, in_channels]
def conv1d(x,filter_size,height,keep_prob,is_train=None,wd=None,scope=None):
	with tf.variable_scope(scope or "conv1d"):
		num_channels = x.get_shape()[-1] # embedding dim[8]
		filter_var = tf.get_variable("filter",shape=[1,height,num_channels,filter_size],dtype="float")
		bias = tf.get_variable('bias',shape=[filter_size],dtype='float')
		strides = [1,1,1,1]
		# add dropout to input
		
		d = tf.nn.dropout(x,keep_prob=keep_prob)
		outd = tf.cond(is_train,lambda:d,lambda:x)
		#conv
		xc = tf.nn.relu(tf.nn.conv2d(outd,filter_var,strides,padding='VALID')+bias)
		# simple max pooling?
		out = tf.reduce_max(xc,2) # [-1,JX,num_channel]

		if wd is not None:
			add_wd(wd)

		return out

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



def batch_norm(x,is_train=True,epsilon=1e-5,decay=0.9,scope=None):
	scope = scope or "batch_norm"
	# what about tf.nn.batch_normalization
	return tf.contrib.layers.batch_norm(x,decay=decay,updates_collections=None,epsilon=epsilon,scale=True,is_training=is_train,scope=scope)

# add current scope's variable's l2 loss to loss collection
def add_wd(wd,scope=None):
	if wd != 0.0:
		scope = scope or tf.get_variable_scope().name
		vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		with tf.variable_scope("weight_decay"):
			for var in vars_:
				weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="%s/wd"%(var.op.name))
				tf.add_to_collection("losses",weight_decay)


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
		# these could be used for visualization
		self.C = tf.constant(-1) # the time correlation matrix
		self.C_win = tf.constant(-1)
		self.att_logits = tf.constant(-1) # the 3d attention logits
		self.q_att_logits = tf.constant(-1) # the question attention logits if there is
		self.JXP = tf.constant(-1)

		self.hat_len = tf.constant(-1)
		self.had_len = tf.constant(-1)
		self.hwhen_len = tf.constant(-1)
		self.hwhere_len = tf.constant(-1)
		self.hpis_len = tf.constant(-1)
		self.hpts_len = tf.constant(-1)

		#for showing the h vectors
		self.warp_h = tf.constant(-1)
		self.hall = tf.constant(-1)

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

		# album title
		# [N,M,JXA]
		self.at = tf.placeholder('int32',[N,None,None],name="at")
		self.at_c = tf.placeholder("int32",[N,None,None,W],name="at_c")
		self.at_mask = tf.placeholder("bool",[N,None,None],name="at_mask") # to get the sequence length

		# album description
		# [N,M,JD]
		self.ad = tf.placeholder('int32',[N,None,None],name="ad")
		self.ad_c = tf.placeholder("int32",[N,None,None,W],name="ad_c")
		self.ad_mask = tf.placeholder("bool",[N,None,None],name="ad_mask")

		# album when, where
		# [N,M,JT/JG]
		self.when = tf.placeholder("int32",[N,None,None],name="when")
		self.when_c = tf.placeholder("int32",[N,None,None,W],name="when_c")
		self.when_mask = tf.placeholder("bool",[N,None,None],name="when_mask")
		self.where = tf.placeholder("int32",[N,None,None],name="where")
		self.where_c = tf.placeholder("int32",[N,None,None,W],name="where_c")
		self.where_mask = tf.placeholder("bool",[N,None,None],name="where_mask")

		# photo titles
		# [N,M,JI,JXP]
		self.pts = tf.placeholder('int32',[N,None,None,None],name="pts")
		self.pts_c = tf.placeholder("int32",[N,None,None,None,W],name="pts_c")
		self.pts_mask = tf.placeholder("bool",[N,None,None,None],name="pts_mask")

		# for vis
		self.JXP = tf.shape(self.pts)[3]

		# photo
		# [N,M,JI] # each is a photo index
		self.pis = tf.placeholder('int32',[N,None,None],name="pis")
		self.pis_mask = tf.placeholder("bool",[N,None,None],name="pis_mask")

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
		M = tf.shape(self.pis)[1] # photo num
		JI = tf.shape(self.pis)[2] # used for photo_title, photo

		#M = config.max_num_albums
		#JI = config.max_num_photos

		JXA = tf.shape(self.at)[2] # for album title, photo title
		JD = tf.shape(self.ad)[2] # description length
		JT = tf.shape(self.when)[2]
		JG = tf.shape(self.where)[2]

		
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
			# char stuff
			if config.use_char:
			#with tf.variable_scope("char"):
				# [char_vocab_size,char_emb_dim]
				with tf.variable_scope("var"): #, tf.device("/cpu:0"): # in cpu for faster in multi-gpu training
					char_emb = tf.get_variable("char_emb",shape=[VC,cdim],dtype="float")

				# the embedding for each of character 
				# [N,M,JXA,W]
				Aat_c = tf.nn.embedding_lookup(char_emb,self.at_c)
				# [N,M,JD,W]
				Aad_c = tf.nn.embedding_lookup(char_emb,self.ad_c)
				# [N,M,JT,W]
				Awhen_c = tf.nn.embedding_lookup(char_emb,self.when_c)
				# [N,M,JG,W]
				Awhere_c = tf.nn.embedding_lookup(char_emb,self.where_c)
				# [N,M,JI,JXP,W] -> [N,M,JI,JXP,W,cdim]
				Apts_c = tf.nn.embedding_lookup(char_emb,self.pts_c)

				# [N,JQ,W]
				Aq_c = tf.nn.embedding_lookup(char_emb,self.q_c)
				Achoices_c = tf.nn.embedding_lookup(char_emb,self.choices_c)

				# flatten for conv2d input like images
				Aat_c = tf.reshape(Aat_c,[-1,JXA,W,cdim])
				Aad_c = tf.reshape(Aad_c,[-1,JD,W,cdim])
				Awhen_c = tf.reshape(Awhen_c,[-1,JT,W,cdim])
				Awhere_c = tf.reshape(Awhere_c,[-1,JG,W,cdim])
				# [N*M*JI,JXP,W,cdim]
				Apts_c = tf.reshape(Apts_c,[-1,JXP,W,cdim])

				Aq_c = tf.reshape(Aq_c,[-1,JQ,W,cdim])
				# [N*4,]
				Achoices_c = tf.reshape(Achoices_c,[-1,JA,W,cdim])

				#char CNN
				filter_size = cwdim # output size for each word
				filter_height = 5
				with tf.variable_scope("conv"):
					xat = conv1d(Aat_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
					tf.get_variable_scope().reuse_variables()
					xad = conv1d(Aad_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
					xwhen = conv1d(Awhen_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
					xwhere = conv1d(Awhere_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
					xpts = conv1d(Apts_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
					qq = conv1d(Aq_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
					qchoices = conv1d(Achoices_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")

					# reshape them back
					xat = tf.reshape(xat,[-1,M,JXA,cwdim])
					xad = tf.reshape(xad,[-1,M,JD,cwdim])
					xwhen = tf.reshape(xwhen,[-1,M,JT,cwdim])
					xwhere = tf.reshape(xwhere,[-1,M,JG,cwdim])
					xpts = tf.reshape(xpts,[-1,M,JI,JXP,cwdim])

					qq = tf.reshape(qq,[-1,JQ,cwdim])
					# [N,num_choice,JA,cwdim]
					qchoices = tf.reshape(qchoices,[-1,self.num_choice,JA,cwdim])

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

			# concat char and word
			if config.use_char:

				xat = tf.concat([xat,Aat],3)
				xad = tf.concat([xad,Aad],3)
				xwhen = tf.concat([xwhen,Awhen],3)
				xwhere = tf.concat([xwhere,Awhere],3)
				# [N,M,JI,JX,wdim+cwdim]
				xpts = tf.concat([xpts,Apts],4)

				# [N,JQ,wdim+cwdim]
				qq = tf.concat([qq,Aq],2)
				qchoices = tf.concat([qchoices,Achoices],3)

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
		cell_text = tf.nn.rnn_cell.BasicLSTMCell(d,state_is_tuple=True)
		cell_img = tf.nn.rnn_cell.BasicLSTMCell(d,state_is_tuple=True)
		#cell = tf.nn.rnn_cell.GRUCell(d)
		# add dropout
		keep_prob = tf.cond(self.is_train,lambda:tf.constant(config.keep_prob),lambda:tf.constant(1.0))

		cell_text = tf.nn.rnn_cell.DropoutWrapper(cell_text,keep_prob)

		cell_img = tf.nn.rnn_cell.DropoutWrapper(cell_img,keep_prob)

		
		# it is important to think about which LSTM shared with which?

		# sequence length for each
		at_len = tf.reduce_sum(tf.cast(self.at_mask,"int32"),2) # [N,M] # each album's title length
		ad_len = tf.reduce_sum(tf.cast(self.ad_mask,"int32"),2)
		when_len = tf.reduce_sum(tf.cast(self.when_mask,"int32"),2)
		where_len = tf.reduce_sum(tf.cast(self.where_mask,"int32"),2) # [N,M]

		pis_len = tf.reduce_sum(tf.cast(self.pis_mask,"int32"),2) #[N,M,JI] #[N,M]

		pts_len = tf.reduce_sum(tf.cast(self.pts_mask,"int32"),3) # [N,M,JI,JXP] -> [N,M,JI]

		q_len = tf.reduce_sum(tf.cast(self.q_mask,"int32"),1) # [N] # each question 's length

		choices_len = tf.reduce_sum(tf.cast(self.choices_mask,"int32"),2) # [N,4]

		# xat -> [N,M,JXA,wdim+cwdim]
		# xad -> [N,M,JD,wdim+cwdim]
		# xwhen/xwhere -> [N,M,JT/JG,wdim+cwdim]
		# xpts -> [N,M,JI,JXP,wdim+cwdim]

		# xpis -> [N,M,JI,idim]

		# qq -> [N,JQ,wdim+cwdim]
		# qchoices -> [N,4,JA,wdim+cwdim]

		# roll the sentence into lstm for context and question
		# from [N,M,JI,JX] -> [N,M,2d]
		with tf.variable_scope("reader"):
			with tf.variable_scope("text"):
				#qq = tf.check_numerics(qq,"NaN or Inf check in trainer,qq")
				(fw_hq,bw_hq),(fw_lq,bw_lq) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,qq,sequence_length=q_len,dtype="float",scope="utext")
				# concat the fw and backward lstm output
				hq = tf.concat([fw_hq,bw_hq],2)
				#hq = tf.check_numerics(hq,"NaN or Inf check in trainer,hq")
				lq = tf.concat([fw_lq.h,bw_lq.h],1) #LSTM CELL
				#lq = tf.concat([fw_lq,bw_lq],1) # GRU

				tf.get_variable_scope().reuse_variables()

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

				
				# put all through LSTM
				# uncomment to use ALL LSTM output or LAST LSTM output

				# album title
				# [N*M,JXA,d]
				(fw_hat_flat,bw_hat_flat),(fw_lat_flat,bw_lat_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xat,sequence_length=flat_xat_len,dtype="float",scope="utext")
				fw_hat = reconstruct(fw_hat_flat,xat,2) # 
				bw_hat = reconstruct(bw_hat_flat,xat,2)
				hat = tf.concat([fw_hat,bw_hat],3) # [N,M,JXA,2d]
				# lstm
				fw_lat = tf.reshape(fw_lat_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
				bw_lat = tf.reshape(bw_lat_flat.h,[N,M,d])
				# GRU
				#fw_lat = tf.reshape(fw_lat_flat,[N,M,d]) # [N*M,d] -> [N,M,d]
				#bw_lat = tf.reshape(bw_lat_flat,[N,M,d])
				lat = tf.concat([fw_lat,bw_lat],2) # [N,M,2d]


				# album desciption
				# [N*M,JD,d]
				(fw_had_flat,bw_had_flat),(fw_lad_flat,bw_lad_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xad,sequence_length=flat_xad_len,dtype="float",scope="utext")
				fw_had = reconstruct(fw_had_flat,xad,2) # 
				bw_had = reconstruct(bw_had_flat,xad,2)
				had = tf.concat([fw_had,bw_had],3) # [N,M,JD,2d]
				# LSTM
				fw_lad = tf.reshape(fw_lad_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
				bw_lad = tf.reshape(bw_lad_flat.h,[N,M,d])
				# GRU
				#fw_lad = tf.reshape(fw_lad_flat,[N,M,d]) # [N*M,d] -> [N,M,d]
				#bw_lad = tf.reshape(bw_lad_flat,[N,M,d])
				lad = tf.concat([fw_lad,bw_lad],2) # [N,M,2d]

				# when
				(fw_hwhen_flat,bw_hwhen_flat),(fw_lwhen_flat,bw_lwhen_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xwhen,sequence_length=flat_xwhen_len,dtype="float",scope="utext")
				fw_hwhen = reconstruct(fw_hwhen_flat,xwhen,2) # 
				bw_hwhen = reconstruct(bw_hwhen_flat,xwhen,2)
				hwhen = tf.concat([fw_hwhen,bw_hwhen],3) # [N,M,JT,2d]
				# LSTM
				fw_lwhen = tf.reshape(fw_lwhen_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
				bw_lwhen = tf.reshape(bw_lwhen_flat.h,[N,M,d])
				# GRU
				#fw_lwhen = tf.reshape(fw_lwhen_flat,[N,M,d]) # [N*M,d] -> [N,M,d]
				#bw_lwhen = tf.reshape(bw_lwhen_flat,[N,M,d])
				lwhen = tf.concat([fw_lwhen,bw_lwhen],2) # [N,M,2d]


				# where
				(fw_hwhere_flat,bw_hwhere_flat),(fw_lwhere_flat,bw_lwhere_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xwhere,sequence_length=flat_xwhere_len,dtype="float",scope="utext")
				fw_hwhere = reconstruct(fw_hwhere_flat,xwhere,2) # 
				bw_hwhere = reconstruct(bw_hwhere_flat,xwhere,2)
				hwhere = tf.concat([fw_hwhere,bw_hwhere],3) # [N,M,JG,2d]
				# LSTM
				fw_lwhere = tf.reshape(fw_lwhere_flat.h,[N,M,d]) # [N*M,d] -> [N,M,d]
				bw_lwhere = tf.reshape(bw_lwhere_flat.h,[N,M,d])
				# GRU
				#fw_lwhere = tf.reshape(fw_lwhere_flat,[N,M,d]) # [N*M,d] -> [N,M,d]
				#bw_lwhere = tf.reshape(bw_lwhere_flat,[N,M,d])
				lwhere = tf.concat([fw_lwhere,bw_lwhere],2) # [N,M,2d]


				# photo title
				# [N*M*JI,JXP,d]
				(fw_hpts_flat,bw_hpts_flat),(fw_lpts_flat,bw_lpts_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_xpts,sequence_length=flat_xpts_len,dtype="float",scope="utext")
				fw_hpts = reconstruct(fw_hpts_flat,xpts,2) # 
				bw_hpts = reconstruct(bw_hpts_flat,xpts,2) # [N,M,JI,JXP,d]
				hpts = tf.concat([fw_hpts,bw_hpts],4) # [N,M,JI,JXP,2d]
				# LSTM
				fw_lpts = tf.reshape(fw_lpts_flat.h,[N,M,JI,d]) # [N*M*JI,d] -> [N,M,JI,d]
				bw_lpts = tf.reshape(bw_lpts_flat.h,[N,M,JI,d])
				# GRU
				#fw_lpts = tf.reshape(fw_lpts_flat,[N,M,JI,d]) # [N*M*JI,d] -> [N,M,JI,d]
				#bw_lpts = tf.reshape(bw_lpts_flat,[N,M,JI,d])
				lpts = tf.concat([fw_lpts,bw_lpts],3) # [N,M,JI,2d]	

				# choices
				(fw_hchoices_flat,bw_hchoices_flat),(fw_lchoices_flat,bw_lchoices_flat) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,flat_qchoices,sequence_length=flat_qchoices_len,dtype="float",scope="utext")
				fw_hchoices = reconstruct(fw_hchoices_flat,qchoices,2) # 
				bw_hchoices = reconstruct(bw_hchoices_flat,qchoices,2)
				hchoices = tf.concat([fw_hchoices,bw_hchoices],3) # [N,4,JA,2d]
				# LSTM
				fw_lchoices = tf.reshape(fw_lchoices_flat.h,[N,-1,d]) # [N*4,d] -> [N,4,d]
				bw_lchoices = tf.reshape(bw_lchoices_flat.h,[N,-1,d])
				# GRU
				#fw_lchoices = tf.reshape(fw_lchoices_flat,[N,-1,d]) # [N*4,d] -> [N,4,d]
				#bw_lchoices = tf.reshape(bw_lchoices_flat,[N,-1,d])
				lchoices = tf.concat([fw_lchoices,bw_lchoices],2) # [N,4,2d]


			with tf.variable_scope("image"):

			
				# image is directly used in MCB
				hpis = xpis #[N,M,JI,idim]

			if config.wd is not None: # l2 weight decay for the reader
				add_wd(config.wd)

		# all rnn output
		# hq -> [N,JQ,2d]

		# hat -> [N,M,JXA,2d]
		# had -> [N,M,JD,2d]
		# hwhen -> [N,M,JT,2d]
		# hwhere -> [N,M,JG,2d]
		# hpts -> [N,M,JI,JXP,2d]
		# hpis -> [N,M,JI,2d]

		# hchoices -> [N,4,JA,2d]

		# last states:
		# lq -> [N,2d]

		# lat -> [N,M,2d]
		# lad -> [N,M,2d]
		# lwhen -> [N,M,2d]
		# lwhere -> [N,M,2d]
		# lpts -> [N,M,JI,2d]
		# lpis -> [N,M,2d]

		# lchoices -> [N,4,2d]
		with tf.variable_scope("question_emb"):
			# the question representation, use last state as the MCB paper

			gq = lq  # [N,2d]
			gq = tf.reshape(gq,[N,2*d])
			#gq = tf.check_numerics(gq,"NaN or Inf check in trainer,gq")
			
   			gq = tf.where(tf.is_nan(gq), tf.zeros_like(gq), gq)

		with tf.variable_scope("mcb_attention"):
			# https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling
			# slightly modified to have static output shape 
			
			# tile the questino for mcb 
			gq_tile = tf.tile(tf.expand_dims(tf.expand_dims(gq,1),1),[1,M,JI,1])
			# gq_tile -> [N,M,JI,2d]
			# hpis -> [N,M,JI,idim]
			# assuming 8 albums, 8 image per album
			gq_tile = tf.reshape(gq_tile,[N,M,JI,2*d])


			hpis = tf.where(tf.is_nan(hpis), tf.zeros_like(hpis), hpis)
			hpis = tf.reshape(hpis,[N,M,JI,idim])
			#print hpis.get_shape() # (16, 8, 8, 2537)
			#print gq_tile.get_shape()  #(16, 8, 8, 100)
			# Compact bilinear pooled results of shape [batch_size, output_dim] or [batch_size, height, width, output_dim], depending on `sum_pool`.
			
			g_mcb = compact_bilinear_pooling_layer(hpis,gq_tile,config.mcb_outdim,sequential=False,compute_size=16,sum_pool=False) # [N,mcb_outdim]
			
			# sign sqrt as the paper
			# l2norm
			g_mcb = tf.nn.l2_normalize(tf.sqrt(tf.sign(g_mcb)*g_mcb),dim=-1)
			g_mcb = tf.where(tf.is_nan(g_mcb), tf.zeros_like(g_mcb), g_mcb)
			g_mcb = tf.check_numerics(g_mcb,"NaN or Inf check in trainer,g_mcb")
			# dropout
			xd = tf.nn.dropout(g_mcb,keep_prob=config.keep_prob)
			g_mcb = tf.cond(self.is_train,lambda:xd,lambda:g_mcb)

			#print g_mcb.get_shape() # (16, 8, 8, 16000)

			# conv
			conv1_out = 512
			filter_var = tf.get_variable("filter_conv1",shape=[1,1,config.mcb_outdim,conv1_out],initializer=tf.truncated_normal_initializer(stddev=0.02),dtype="float")
			strides = [1,1,1,1]
			g_mcb_conv1 = tf.nn.relu(tf.nn.conv2d(g_mcb,filter_var,strides,padding='SAME'))
			#print g_mcb_conv1.get_shape() # ...512

			conv2_out = 1
			filter_var = tf.get_variable("filter_conv2",shape=[1,1,conv1_out,conv2_out],initializer=tf.truncated_normal_initializer(stddev=0.02),dtype="float")
			strides = [1,1,1,1]
			g_mcb_conv2 = tf.nn.relu(tf.nn.conv2d(g_mcb_conv1,filter_var,strides,padding='SAME'))
			#print g_mcb_conv2.get_shape()  # ...1
			g_mcb_att = tf.nn.softmax(tf.reshape(g_mcb_conv2,[N,-1]))
			g_mcb_att = tf.reshape(g_mcb_att,[N,M*JI,1])
			self.att_logits = g_mcb_att

			#print g_mcb_att.get_shape()

			g_mcb_attended = tf.reduce_sum(tf.reshape(hpis,[N,M*JI,idim])*g_mcb_att,1)

			#print g_mcb_attended.get_shape() # [N,idim]
			#sys.exit()
			self.mcb1 = g_mcb_attended
			
			#g_mcb_attended = tf.reduce_mean(hpis,[1,2])


			g_mcb_attended = linear(g_mcb_attended,output_size=2*d,add_tanh=config.add_tanh,scope="g_mcb_trans")


		# now from [N,M,2d] -> [N,2d]
		# attention layer
		with tf.variable_scope("text_rep"): # for baseline here has no attention

			# for all text context, use last state
			# use last lstm output (last hidden state)
			# outputs_fw[k,X_len[k]-1] == states_fw.h[k]
			# at_len -> [N,M]
			"""
			lat = tf.check_numerics(lat,"NaN or Inf check in trainer,lat")
			lad = tf.check_numerics(lad,"NaN or Inf check in trainer,lad")
			lwhen = tf.check_numerics(lwhen,"NaN or Inf check in trainer,lwhen")
			lwhere = tf.check_numerics(lwhere,"NaN or Inf check in trainer,lwhere")
			lpts = tf.check_numerics(lpts,"NaN or Inf check in trainer,lpts")
			"""

			g0at = lat #[N,M,2d]
			g0ad = lad # [N,M,2d]
			g0when = lwhen
			g0where = lwhere
			g0pts = tf.reduce_mean(lpts,2) #[N,M,JI,2d] -> [N,M,2d]
			
			g1at = tf.reduce_mean(g0at,1) # [N,2d]
			g1ad = tf.reduce_mean(g0ad,1)
			g1when = tf.reduce_mean(g0when,1)
			g1where = tf.reduce_mean(g0where,1)
			g1pts = tf.reduce_mean(g0pts,1)

			# stack them and mean pool
			K = 5
			g1 = tf.stack([g1at,g1ad,g1when,g1where,g1pts],axis=1) # [N,K,2d]
			g1 = tf.where(tf.is_nan(g1), tf.zeros_like(g1), g1)
			g1 = tf.reduce_mean(g1,1) # [N,2d]
			g1 = tf.reshape(g1,[N,2*d])
			#g1 = tf.check_numerics(g1,"NaN or Inf check in trainer,g1")
			


		with tf.variable_scope("choices_emb"):

			gchoices = lchoices #[N,4,2d] # last LSTM state for each choice


		# the modeling layer
		with tf.variable_scope("output"):
			# tile gq for all choices
			gq = tf.tile(tf.expand_dims(gq,1),[1,self.num_choice,1]) # [N,4,2d]
			# [N,4,2d]
			g1_a_t = tf.tile(tf.expand_dims(g1,1),[1,self.num_choice,1])


			g_mcb_tile = tf.tile(tf.expand_dims(g_mcb_attended,1),[1,self.num_choice,1])

			# MCB the question and the text context first
			# gq ->[N,2d]
			# g1 -> [N,2d]
			logits = linear(tf.concat([gq,g1_a_t,g_mcb_tile,gchoices,gq*g1_a_t,gq*g_mcb_tile,gq*gchoices],2),output_size=1,add_tanh=config.add_tanh,scope="att_logits")

			logits = tf.squeeze(logits,2)
			yp = tf.nn.softmax(logits)

			

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

		# there might be l2 weight loss in some layer
		self.loss = tf.add_n(tf.get_collection("losses"),name="total_losses")
		#tf.summary.scalar(self.loss.op.name, self.loss)

	# givng a batch of data, construct the feed dict
	def get_feed_dict(self,batch,is_train=False):
		assert isinstance(batch,Dataset)
		# get the cap for each kind of step first
		config = self.config
		N = config.batch_size
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
		if config.is_train:
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

			
			new_JI = max(len(album) for sample in batch.data['photo_ids'] for album in sample)
			if new_JI == 0: # empty??
				new_JI = 1
			JI = min(JI,new_JI)
			

			new_JQ = max(len(ques) for ques in batch.data['q'])
			if(new_JQ == 0):
				new_JQ = 1
			JQ = min(JQ,new_JQ)

			
			new_M = max(len(onesample) for onesample in batch.data['album_title'])
			if(new_M == 0):
				new_M = 1
			M = min(M,new_M)
		

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


#--------------------------https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling



def _generate_sketch_matrix(rand_h, rand_s, output_dim):
	"""
	Return a sparse matrix used for tensor sketch operation in compact bilinear
	pooling

	Args:
		rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
		rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
		output_dim: the output dimensions of compact bilinear pooling.

	Returns:
		a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
	"""

	# Generate a sparse matrix for tensor count sketch
	rand_h = rand_h.astype(np.int64)
	rand_s = rand_s.astype(np.float32)
	assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
	assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

	input_dim = len(rand_h)
	indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
							  rand_h[..., np.newaxis]), axis=1)
	sparse_sketch_matrix = tf.sparse_reorder(
		tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
	return sparse_sketch_matrix

def compact_bilinear_pooling_layer(bottom1, bottom2, output_dim, sum_pool=True,
	rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
	seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7, sequential=True,
	compute_size=128):
	"""
	Compute compact bilinear pooling over two bottom inputs. Reference:

	Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
	Conference on Computer Vision and Pattern Recognition (2016).
	Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
	Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).

	Args:
		bottom1: 1st input, 4D Tensor of shape [batch_size, height, width, input_dim1].
		bottom2: 2nd input, 4D Tensor of shape [batch_size, height, width, input_dim2].

		output_dim: output dimension for compact bilinear pooling.

		sum_pool: (Optional) If True, sum the output along height and width
				  dimensions and return output shape [batch_size, output_dim].
				  Otherwise return [batch_size, height, width, output_dim].
				  Default: True.

		rand_h_1: (Optional) an 1D numpy array containing indices in interval
				  `[0, output_dim)`. Automatically generated from `seed_h_1`
				  if is None.
		rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
				  as `rand_h_1`. Automatically generated from `seed_s_1` if is
				  None.
		rand_h_2: (Optional) an 1D numpy array containing indices in interval
				  `[0, output_dim)`. Automatically generated from `seed_h_2`
				  if is None.
		rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
				  as `rand_h_2`. Automatically generated from `seed_s_2` if is
				  None.

		sequential: (Optional) if True, use the sequential FFT and IFFT
					instead of tf.batch_fft or tf.batch_ifft to avoid
					out-of-memory (OOM) error.
					Note: sequential FFT and IFFT are only available on GPU
					Default: True.
		compute_size: (Optional) The maximum size of sub-batch to be forwarded
					  through FFT or IFFT in one time. Large compute_size may
					  be faster but can cause OOM and FFT failure. This
					  parameter is only effective when sequential == True.
					  Default: 128.

	Returns:
		Compact bilinear pooled results of shape [batch_size, output_dim] or
		[batch_size, height, width, output_dim], depending on `sum_pool`.
	"""

	# Static shapes are needed to construction count sketch matrix
	batch_size, height, width, input_dim1 = bottom1.get_shape().as_list()
	input_dim2 = bottom2.get_shape().as_list()[-1]

	# Step 0: Generate vectors and sketch matrix for tensor count sketch
	# This is only done once during graph construction, and fixed during each
	# operation
	if rand_h_1 is None:
		np.random.seed(seed_h_1)
		rand_h_1 = np.random.randint(output_dim, size=input_dim1)
	if rand_s_1 is None:
		np.random.seed(seed_s_1)
		rand_s_1 = 2*np.random.randint(2, size=input_dim1) - 1
	#print(input_dim1,rand_h_1,rand_s_1)
	sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)
	if rand_h_2 is None:
		np.random.seed(seed_h_2)
		rand_h_2 = np.random.randint(output_dim, size=input_dim2)
	if rand_s_2 is None:
		np.random.seed(seed_s_2)
		rand_s_2 = 2*np.random.randint(2, size=input_dim2) - 1
	sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

	# Step 1: Flatten the input tensors and count sketch
	bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
	bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])
	# Essentially:
	#   sketch1 = bottom1 * sparse_sketch_matrix
	#   sketch2 = bottom2 * sparse_sketch_matrix
	# But tensorflow only supports left multiplying a sparse matrix, so:
	#   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
	#   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
	sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
		bottom1_flat, adjoint_a=True, adjoint_b=True))
	sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
		bottom2_flat, adjoint_a=True, adjoint_b=True))

	# Step 2: FFT
	fft1 = tf.fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
	fft2 = tf.fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

	# Step 3: Elementwise product
	fft_product = tf.multiply(fft1, fft2)

	# Step 4: Inverse FFT and reshape back
	# Compute output shape dynamically: [batch_size, height, width, output_dim]
	cbp_flat = tf.real(tf.ifft(fft_product))

	output_shape = tf.add(tf.multiply(tf.shape(bottom1), [1, 1, 1, 0]), [0, 0, 0, output_dim])
	#output_shape = [batch_size, height, width, output_dim]

	cbp = tf.reshape(cbp_flat, output_shape)

	# Step 5: Sum pool over spatial dimensions, if specified
	if sum_pool:
		cbp = tf.reduce_sum(cbp, reduction_indices=[1, 2])

	return cbp