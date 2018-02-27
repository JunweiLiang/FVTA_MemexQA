# coding=utf-8
# utils for tensorflow

import tensorflow as tf
from operator import mul
from itertools import izip_longest
import random,itertools
from collections import defaultdict
import math
import numpy as np

def grouper(l,n):
	# given a list and n(batch_size), devide list into n sized chunks
	# last one will fill None
	args = [iter(l)]*n
	out = izip_longest(*args,fillvalue=None)
	out = list(out)
	return out

def sec2time(secs):
	#return strftime("%H:%M:%S",time.gmtime(secs)) # doesnt support millisecs
	
	m,s = divmod(secs,60)
	#print m,s
	h,m = divmod(m,60)
	if(s >= 10.0):
		return "%02d:%02d:%.3f"%(h,m,s)
	else:
		return "%02d:%02d:0%.3f"%(h,m,s)

class Dataset():
	# data should be 
	"""
	data = {
		'q':q,
		'cq':cq,
		'y':y,
		'cy':cy,
		'aid':aid, # each is a list of aids
		'qid':qid,
		'idxs':idxs,
		'cs':cs, # each is a list of wrong choices
		'ccs':ccs,
		################# new for a mini batch##################
		album_title = []
		album_title_c = []
		album_description = []
		album_description_c = []
		where = []
		where_c = []
		when = []
		when_c = []
		photo_titles = []
		photo_titles_c = []
		photo_ids = [] -> original pids , string
		photo_idxs = [] -> pids transform to the image_feat_matrix idx

		image_feat_matrix
	}
	in mini batch,
	data added all other things it need from the shared
	, shared is the whole shared dict

	"""

	def __init__(self,data,datatype,shared=None,valid_idxs=None):
		self.data = data 
		self.datatype = datatype
		self.shared = shared

		self.valid_idxs = range(self.get_data_size()) if valid_idxs is None else valid_idxs
		self.num_examples = len(self.valid_idxs)


	def get_data_size(self):
		return len(next(iter(self.data.values()))) # get one var "q" and get the len

	def get_by_idxs(self,idxs):
		out = defaultdict(list) # so the initial value is a list
		for key,val in self.data.items(): # "q",[] ; "cq", [] ;"y",[]
			out[key].extend(val[idx] for idx in idxs) # extend with one whole list
		# so we get a batch_size of data : {"q":[] -> len() == batch_size}
		return out


	# should return num_steps -> batches
	# step is total/batchSize * epoch
	# cap means limits max number of generated batches to 1 epoch
	def get_batches(self,batch_size,num_steps,shuffle=True,cap=False):

		num_batches_per_epoch = int(math.ceil(self.num_examples / float(batch_size)))
		if cap and (num_steps > num_batches_per_epoch):
			num_steps = num_batches_per_epoch
		# this may be zero
		num_epochs = int(math.ceil(num_steps/float(num_batches_per_epoch)))
		# TODO: Group single-album and cross-album question to train separately?
		# shuflle
		if(shuffle):
			# shuffled idx
			# but all epoch has the same order
			random_idxs = random.sample(self.valid_idxs,len(self.valid_idxs))
			random_grouped = lambda: list(grouper(random_idxs,batch_size)) # all batch idxs for one epoch
			# grouper
			# given a list and n(batch_size), devide list into n sized chunks
			# last one will fill None
			grouped = random_grouped
		else:
			raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
			grouped = raw_grouped
		# grouped is a list of list, each is batch_size items make up to -> total_sample

		# all batches idxs from multiple epochs
		batch_idxs_iter = itertools.chain.from_iterable(grouped() for _ in xrange(num_epochs))
		#print "in get batches, num_steps:%s,num_epch:%s"%(num_steps,num_epochs)
		for _ in xrange(num_steps): # num_step should be batch_idxs length
			# so in the end batch, the None will not included
			batch_idxs = tuple(i for i in next(batch_idxs_iter) if i is not None) # each batch idxs
			# so batch_idxs might not be size batch_size

			#print "batch size:%s"%len(batch_idxs)
			# a dict of {"q":[],"cq":[],"y":[]...}
			# get from dataset class:{"q":...} all the key items with idxs
			# so no album info anything
			batch_data = self.get_by_idxs(batch_idxs) # get the actual data based on idx
			#print len(batch_data['q'])

			# go through all album to get pid2idx first,
			pid2idx = {} # get all the pid to a index
			for albumIds in batch_data['aid']: # each QA has album list
				for albumId in albumIds:
					for pid in self.shared['albums'][albumId]['photo_ids']:
						if(not pid2idx.has_key(pid)):
							pid2idx[pid] = len(pid2idx.keys())# start from zero

			# fill in the image feature
			image_feats = np.zeros((len(pid2idx),self.shared['pid2feat'][self.shared['pid2feat'].keys()[0]].shape[0]),dtype="float32")

			# here image_matrix idx-> feat, will replace the pid in each instance to this idx
			for pid in pid2idx: # fill each idx with feature, -> pid
				image_feats[pid2idx[pid]] = self.shared['pid2feat'][pid]


			batch_data['pidx2feat'] = image_feats


			shared_batch_data = defaultdict(list)
			
			# all the shared data need for this mini batch
			for albumIds in batch_data['aid']:
				# one shared album info for one qa, could be multiple albums
				album_title = []
				album_title_c = []
				album_description = []
				album_description_c = []
				album_where = []
				album_where_c = []
				album_when = []
				album_when_c = []
				photo_titles = []
				photo_titles_c = []
				photo_idxs = []
				photo_ids = [] # for debug
				for albumId in albumIds:
					album = self.shared['albums'][albumId]

					album_title.append(album['title'])
					album_title_c.append(album['title_c'])
					album_description.append(album['description'])
					album_description_c.append(album['description_c'])
					album_where.append(album['where'])
					album_when.append(album['when'])
					album_where_c.append(album['where_c'])
					album_when_c.append(album['when_c'])
					photo_titles.append(album['photo_titles'])
					photo_titles_c.append(album['photo_titles_c'])
					photo_idxs.append([pid2idx[pid] for pid in album['photo_ids']])
					# this will not be used, just for debug
					photo_ids.append(album['photo_ids'])

				shared_batch_data['album_title'].append(album_title)
				shared_batch_data['album_title_c'].append(album_title_c)
				shared_batch_data['album_description'].append(album_description)
				shared_batch_data['album_description_c'].append(album_description_c)
				shared_batch_data['where'].append(album_where)
				shared_batch_data['where_c'].append(album_where_c)
				shared_batch_data['when'].append(album_when)
				shared_batch_data['when_c'].append(album_when_c)
				shared_batch_data['photo_titles'].append(photo_titles)
				shared_batch_data['photo_titles_c'].append(photo_titles_c)
				# all pid should be change to a local batch idx
				shared_batch_data['photo_idxs'].append(photo_idxs)
				# for debug
				shared_batch_data['photo_ids'].append(photo_ids)

			batch_data.update(shared_batch_data) # combine the shared data in to the minibatch
			# so it be {"q","cq","y"...,"pidx2feat","album_info"...}

			yield batch_idxs,Dataset(batch_data,self.datatype,shared=self.shared)






VERY_NEGATIVE_NUMBER = -1e30


# exponetial mask (so the False element doesn't get zero, it get a very_negative_number so that e(numer) == 0.0)
# [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
def exp_mask(val,mask):
	# tf.cast(a,"float") -> [True,True,False] -> [1.0,1.0,0.0] (1 - cast) -> [0.0,0.0,1.0]
	# then the 1.0 * very_negative_number and become a very_negative_number (add val and still very negative), then e(ver_negative_numer) is zero
	return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name="exp_mask")


# flatten a tensor
# [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]
def flatten(tensor, keep): # keep how many dimension in the end, so final rank is keep + 1
	# get the shape
	fixed_shape = tensor.get_shape().as_list() #[N, JQ, di] # [N, M, JX, di] 
	start = len(fixed_shape) - keep # len([N, JQ, di]) - 2 = 1 # len([N, M, JX, di] ) - 2 = 2
	# each num in the [] will a*b*c*d...
	# so [0] -> just N here for left
	# for [N, M, JX, di] , left is N*M
	left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
	# [N, JQ,di]
	# [N*M, JX, di] 
	out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
	# reshape
	flat = tf.reshape(tensor, out_shape)
	return flat

def reconstruct(tensor, ref, keep): # reverse the flatten function
	ref_shape = ref.get_shape().as_list()
	tensor_shape = tensor.get_shape().as_list()
	ref_stop = len(ref_shape) - keep
	tensor_start = len(tensor_shape) - keep
	pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
	keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
	# pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
	# keep_shape = tensor.get_shape().as_list()[-keep:]
	target_shape = pre_shape + keep_shape
	out = tf.reshape(tensor, target_shape)
	return out

# get answer baased on model output and groudtruth
def getAnswers(yp,batch):
	id2predanswers = {}
	id2realanswers = {}
	#print yp.shape
	for qid, yidxi,ypi in zip(batch[1].data['qid'],batch[1].data['yidx'],yp):
		#print qid
		#print yidxi
		#print ypi
		id2predanswers[qid] = np.argmax(ypi)
		id2realanswers[qid] = yidxi # available answers
		assert yidxi < 4
		assert np.argmax(ypi) < 4
		#print q,id2answers[qid
	return id2predanswers,id2realanswers

def getAnswers_yp(yp,batch):
	id2predanswers = {}
	id2realanswers = {}
	id2yp = {}
	#print yp.shape
	for qid, yidxi,ypi in zip(batch[1].data['qid'],batch[1].data['yidx'],yp):
		#print qid
		#print yidxi
		#print ypi
		id2predanswers[qid] = np.argmax(ypi)
		id2realanswers[qid] = yidxi # available answers
		id2yp[qid] = ypi
		assert yidxi < 4
		assert np.argmax(ypi) < 4
		#print q,id2answers[qid
	return id2predanswers,id2realanswers,id2yp

def getEvalScore(pred,gt):
	assert len(pred) == len(gt) 
	assert len(pred) > 0
	total = len(pred)
	correct=0
	for qid in pred.keys():
		if pred[qid] == gt[qid]:
			correct+=1
	return correct/float(total)

"""
max_num_albums:8 ,max_num_photos:10 ,max_sent_title_size:35 ,max_sent_des_size:2574 ,max_when_size:4 ,max_where_size:10 ,max_answer_size:18 ,max_question_size:25 ,max_word_size:42
"""

# datasets[0] should always be the train set
def update_config(config,datasets,showMeta=False):
	config.max_num_albums = 0
	config.max_num_photos = 0 # max photo per album
	config.max_sent_album_title_size = 0 # max sentence word count for album title
	config.max_sent_photo_title_size = 0
	config.max_sent_des_size = 0
	config.max_when_size=0
	config.max_where_size = 0
	config.max_answer_size = 0
	config.max_question_size = 0
	config.max_word_size = 0 # word letter count

	# go through all datasets to get the max count
	for dataset in datasets:
		for idx in dataset.valid_idxs:

			question = dataset.data['q'][idx]
			answer = dataset.data['y'][idx]
			choices = dataset.data['cs'][idx]


			config.max_question_size = max(config.max_question_size,len(question))
			config.max_word_size = max(config.max_word_size, max(len(word) for word in question))

			for sent in choices + [answer]:
				config.max_answer_size = max(config.max_answer_size,len(sent))
				config.max_word_size = max(config.max_word_size, max(len(word) for word in sent))

			albums = [dataset.shared['albums'][aid] for aid in dataset.data['aid'][idx]]

			config.max_num_albums = max(config.max_num_albums,len(albums))

			for album in albums:
				config.max_num_photos = max(config.max_num_photos,len(album['photo_ids']))

				# title
				#config.max_sent_title_size = max(config.max_sent_title_size,len(album['title']))
				config.max_sent_album_title_size = max(config.max_sent_album_title_size,len(album['title']))
				
				for title in album['photo_titles']:
					if len(title) > 0:
						#config.max_sent_title_size = max(config.max_sent_title_size,len(title))
						config.max_sent_photo_title_size = max(config.max_sent_photo_title_size,len(title))
						config.max_word_size = max(config.max_word_size, max(len(word) for word in title))

				# description
				if len(album['description'])>0:
					config.max_sent_des_size = max(config.max_sent_des_size,len(album['description']))
					config.max_word_size = max(config.max_word_size, max(len(word) for word in album['description']))

				#when
				config.max_when_size = max(config.max_when_size,len(album['when']))
				

				# got word size for all
				config.max_word_size = max(config.max_word_size, max(len(word) for word in album['title']))
				config.max_word_size = max(config.max_word_size, max(len(word) for word in album['when']))
				# where could be empty
				if(len(album['where']) != 0):
					config.max_word_size = max(config.max_word_size, max(len(word) for word in album['where']))
					config.max_where_size = max(config.max_where_size,len(album['where']))
						

	
	if showMeta:
		config_vars = vars(config)
		print "max meta:"
		print "\t" + " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.maxmeta])

	# adjust the max based on the threshold argument input as well
	if config.is_train:
		# album and photo counts
		config.max_num_albums = min(config.max_num_albums,config.num_albums_thres)
		config.max_num_photos = min(config.max_num_photos,config.num_photos_thres)

		#config.max_sent_title_size = min(config.max_sent_title_size,config.sent_title_size_thres)
		config.max_sent_album_title_size = min(config.max_sent_album_title_size,config.sent_album_title_size_thres)
		config.max_sent_photo_title_size = min(config.max_sent_photo_title_size,config.sent_photo_title_size_thres)

		config.max_sent_des_size = min(config.max_sent_des_size,config.sent_des_size_thres)

		config.max_when_size = min(config.max_when_size,config.sent_when_size_thres)
		config.max_where_size = min(config.max_where_size,config.sent_where_size_thres)

		config.max_answer_size = min(config.max_answer_size,config.answer_size_thres)
		
		# not cliping question
		#config.question_size_thres = max(config.max_question_size,config.question_size_thres)
	else:
		# for testing, still removing the description since it could be 2k+ tokens
		config.max_sent_des_size = min(config.max_sent_des_size,config.sent_des_size_thres)
		# also cap the photo title size
		config.max_sent_photo_title_size = min(config.max_sent_photo_title_size,config.sent_photo_title_size_thres)

	# always clip word_size
	config.max_word_size = min(config.max_word_size, config.word_size_thres)

	# get the vocab size # the charater in the charCounter
	config.char_vocab_size = len(datasets[0].shared['char2idx'])
	# the word embeding's dimension
	config.word_emb_size = len(next(iter(datasets[0].shared['word2vec'].values())))
	# the size of word vocab not in existing glove
	config.word_vocab_size = len(datasets[0].shared['word2idx'])
