# coding=utf-8

import sys,os,argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # so here won't have poll allocator info

import cPickle as pickle
import numpy as np

from trainer import Trainer
from tester import Tester
import math,time,json

import tensorflow as tf

from tqdm import tqdm

from utils import Dataset,update_config,getAnswers,getEvalScore,sec2time,getAnswers_yp

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

get_model = None # the model we will use, based on parameter in the get_args()

def get_args():
	global get_model
	parser = argparse.ArgumentParser()
	parser.add_argument("prepropath",type=str)
	parser.add_argument("outbasepath",type=str,help="full path will be outbasepath/modelname/runId")
	parser.add_argument("--modelname",type=str,default="memoryqa")
	parser.add_argument("--runId",type=int,default=0,help="used for run the same model multiple times")

	parser.add_argument("--use_3d",action="store_true", help="use 3D tensor model")

	parser.add_argument("--dmnplus",action="store_true", help="DMN+ model exp")
	parser.add_argument("--dmnplus_num_hops",type=int,default=3,help="dmn hop times")

	parser.add_argument("--mcb",action="store_true", help="MCB attention model exp")
	parser.add_argument("--mcb_outdim",type=int,default=16000, help="MCB attention model exp")

	
	parser.add_argument("--no_photo",action="store_true",default=False,help="text-only experiment")

	parser.add_argument("--load",action="store_true",default=False,help="whether to load existing model")
	parser.add_argument("--load_best",action="store_true",default=False,help="whether to load the best model")
	

	parser.add_argument("--is_train",action="store_true",default=False,help="training mode, ")
	parser.add_argument("--is_test",action="store_true",default=False,help="testing mode, otherwise test mode")

	parser.add_argument("--get_yp",action="store_true",default=False,help="testing mode, whether to save all yp")

	parser.add_argument("--is_test_on_val",action="store_true",default=False,help="test on validation set")
	parser.add_argument("--is_save_weights",action="store_true",default=False,help="whether to save model weights to val_path")
	parser.add_argument("--is_save_vis",action="store_true",default=False,help="whether to save each layer output for visualization during testing, will save into val_path")


	parser.add_argument("--showspecs",action="store_true",default=False,help="show the meta of the data and then exit")
	parser.add_argument("--save_period",type=int,default=200,help="num steps to save model and eval")
	parser.add_argument("--val_path",type=str,default="",help="path to store the eval file[for testing]")


	#training detail
	parser.add_argument('--batch_size',type=int,default=20)
	parser.add_argument('--val_num_batches',type=int,default=100,help="eval during training, get how many batch in train/val to eval")

	parser.add_argument("--num_epochs",type=int,default=20) # num_step will be num_example/batch_size * epoch

	#------------------------------------------ all kinds of threshold

	# cap of the word
	parser.add_argument('--word_count_thres',default=2,type=int,help="word count threshold")
	parser.add_argument('--char_count_thres',default=10,type=int,help="char count threshold")

	parser.add_argument('--sent_album_title_size_thres',default=10,type=int,help="max sentence word count for album_title")
	parser.add_argument('--sent_photo_title_size_thres',default=8,type=int,help="max sentence word count for photo_title")
	parser.add_argument('--sent_des_size_thres',default=10,type=int,help="max sentence word count for album_description")
	parser.add_argument('--sent_when_size_thres',default=4,type=int,help="max sentence word count for album_when")
	parser.add_argument('--sent_where_size_thres',default=4,type=int,help="max sentence word count for album_where")
	parser.add_argument('--answer_size_thres',default=5,type=int,help="answer word count")
	parser.add_argument('--question_size_thres',default=25,type=int,help="max question word count")
	parser.add_argument('--word_size_thres',default=16,type=int,help="max word character count")

	parser.add_argument("--num_photos_thres",default=10,type=int,help="maximum photo number per album")
	parser.add_argument("--num_albums_thres",default=8,type=int,help="maximum album number")


	# model detail
	parser.add_argument('--hidden_size',type=int,default=100)

	# whether to use char emb
	parser.add_argument("--use_char",default=False,action="store_true",help="use character CNN embeding")
	# char embeding size
	parser.add_argument('--char_emb_size',default=8,type=int,help="char-CNN channel size")
	parser.add_argument("--char_out_size",default=100,type=int,help="char-CNN output size for each word")


	parser.add_argument("--concat",default=False,action="store_true",help="For simple lstm exp, whether to concat all modality")
	

	# drop out rate
	parser.add_argument('--keep_prob',default=1.0,type=float,help="1.0 - drop out rate;remember to set it to 1.0 in eval")

	# l2 weight decay rate
	parser.add_argument("--wd",default=None,type=float,help="l2 weight decay loss, 0.002 is a good number, default not applied")

	parser.add_argument("--image_feat_dim",default=3048,type=int,help="image feature length")
	# use linear transform from image_feat_dim to image_tran_dim
	parser.add_argument("--use_image_trans",default=False,action="store_true",help="use image transform from image_feat_dim to image_tran_dim")
	parser.add_argument("--image_trans_dim",default=300,type=int,help="image transformed feature length")


	# training parameters
	parser.add_argument("--init_lr",default=0.5,type=float,help=("Start learning rate"))


	# -------------------------------- abalaion
	parser.add_argument("--use_ml_att",default=False,action="store_true",help="use multi-level attention")
	parser.add_argument("--use_tgif_ml_att",default=False,action="store_true",help="use TGIF temporal attention")
	parser.add_argument("--use_mm_att",default=False,action="store_true",help="use multi-modal attention")
	parser.add_argument("--use_bidirection",default=False,action="store_true",help="bidirectional attention in all attention layer")

	parser.add_argument("--use_time_warp",default=False,action="store_true",help="whether to use time warping to get new context")

	parser.add_argument("--warp_type",default=1,type=int,help="time warping type,1:all,2:current, 3:past,4:future,5:past-future")

	parser.add_argument("--use_time_warp_att",default=False,action="store_true",help="whether to use time warping to get attention logit")


	parser.add_argument("--add_tanh",default=False,action="store_true",help="whether to add tanh activation on attention layer")

	

	parser.add_argument("--simiMatrix",default=1,type=int,help="similarity matrixed used in attention layer, 1:(h,q,h*q), 2:((h-q)^2,h*q), 3:1 & 2, 4: cosine")

	parser.add_argument("--use_direct_links",action="store_true",help="whether to use direct links to attend all info and add to g1")
	parser.add_argument("--direct_links_only",action="store_true",help="")

	parser.add_argument("--use_question_att",default=False,action="store_true",help="use info reqresentation to attend question")

	parser.add_argument("--use_choices_att",default=False,action="store_true",help="use question reqresentation to attend choices")

	parser.add_argument("--use_eu_output",action="store_true",help="whether to add eu similarity in the output layer")

	parser.add_argument("--is_pack_model",action="store_true",default=False,help="with is_test, this will pack the model to a path instead of testing")
	parser.add_argument("--pack_model_path",type=str,default=None,help="path to save model")
	parser.add_argument("--pack_model_note",type=str,default=None,help="leave a note for this packed model for future reference")

	args = parser.parse_args()

	if args.is_pack_model:
		assert args.is_test,"use pack model with is_test"
		assert args.pack_model_path is not None, "please provide where pack model to"
		assert args.pack_model_note is not None, "please provide some note for the packed model"

	if args.dmnplus or args.mcb:
		if args.dmnplus:
			from model_dmnplus import get_model 
		else:
			from model_mcb import get_model 
	else:
		if args.use_3d:
			from model_v2 import get_model # using the attention cube
		else:
			from model import get_model

	args.outpath = os.path.join(args.outbasepath,args.modelname,str(args.runId).zfill(2))
	mkdir(args.outpath)

	args.save_dir = os.path.join(args.outpath, "save")#,"save" # tf saver will be save/save-*.meta
	mkdir(args.save_dir)
	args.save_dir_model = os.path.join(args.save_dir,"save") # tf saver will be save/save-*step*.meta

	args.save_dir_best = os.path.join(args.outpath, "best")
	mkdir(args.save_dir_best)
	args.save_dir_best_model = os.path.join(args.save_dir_best,"save-best")

	args.write_self_sum = True
	args.self_summary_path = os.path.join(args.outpath,"train_sum.txt")

	
	if args.load_best:
		args.load = True


	# if test, has to load
	if not args.is_train:
		assert args.is_test, "if not train, please use is_test flag"
		args.load = True
		args.num_epochs = 1
		args.keep_prob = 1.0
		#assert args.val_path!="","Please provide val_path"
		if args.val_path == "":
			if args.load_best:
				args.val_path = os.path.join(args.outpath,"test_best")
			else:
				args.val_path = os.path.join(args.outpath,"test")
		print "test result will be in %s"% args.val_path
		mkdir(args.val_path)

		args.vis_path = os.path.join(args.val_path,"vis")
		args.weights_path = os.path.join(args.val_path,"weights")
		if args.is_save_vis:
			mkdir(args.vis_path)
			print "visualization output will be in  %s"% args.vis_path
		if args.is_save_weights:
			mkdir(args.weights_path)
			print "model weights will be in %s"% args.weights_path

	return args




def read_data(config,datatype,loadExistModelShared=False): 
	data_path = os.path.join(config.prepropath,"%s_data.p"%datatype)
	shared_path = os.path.join(config.prepropath,"%s_shared.p"%datatype)

	with open(data_path,"rb")as f:
		data = pickle.load(f)
	with open(shared_path,"rb") as f:
		shared = pickle.load(f) # this will be added later with word id, either new or load from exists

	num_examples = len(data['q'])
	
	valid_idxs = range(num_examples)

	print "loaded %s/%s data points for %s"%(len(valid_idxs),num_examples,datatype)

	# this is the file for the model' training, with word ID and stuff, if set load in config, will read from existing, otherwise write a new one
	# load the word2idx info into shared[]
	model_shared_path = os.path.join(config.outpath,"shared.p")
	if(loadExistModelShared):
		with open(model_shared_path,"rb") as f:
			model_shared = pickle.load(f)
		for key in model_shared:
			shared[key] = model_shared[key]
	else:
		# no fine tuning of word vector

		# the word larger than word_count_thres and not in the glove word2vec
		# word2idx -> the idx is the wordCounter's item() idx 
		# the new word to index
		# 
		shared['word2idx'] = {word:idx+2 for idx,word in enumerate([word for word,count in shared['wordCounter'].items() if (count > config.word_count_thres) and not shared['word2vec'].has_key(word)])}
		shared['char2idx'] = {char:idx+2 for idx,char in enumerate([char for char,count in shared['charCounter'].items() if count > config.char_count_thres])}
		#print "len of shared['word2idx']:%s"%len(shared['word2idx']) 

		NULL = "<NULL>"
		UNK = "<UNK>"
		shared['word2idx'][NULL] = 0
		shared['char2idx'][NULL] = 0
		shared['word2idx'][UNK] = 1
		shared['char2idx'][UNK] = 1

		# existing word in word2vec will be put after len(new word)+2
		pickle.dump({"word2idx":shared['word2idx'],'char2idx':shared['char2idx']},open(model_shared_path,"wb"))

	# load the word embedding for word in word2vec

	shared['existing_word2idx'] = {word:idx for idx,word in enumerate([word for word in sorted(shared['word2vec'].keys()) if not shared['word2idx'].has_key(word)])}

	# idx -> vector
	idx2vec = {idx:shared['word2vec'][word] for word,idx in shared['existing_word2idx'].items()}
	# load all this vector into a matrix
	# so you can use word -> idx -> vector
	# using xrange(len) so that the idx is 0,1,2,3...
	# then it could be call with embedding lookup with the correct idx

	shared['existing_emb_mat'] = np.array([idx2vec[idx] for idx in xrange(len(idx2vec))],dtype="float32")

	assert config.image_feat_dim == shared['pid2feat'][shared['pid2feat'].keys()[0]].shape[0], ("image dim is not %s, it is %s"%(config.image_feat_dim,shared['pid2feat'][shared['pid2feat'].keys()[0]].shape[0]))

	return Dataset(data,datatype,shared=shared,valid_idxs=valid_idxs)



def train(config):
	self_summary_strs = [] # summary string to print out for later

	# first, read both data and filter stuff,  to get the word2vec idx,
	train_data = read_data(config,'train',config.load)
	val_data = read_data(config,'val',True) # dev should always load model shared data(word2idx etc.) from train

	config_vars = vars(config)
	str_ = "threshold setting--\n" + "\t"+ " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.thresmeta])
	print str_
	self_summary_strs.append(str_)

	# cap the numbers
	# max sentence word count etc.
	update_config(config,[train_data,val_data],showMeta=True) # all word num is <= max_thres   

	str_ = "renewed ----\n"+"\t" + " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.maxmeta])
	print str_
	self_summary_strs.append(str_)


	# now we initialize the matrix for word embedding for word not in glove
	word2vec_dict = train_data.shared['word2vec']
	word2idx_dict = train_data.shared['word2idx'] # this is the word not in word2vec

	# we are not fine tuning , so this should be empty
	idx2vec_dict = {word2idx_dict[word]:vec for word,vec in word2vec_dict.items() if word in word2idx_dict}


	# random initial embedding matrix for new words
	config.emb_mat = np.array([idx2vec_dict[idx] if idx2vec_dict.has_key(idx) else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size)) for idx in xrange(config.word_vocab_size)],dtype="float32") 

	model = get_model(config) # construct model under gpu0

	trainer = Trainer(model,config)
	tester = Tester(model,config)
	saver = tf.train.Saver(max_to_keep=5) # how many model to keep
	bestsaver = tf.train.Saver(max_to_keep=5) # just for saving the best model

	save_period = config.save_period # also the eval period

	# for debug, show the batch content
	if(config.showspecs):
		for batch in train_data.get_batches(2,num_steps=20):

			batchIdx, batchDs = batch

			print "showing a batch with batch_size=2"
			# show each data point
			print "keys:%s"%batchDs.data.keys()
			for key in sorted(batchDs.data.keys()):
				print "\t%s:%s"%(key,batchDs.data[key])

			# show some image feature
			photo_idx1 = batchDs.data['photo_idxs'][0][0][0] # [bacth_num][album_num][photo_num]
			photo_id1 = batchDs.data['photo_ids'][0][0][0]
			photo_idx2 = batchDs.data['photo_idxs'][1][0][0]
			photo_id2 = batchDs.data['photo_ids'][1][0][0]

			print "pidx:%s,pid:%s,feature:\n %s (%s)\n,should be:\n %s (%s)"%(photo_idx1,photo_id1,batchDs.data['pidx2feat'][photo_idx1][:10],batchDs.data['pidx2feat'][photo_idx1].shape,train_data.shared['pid2feat'][photo_id1][:10],train_data.shared['pid2feat'][photo_id1].shape)

			print "pidx:%s,pid:%s,feature:\n %s (%s)\n,should be:\n %s (%s)"%(photo_idx2,photo_id2,batchDs.data['pidx2feat'][photo_idx2][:10],batchDs.data['pidx2feat'][photo_idx2].shape,train_data.shared['pid2feat'][photo_id2][:10],train_data.shared['pid2feat'][photo_id2].shape)

			# get the feed_dict to check
			#feed_dict = model.get_feed_dict(batchDs,is_train=True)

			feed_dict = model.get_feed_dict(batchDs,is_train=False)
			
			sys.exit()

	# start training!
	# allow_soft_placement :  tf will auto select other device if the tf.device(*) not available
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	with tf.Session(config=tfconfig) as sess:

		# calculate total parameters 
		totalParam = cal_total_param()
		str_ = "total parameters: %s"%(totalParam)
		print str_
		self_summary_strs.append(str_)

		initialize(load=config.load,load_best=config.load_best,model=model,config=config,sess=sess)

		# the total step (iteration) the model will run
		last_time = time.time()
		# total / batchSize  * epoch
		num_steps = int(math.ceil(train_data.num_examples/float(config.batch_size)))*config.num_epochs
		# get_batches is a generator, run on the fly
		# there will be num_steps batch
		str_ = " batch_size:%s, epoch:%s,total step:%s,eval/save every %s steps"%(config.batch_size,config.num_epochs,num_steps,config.save_period)
		print str_
		self_summary_strs.append(str_)


		best = {"acc":0.0,"step":-1} # remember the best eval acc during training

		finalAcc = None
		isStart = True

		for batch in tqdm(train_data.get_batches(config.batch_size,num_steps=num_steps),total=num_steps):
			# each batch has (batch_idxs,Dataset(batch_data, full_shared))
			# batch_data has {"q":,"y":..."pidx2feat",.."photo_idxs"..}

			global_step = sess.run(model.global_step) + 1 # start from 0

			# if load from existing model, save if first
			if config.load and isStart:
				tqdm.write("saving original model...")
				tqdm.write("\tsaving model...")
				saver.save(sess,config.save_dir_model,global_step=global_step)
				tqdm.write("\tdone")
				isStart=False

				id2predanswers = {}
				id2realanswers = {}
				for evalbatch in val_data.get_batches(config.batch_size,num_steps=config.val_num_batches,shuffle=False,cap=True):
					yp = tester.step(sess,evalbatch) # [N,4] # id2realanswersprob for each answer
					pred,gt = getAnswers(yp,evalbatch) # from here we get the qid:yindx,
					id2predanswers.update(pred)
					id2realanswers.update(gt)
				evalAcc = getEvalScore(id2predanswers,id2realanswers)
				

				tqdm.write("\teval on validation %s batches Acc:%s, (best:%s at step %s) "%(config.val_num_batches,evalAcc,best['acc'],best['step']))
				# remember the best acc
				if(evalAcc > best['acc']):
					best['acc'] = evalAcc
					best['step'] = global_step
					# save the best model
					tqdm.write("\t saving best model...")
					bestsaver.save(sess,config.save_dir_best_model,global_step=global_step)
					tqdm.write("\t done.")

				finalAcc = evalAcc

			loss,summary,train_op = trainer.step(sess,batch,get_summary=False)

			if global_step % save_period == 0: # time to save model

				duration = time.time() - last_time # in seconds
				sec_per_step = duration/float(save_period)
				last_time = time.time()
				#use tqdm to print
				tqdm.write("step:%s/%s (epoch:%.3f), took %s, loss:%s, estimate remaining:%s"%(global_step,num_steps,(config.num_epochs*global_step/float(num_steps)),sec2time(duration),loss,sec2time((num_steps - global_step)*sec_per_step)))
				tqdm.write("\tsaving model...")
				saver.save(sess,config.save_dir_model,global_step=global_step)
				tqdm.write("\tdone")

				
				id2predanswers = {}
				id2realanswers = {}
				for evalbatch in val_data.get_batches(config.batch_size,num_steps=config.val_num_batches,shuffle=False,cap=True):
					yp = tester.step(sess,evalbatch) # [N,4] # id2realanswersprob for each answer
					pred,gt = getAnswers(yp,evalbatch) # from here we get the qid:yindx,
					id2predanswers.update(pred)
					id2realanswers.update(gt)
				evalAcc = getEvalScore(id2predanswers,id2realanswers)
				

				tqdm.write("\teval on validation %s batches Acc:%s, (best:%s at step %s) "%(config.val_num_batches,evalAcc,best['acc'],best['step']))
				# remember the best acc
				if(evalAcc > best['acc']):
					best['acc'] = evalAcc
					best['step'] = global_step
					# save the best model
					tqdm.write("\t saving best model...")
					bestsaver.save(sess,config.save_dir_best_model,global_step=global_step)
					tqdm.write("\t done.")

				finalAcc = evalAcc

		if global_step % save_period != 0: # time to save model
			saver.save(sess,config.save_dir_model,global_step=global_step)
		str_ = "best eval on val Accurucy: %s at %s step, final step %s Acc is %s"%(best['acc'],best['step'], global_step,finalAcc)
		print str_
		self_summary_strs.append(str_)
		if config.write_self_sum:
			f = open(config.self_summary_path,"w")
			f.writelines("%s"%("\n".join(self_summary_strs)))
			f.close()



def test(config):
	if config.is_test_on_val:
		test_data = read_data(config,'val',True)
		print "total val samples:%s"%test_data.num_examples
	else:
		test_data = read_data(config,'test',True) # here will load shared.p from config.outpath (outbase/modelname/runId/)
		print "total test samples:%s"%test_data.num_examples
	# get the max_sent_size and other stuff
	print "threshold setting--"
	config_vars = vars(config)
	print "\t"+ " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.thresmeta])

	# cap the numbers
	update_config(config,[test_data],showMeta=True)

	# a hack for dmn model, since we fix the number of album, we need to keep the same during test
	if config.dmnplus:
		config.max_num_albums = config.num_albums_thres

	print "renewed ----"
	print "\t" + " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.maxmeta])

	model = get_model(config)

	# update each batch forward into this dict
	id2predanswers = {}
	id2realanswers = {}
	id2yp = {}

	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	# or you can set hard limit

	with tf.Session(config=tfconfig) as sess:
		initialize(load=True,load_best=config.load_best,model=model,config=config,sess=sess)

		if config.is_pack_model:
			saver = tf.train.Saver()
			global_step = model.global_step
			# put input and output to a universal name for reference when in deployment
				# find the nessary stuff in model.get_feed_dict

			# multiple input
			tf.add_to_collection("at",model.at)
			tf.add_to_collection("at_c",model.at_c)
			tf.add_to_collection("at_mask",model.at_mask)

			tf.add_to_collection("ad",model.ad)
			tf.add_to_collection("ad_c",model.ad_c)
			tf.add_to_collection("ad_mask",model.ad_mask)

			tf.add_to_collection("when",model.when)
			tf.add_to_collection("when_c",model.when_c)
			tf.add_to_collection("when_mask",model.when_mask)
			tf.add_to_collection("where",model.where)
			tf.add_to_collection("where_c",model.where_c)
			tf.add_to_collection("where_mask",model.where_mask)

			tf.add_to_collection("pts",model.pts)
			tf.add_to_collection("pts_c",model.pts_c)
			tf.add_to_collection("pts_mask",model.pts_mask)

			tf.add_to_collection("pis",model.pis)
			tf.add_to_collection("pis_mask",model.pis_mask)

			tf.add_to_collection("q",model.q)
			tf.add_to_collection("q_c",model.q_c)
			tf.add_to_collection("q_mask",model.q_mask)

			tf.add_to_collection("choices",model.choices)
			tf.add_to_collection("choices_c",model.choices_c)
			tf.add_to_collection("choices_mask",model.choices_mask)

			# image and text feature
			tf.add_to_collection("image_emb_mat",model.image_emb_mat)
			tf.add_to_collection("existing_emb_mat",model.existing_emb_mat)

			# for getting the highest ranked photo
			tf.add_to_collection("att_logits",model.att_logits)
			

			
			

			tf.add_to_collection("is_train",model.is_train) # TODO, change this to a constant 
			tf.add_to_collection("output",model.yp)
			# also save all the model config and note into the model
			pack_model_note = tf.get_variable("model_note",shape=[],dtype=tf.string,initializer=tf.constant_initializer(config.pack_model_note),trainable=False)
			full_config = tf.get_variable("model_config",shape=[],dtype=tf.string,initializer=tf.constant_initializer(json.dumps(vars(config))),trainable=False)

			print "saving packed model"
			# the following wont save the var model_note, model_config that's not in the graph, 
			# TODO: fix this
			"""
			# put into one big file to save
			input_graph_def = tf.get_default_graph().as_graph_def()
			#print [n.name for n in input_graph_def.node]
			 # We use a built-in TF helper to export variables to constants
			output_graph_def = tf.graph_util.convert_variables_to_constants(
				sess, # The session is used to retrieve the weights
				input_graph_def, # The graph_def is used to retrieve the nodes 
				[tf.get_collection("output")[0].name.split(":")[0]] # The output node names are used to select the usefull nodes
			) 
			output_graph = os.path.join(config.pack_model_path,"final.pb")
			# Finally we serialize and dump the output graph to the filesystem
			with tf.gfile.GFile(output_graph, "wb") as f:
				f.write(output_graph_def.SerializeToString())
			print("%d ops in the final graph." % len(output_graph_def.node))
			"""
			# save it into a path with multiple files
			saver.save(sess,
				os.path.join(config.pack_model_path,"final"),
				global_step=global_step)
			print "model saved in %s"%(config.pack_model_path)
			return 

		if config.is_save_weights:
			weight_dict = {}
			weight_sum = open(os.path.join(config.weights_path,"all.txt"),"w")
			for var in tf.trainable_variables():
				shape = var.get_shape()
				weight_sum.writelines("%s %s\n"%(var.name,shape))
				var_val = sess.run(var)
				weight_dict[var.name] = var_val

			np.savez(os.path.join(config.weights_path,"weights.npz"),**weight_dict)
			weight_sum.close()

		last_time = time.time()
		# num_epoch should be 1
		num_steps = int(math.ceil(test_data.num_examples/float(config.batch_size)))*config.num_epochs

		# load the graph and variables
		tester = Tester(model,config,sess)

		count=0
		print "total step:%s"%num_steps
		for batch in tqdm(test_data.get_batches(config.batch_size,num_steps=num_steps,shuffle=False),total=num_steps):
			count+=1
			if config.is_save_vis:
				# save all variables to pickle for visualization
				batchIdxs,batch_data =  batch
				qid = batch_data.data['qid']
				yp,C,C_win,att_logits,q_att_logits,at_mask,ad_mask,when_mask,where_mask,pts_mask,pis_mask,q_mask,hat_len,had_len,hwhen_len,hwhere_len,hpts_len,hpis_len,JXP,warp_h,h,at,ad,when,where,pts,pis,q = tester.step_vis(sess,batch)
				# each batch save as a pickle file
				# these all should have the same order
				vis = {"yp":yp,"data":batch_data.data,"C":C,"C_win":C_win,"att_logits":att_logits,"q_att_logits":q_att_logits,"at_mask":at_mask,"ad_mask":ad_mask,"when_mask":when_mask,"where_mask":where_mask,"pts_mask":pts_mask,"pis_mask":pis_mask,"q_mask":q_mask,"hat_len":hat_len,"had_len":had_len,"hwhen_len":hwhen_len,'hwhere_len':hwhere_len,"hpts_len":hpts_len,"hpis_len":hpis_len,"photo_title_len":JXP,"warp_h":warp_h,"h":h,"at":at,"ad":ad,"when":when,"where":where,"pts":pts,"pis":pis,"q":q}
				"""
				print batch_data.data['qid']
				print batch_data.data['q']
				batch_data.data['cs'][0].insert(batch_data.data['yidx'][0],batch_data.data['y'][0])
				print batch_data.data['cs'][0]
				print batch_data.data['photo_ids']
				print yp
				print batch_data.data['pidx2feat']
				print batch_data.data['photo_idxs']
				sys.exit()
				"""

				pickle.dump(vis,open(os.path.join(config.vis_path,"%s.p"%count),"wb"))
			else:
				yp = tester.step(sess,batch) # [N,4] # id2realanswersprob for each answer

			if config.get_yp:
				pred,gt,yp = getAnswers_yp(yp,batch)
				id2yp.update(yp)
			else:
				pred,gt = getAnswers(yp,batch) # from here we get the qid:yindx,
			id2predanswers.update(pred)
			id2realanswers.update(gt)

	acc = getEvalScore(id2predanswers,id2realanswers)
	print "done, got %s answers, accuracy:%s"%(len(id2predanswers),acc)
	json.dump(id2predanswers,open("%s/answers.json"%config.val_path,"w"))
	if config.get_yp:
		json.dump({id_:"%s"%(id2yp[id_]) for id_ in id2yp},open("%s/yps.json"%config.val_path,"w"))


def initialize(load,load_best,model,config,sess):
	tf.global_variables_initializer().run()
	if load:
		#print len(tf.global_variables())
		#print [var.name for var in tf.global_variables()]
		# var_name to the var object
		vars_ = {var.name.split(":")[0]: var for var in tf.global_variables()}

		saver = tf.train.Saver(vars_, max_to_keep=5)

		# load the lateste model

		ckpt = tf.train.get_checkpoint_state(config.save_dir)
		if load_best:
			#loadpath = config.save_dir_model + "-best-0"
			#loadpath = config.save_dir_best_model + "-best-0"
			ckpt = tf.train.get_checkpoint_state(config.save_dir_best)

		if ckpt and ckpt.model_checkpoint_path:
			loadpath = ckpt.model_checkpoint_path
			saver.restore(sess, loadpath)
			print "Model:"
			print "\tloaded %s"%loadpath
			print ""
		else:
			raise Exception("Model not exists %s"%(ckpt))



# https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
def cal_total_param():
	total = 0
	for var in tf.trainable_variables():
		shape = var.get_shape()
		var_num = 1
		for dim in shape:
			var_num*=dim.value
		total+=var_num
	return total


if __name__ == "__main__":
	config = get_args()
	# some useful info of the dataset
	config.thresmeta = (
		"sent_album_title_size_thres",
		"sent_photo_title_size_thres",
		"sent_des_size_thres",
		"sent_when_size_thres",
		"sent_where_size_thres",
		"answer_size_thres",
		"question_size_thres",
		"num_photos_thres",
		"num_albums_thres",
		"word_size_thres"
	)
	config.maxmeta = (
			"max_num_albums",
			"max_num_photos",
			"max_sent_album_title_size",
			"max_sent_photo_title_size",
			"max_sent_des_size",
			"max_when_size",
			"max_where_size",
			"max_answer_size",
			"max_question_size",
			"max_word_size"
	)
	if config.is_train:
		train(config)
	else:
		test(config)
