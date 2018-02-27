# coding=utf-8
# author: Junwei Liang junweil@cs.cmu.edu
# preprocess all usefull information into 3 set 

d = "giving the original memoryqa dataset, will generate a *_data.p, *_shared.p for each split. tokenized question, answer, pointer to album will be in data, wordcounter, album will be in shared"

import os,sys,json,re
import argparse,nltk
import numpy as np
from collections import Counter
import cPickle as pickle

def get_args():
	parser = argparse.ArgumentParser(description=d)
	#parser.add_argument("datapath",action="store",type=str,help="/path/to/dataset(qa.json,album_info.json,bad_id,test.id)")
	parser.add_argument("datajson",type=str,help="path to the qas.json")
	parser.add_argument("albumjson",type=str,help="path to album_info.json")
	parser.add_argument("testids",type=str,help="path to test id list")
	parser.add_argument("--valids",type=str,default=None,help="path to validation id list, if not set will be random 20%% of the training set")

	parser.add_argument("imgfeat",action="store",type=str,help="/path/to img feat npz file")
	parser.add_argument("glove",action="store",type=str,help="/path/to glove vector file")
	parser.add_argument("outpath",type=str,help="output path")

	return parser.parse_args()


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

# ------------------------------------------------ for lbum descript html tag
from HTMLParser import HTMLParser
class MLStripper(HTMLParser):
	def __init__(self):
		self.reset()
		self.fed = []
	def handle_data(self, d):
		self.fed.append(d)
	def get_data(self):
		return ''.join(self.fed)

def strip_tags(html):
	s = MLStripper()
	s.feed(html)
	return s.get_data()
# ------------------------------------------------------------


# for each token with "-" or others, remove it and split the token 
def process_tokens(tokens):
	newtokens = []
	l = ("-","/", "~", '"', "'", ":","\)","\(","\[","\]","\{","\}")
	for token in tokens:
		# split then add multiple to new tokens
		newtokens.extend([one for one in re.split("[%s]"%("").join(l),token) if one != ""])
	return newtokens

def l2norm(feat):
	l2norm = np.linalg.norm(feat,2)
	return feat/l2norm

# word_counter words are lowered already
def get_word2vec(args,word_counter):
	word2vec_dict = {}
	import io
	with io.open(args.glove, 'r', encoding='utf-8') as fh:
		for line in fh:
			array = line.lstrip().rstrip().split(" ")
			word = array[0]
			vector = list(map(float, array[1:]))
			if word in word_counter:
				word2vec_dict[word] = vector
			#elif word.capitalize() in word_counter:
			#	word2vec_dict[word.capitalize()] = vector
			elif word.lower() in word_counter:
				word2vec_dict[word.lower()] = vector
			#elif word.upper() in word_counter:
			#	word2vec_dict[word.upper()] = vector

	#print "{}/{} of word vocab have corresponding vectors ".format(len(word2vec_dict), len(word_counter))
	return word2vec_dict

from tqdm import tqdm
def prepro_each(args,data_type,question_ids,start_ratio=0.0,end_ratio=1.0):
	debug = False
	sent_tokenize = nltk.sent_tokenize
	sent_tokenize = lambda para:[para] # right now we don't do sentence tokenization # just for album_description
	def word_tokenize(tokens):
		# nltk.word_tokenize will split ()
		# "a" -> '``' + a + "''"
		# lizzy's -> lizzy + 's
		# they're -> they + 're
		# then we remove and split "-"
		return process_tokens([token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)])

	qas = {str(qa['question_id']):qa for qa in args.qas}

	global_aids = {} # all the album Id the question used, also how many question used that album

	q,cq,y,cy,aid,qid,cs,ccs,idxs,yidx = [],[],[],[],[],[],[],[],[],[] # choices, char choices
	word_counter,char_counter = Counter(),Counter() # lower word counter

	start_idx = int(round(len(question_ids) * start_ratio))
	end_idx = int(round(len(question_ids) * end_ratio))

	# go through all question first, then albums
	for idx,question_id in enumerate(tqdm(question_ids[start_idx:end_idx])):
		assert isinstance(question_id,str)
		qa = qas[question_id]

		# question
		qi = word_tokenize(qa['question']) # no lower here
		cqi = [list(qij) for qij in qi]

		for qij in qi:
			word_counter[qij.lower()] += 1
			for qijk in qij:
				char_counter[qijk] += 1

		# album ids
		for albumId in qa['album_ids']:
			albumId = str(albumId)
			if(not global_aids.has_key(albumId)):
				global_aids[albumId] = 0
			global_aids[albumId]+=1 # remember how many times this album is used

		# answer, choices
		yi = word_tokenize(qa['answer'])
		cyi = [list(yij) for yij in yi]
		for yij in yi:
			word_counter[yij.lower()] += 1
			for yijk in yij:
				char_counter[yijk] +=1


		ci = qa['multiple_choices_4'][:] # copy it
		# remove the answer in choices
		yidxi = ci.index(qa['answer']) # this is for during testing, we need to reconstruct the answer in the original order
		ci.remove(qa['answer']) # will error if answer not in choice
		assert len(ci) == 3
		cci = [] # char for choices
		for i,c in enumerate(ci):
			ci[i] = word_tokenize(c)
			cci.append([list(ciij) for ciij in ci[i]])
			for ciij in ci[i]:
				word_counter[ciij.lower()]+=1
				for ciijk in ciij:
					char_counter[ciijk]+=1
		
		# for debug
		if(debug):
			print "questiion:%s"%qa['question']
			print qi
			print cqi
			print "answer:%s"%(qa['answer'])
			print yi
			print cyi
			print "choices:%s"%("/".join(qa['multiple_choices_4']))
			print ci
			print cci
			break

		q.append(qi)
		cq.append(cqi)
		y.append(yi)
		cy.append(cyi)
		yidx.append(yidxi)
		cs.append(ci)
		ccs.append(cci)
		aid.append([str(one) for one in qa['album_ids']])
		qid.append(question_id)
		idxs.append(idx) # increament index for each qa

	# get the shared now
	albums = {str(album['album_id']):album for album in args.albums}
	album_info = {}
	pid2feat = {}
	for albumId in tqdm(global_aids):
		album = albums[albumId]
		used = global_aids[albumId]

		temp = {'aid':album['album_id']}

		# album info
		temp['title'] = word_tokenize(album['album_title'])
		temp['title_c'] = [list(tok) for tok in temp['title']]
		#temp['description'] = list(map(word_tokenize, sent_tokenize(strip_tags(album['album_description']))))
		# treat description as one sentence
		temp['description'] = word_tokenize(strip_tags(album['album_description']))
		temp['description_c'] = [list(tok) for tok in temp['description']]

		# use _ to connect?
		if album['album_where'] is None:
			temp['where'] = []
			temp['where_c'] = []
		else:
			temp['where'] = word_tokenize(album['album_where'])
			temp['where_c'] = [list(tok) for tok in temp['where']]
		temp['when'] = word_tokenize(album['album_when'])
		temp['when_c'] = [list(tok) for tok in temp['when']]

		# photo info
		temp['photo_titles'] = [word_tokenize(title) for title in album['photo_titles']]
		temp['photo_titles_c'] = [[list(tok) for tok in title] for title in temp['photo_titles']]

		# no feat for each album, we keep the photoId here
		# another dict for pid2feat
		#temp['photo_feats'] = [l2norm(args.images[str(pid)]) for pid in album['photo_ids']]

		temp['photo_ids'] = [str(pid) for pid in album['photo_ids']]
		assert len(temp['photo_ids']) == len(temp['photo_titles'])
		for pid in temp['photo_ids']:
			assert isinstance(pid, str)
			if(not pid2feat.has_key(pid)):
				#if(args.imageispca): # pca feature needs not l2norm
				#	pid2feat[pid] = args.images[pid]
				#else:
				#	pid2feat[pid] = l2norm(args.images[pid])
				# feature itself should be l2normed first
				pid2feat[pid] = args.images[pid]

		#assert len(temp['photo_feats']) == len(temp['photo_titles'])

		if(debug):
			print "album title:%s"%album['album_title']
			print temp['title']
			print temp['title_c']
			print "album description:%s"%album['album_description']
			print temp['description']
			print temp['description_c']
			print "album when:%s,where:%s"%(album['album_when'],album['album_where'])
			print temp['when'],temp['where']
			print temp['when_c'],temp['where_c']
			print "album photo tile 1:%s"%album['photo_titles'][0]
			print temp['photo_titles'][0]
			print temp['photo_titles_c'][0]
			sys.exit()

		#print [tok for title in temp['photo_titles'] for tok in title ]
		for t in temp['title'] + temp['description'] + temp['where'] + temp['when'] + [tok for title in temp['photo_titles'] for tok in title ]:
			#print t
			word_counter[t.lower()] += used
			for c in t:
				char_counter[c] += used
		
		album_info[albumId] = temp

	word2vec_dict = get_word2vec(args,word_counter)

	#q,cq,y,cy,aid,qid,cs,ccs,idxs 
	data = {
		'q':q,
		'cq':cq,
		'y':y,
		'cy':cy,
		'yidx': yidx,# the original answer idx in the choices list # this means the correct index
		'aid':aid, # each is a list of aids
		'qid':qid,
		'idxs':idxs,
		'cs':cs, # each is a list of wrong choices
		'ccs':ccs,
	}

	shared = {
		"albums" :album_info, # albumId -> photo_ids/title/when/where ...
		"pid2feat":pid2feat, # pid -> image feature
		"wordCounter":word_counter,
		"charCounter":char_counter,
		"word2vec":word2vec_dict
	}
	print "data:%s, char entry:%s, word entry:%s, word2vec entry:%s,album: %s/%s, image_feat:%s"%(data_type,len(char_counter),len(word_counter),len(word2vec_dict),len(album_info),len(albums),len(pid2feat))

	pickle.dump(data,open(os.path.join(args.outpath,"%s_data.p"%data_type),"wb"))
	pickle.dump(shared,open(os.path.join(args.outpath,"%s_shared.p"%data_type),"wb"))



def getTrainValIds(qas,validlist,testidlist):
	testIds = [one.strip() for one in open(testidlist,"r").readlines()]

	valIds = []
	if validlist is not None:
		valIds = [one.strip() for one in open(validlist,"r").readlines()]
	
	trainIds = []
	
	for one in qas:
		qid = str(one['question_id'])
		if((qid not in testIds) and (qid not in valIds)):
			trainIds.append(qid)

	# if validation id not provided, get from trainIds
	if validlist is None:
		valcount = int(len(trainIds)*0.2)
		random.seed(1)
		random.shuffle(trainIds)
		random.shuffle(trainIds)
		valIds = trainIds[:valcount]
		trainIds = trainIds[valcount:]

	print "total trainId:%s,valId:%s,testId:%s, total qa:%s"%(len(trainIds),len(valIds),len(testIds),len(qas))
	return trainIds,valIds,testIds


import random
import cPickle as pickle
if __name__ == "__main__":
	args = get_args()
	mkdir(args.outpath)
	
	
	# get the qids for training
	args.qas = json.load(open(args.datajson,"r"))
	args.albums = json.load(open(args.albumjson,"r"))

	# if the image is a .p file, then we will read it differently
	if(args.imgfeat.endswith(".p")):
		print "read pickle image feat."
		imagedata = pickle.load(open(args.imgfeat,"r"))
		args.images = {}
		assert len(imagedata[0]) == len(imagedata[1])
		for i,pid in enumerate(imagedata[0]):
			args.images[pid] = imagedata[1][i]

	else:
		print "read npz image feat."
		args.images = np.load(args.imgfeat)


	trainIds,valIds,testIds = getTrainValIds(args.qas,args.valids,args.testids)

	prepro_each(args,"train",trainIds,0.0,1.0)
	prepro_each(args,"val",valIds,0.0,1.0)
	prepro_each(args,"test",testIds,0.0,1.0)

	
	