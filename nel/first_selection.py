import nel.utils as utils
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

voca_emb_dir = 'data/generated/embeddings/word_ent_embs/'
wiki_prefix = 'en.wikipedia.org/wiki/'

#first choose csim top, then append the top p_e_m
class choose_cands:
	def __init__(self):
		self.entity_voca, self.entity_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.entity',
													voca_emb_dir + 'entity_embeddings.npy')
	def ment_cos(self, content, alpha, chosed_top, chose_num): # one doc
		ment_cands = [] #mention cands embedding
		ment_name = [] #mention name
		cands = {}
		chosed = {} # we just need the cands_name and p_e_m
		for m in content:
			chosed_top_cands = []
			chosed_top_p_e_m = [] 

			named_cands = [c[0] for c in m['candidates']] #all cands
			p_e_m = [min(1., max(1e-3, c[1])) for c in m['candidates']]
			cands_id = [self.entity_voca.get_id(wiki_prefix + c) for c in named_cands]
			cands_emb = [self.entity_embeddings[id] for id in cands_id]

			cands.setdefault(m['mention'], {'named_cands': named_cands,
											'p_e_m' : p_e_m,
											'cands_emb' : cands_emb})
			if len(named_cands) != 0:
				ment_name += [m['mention']]
				ment_cands += [cands_emb]
			else:
				chosed.setdefault(m['mention'], {'named_cands': [],
												 'p_e_m' : []})

		#we use dict to delete repeat mention
		#now we want to have the original mention order to cal the cos
		ment_order = list(set(ment_name))
		ment_order.sort(key= ment_name.index)

		#cal the cos and put the choose cands into dict
		for m1 in ment_order:
			if len(cands[m1]['named_cands']) <= chose_num : #if cands <30 then we just choose all
				chosed.setdefault(m1, {'named_cands': cands[m1]['named_cands'],
									   'p_e_m' : cands[m1]['p_e_m']})
				continue
			# if doc only have one mention then we can't cal the similarity between mention 
			# we now just chosed top 30 candidates
			if len(ment_order) == 1: 
				chosed.setdefault(m1, {'named_cands':[cands[m1]['named_cands'][i] for i in range(chose_num)],
									   'p_e_m' :[cands[m1]['p_e_m'][i] for i in range(chose_num)]})
				continue
			link_num = []
			for m2 in ment_order:
				if(m1 != m2):
					try:
						count = np.array(cosine_similarity(cands[m1]['cands_emb'],cands[m2]['cands_emb']))
						if(alpha != 0):
							count[count >= alpha] = 1
							count[count < alpha] = 0
						link_num += [np.sum(count, axis = 1)]
					except: #if mention only have 1 candidate
						count = np.array(cosine_similarity(cands[m1]['cands_emb'],[cands[m2]['cands_emb']]))
						if(alpha != 0):
							count[count >= alpha] = 1
							count[count < alpha] = 0
						link_num += [count]
			link_num = torch.tensor(np.sum(link_num, axis = 0))
			_, top_ids = torch.topk(link_num, k=chose_num - chosed_top) # _ is topk data, top_ids is topk index
			top_ids = sorted(top_ids.numpy()) #sorted the index we chose cause we need the p_e_m order

			selected = set(top_ids)
			idx = 0
			while len(selected) < chose_num:
				if idx not in selected:
					selected.add(idx)
				idx += 1
			selected = sorted(list(selected))

			chosed.setdefault(m1, {'named_cands': [cands[m1]['named_cands'][i] for i in selected],
								   'p_e_m' : [cands[m1]['p_e_m'][i] for i in selected]})

		return chosed