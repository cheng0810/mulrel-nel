import nel.ntee as ntee
from nel.vocabulary import Vocabulary
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import nel.dataset as D
from tqdm import tqdm

from nel.abstract_word_entity import load as load_model
from nel.mulrel_ranker import MulRelRanker
from nel.first_selection import choose_cands
import nel.utils as utils

from random import shuffle
import torch.optim as optim

from pprint import pprint

from hanziconv import HanziConv
import jieba
import time
import pickle


ModelClass = MulRelRanker
wiki_prefix = 'en.wikipedia.org/wiki/'
preprocessing_path = 'nel/preprocessing/'

class EDRanker:
    """
    ranking candidates
    """

    def __init__(self, config):
        print('--- create model ---')

        config['entity_embeddings'] = config['entity_embeddings'] / \
                                      np.maximum(np.linalg.norm(config['entity_embeddings'],
                                                                axis=1, keepdims=True), 1e-12)
        config['entity_embeddings'][config['entity_voca'].unk_id] = 1e-10
        config['word_embeddings'] = config['word_embeddings'] / \
                                    np.maximum(np.linalg.norm(config['word_embeddings'],
                                                              axis=1, keepdims=True), 1e-12)
        config['word_embeddings'][config['word_voca'].unk_id] = 1e-10

        print('prerank model')
        self.prerank_model = ntee.NTEE(config)
        self.args = config['args']

        print('main model')
        # by Cheng
        if self.args.mode == 'pretrain':
            self.model = pretrain_cands(config)
        elif self.args.mode == 'eval':
            print('try loading model from', self.args.model_path)
            self.model = load_model(self.args.model_path, ModelClass)
        else:
            print('create new model')
            if config['mulrel_type'] == 'rel-norm':
                config['use_stargmax'] = False
            if config['mulrel_type'] == 'ment-norm':
                config['first_head_uniform'] = False
                config['use_pad_ent'] = True

            config['use_local'] = True
            config['use_local_only'] = False
            config['oracle'] = False
            self.model = ModelClass(config)

        self.prerank_model.cuda()
        self.model.cuda()

    def prerank(self, dataset, predict=False):
        new_dataset = []
        has_gold = 0
        total = 0
        f = 0

        for content in dataset: #content is a doc(include mentions)
            items = []
            f+=1
            if self.args.keep_ctx_ent > 0:
                # rank the candidates by ntee scores
                lctx_ids = [m['context'][0][max(len(m['context'][0]) - self.args.prerank_ctx_window // 2, 0):]
                            for m in content]
                rctx_ids = [m['context'][1][:min(len(m['context'][1]), self.args.prerank_ctx_window // 2)]
                            for m in content]
                ment_ids = [[] for m in content] #meaning? 
                token_ids = [l + m + r if len(l) + len(r) > 0 else [self.prerank_model.word_voca.unk_id]
                             for l, m, r in zip(lctx_ids, ment_ids, rctx_ids)] #combine all the left right context wiki id
                entity_ids = [m['cands'] for m in content]
                entity_ids = Variable(torch.LongTensor(entity_ids).cuda())

                entity_mask = [m['mask'] for m in content]
                entity_mask = Variable(torch.FloatTensor(entity_mask).cuda())

                token_ids, token_offsets = utils.flatten_list_of_lists(token_ids)
                token_offsets = Variable(torch.LongTensor(token_offsets).cuda())
                token_ids = Variable(torch.LongTensor(token_ids).cuda())

                log_probs = self.prerank_model.forward(token_ids, token_offsets, entity_ids, use_sum=True) 
                log_probs = (log_probs * entity_mask).add_((entity_mask - 1).mul_(1e10)) #use mask to let the unk_id won't be choose
                #topk will return (score,order of the score) 
                #so the model didn't use the log_prob to train the model just choose the candidates
                _, top_pos = torch.topk(log_probs, dim=1, k=self.args.keep_ctx_ent)
                top_pos = top_pos.data.cpu().numpy()
            else:
                top_pos = [[]] * len(content)

            # select candidats: mix between keep_ctx_ent best candidates (ntee scores) with
            # keep_p_e_m best candidates (p_e_m scores)
            for i, m in enumerate(content): #so m means the mention of this doc
                sm = {'cands': [],
                      'named_cands': [],
                      'p_e_m': [],
                      'mask': [],
                      'true_pos': -1}
                m['selected_cands'] = sm
                selected = set(top_pos[i])
                idx = 0
                while len(selected) < self.args.keep_ctx_ent + self.args.keep_p_e_m:
                    if idx not in selected:
                        selected.add(idx)
                    idx += 1
                selected = sorted(list(selected))
                for idx in selected:
                    sm['cands'].append(m['cands'][idx])
                    sm['named_cands'].append(m['named_cands'][idx])
                    sm['p_e_m'].append(m['p_e_m'][idx])
                    sm['mask'].append(m['mask'][idx])
                    if idx == m['true_pos']:
                        sm['true_pos'] = len(sm['cands']) - 1

                if not predict:
                    if sm['true_pos'] == -1:
                        continue
                        # this insertion only makes the performance worse (why???)
                        # sm['true_pos'] = 0
                        # sm['cands'][0] = m['cands'][m['true_pos']]
                        # sm['named_cands'][0] = m['named_cands'][m['true_pos']]
                        # sm['p_e_m'][0] = m['p_e_m'][m['true_pos']]
                        # sm['mask'][0] = m['mask'][m['true_pos']]
                items.append(m)
                if sm['true_pos'] >= 0:
                    has_gold += 1
                total += 1

                if predict: 
                    # only for oracle model, not used for eval
                    if sm['true_pos'] == -1:
                        sm['true_pos'] = 0  # a fake gold, happens only 2%, but avoid the non-gold

            if len(items) > 0:
                new_dataset.append(items)

        print('recall', has_gold / total)

        return new_dataset

    def get_data_items(self, dataset, data_name, predict=False):
        data = []
        cand_source = 'candidates'
        save_cands = {} # save preprocessing (cheng)

        tStart = time.time()
        for doc_name, content in tqdm(dataset.items()):
            items = []
            conll_doc = content[0].get('conll_doc', None)         
            chosed = choose_cands().ment_cos(content, self.args.cands_threshold, self.args.keep_top, self.args.n_cands_before_rank)
            content_tmp = [] # save preprocessing (cheng)
            for m in content:
                named_cands = chosed[m['mention']]['named_cands']
                p_e_m = chosed[m['mention']]['p_e_m']
                gt_pos = [c[0] for c in m['candidates']]
                gt_p_e_m = [min(1., max(1e-3, c[1])) for c in m['candidates']]

                named_cands_t = [] # avoid variable to change dict value
                named_cands_t += chosed[m['mention']]['named_cands']
                p_e_m_t = [] # avoid variable to change dict value
                p_e_m_t += chosed[m['mention']]['p_e_m']

                try:
                    true_pos = named_cands_t.index(m['gold'][0])
                    p = p_e_m_t[true_pos]
                except:
                    # now we are not choose base on top30, so the list chosen by our way could be[0,1,2,6,15,17,...], 
                    # but we let the list to reorder by[0,1,2,...,29],
                    # when we didn't choose the gt, we let the gt to be the order 31, cause gt could be the order 4 in the oringinal order,
                    # but we just choose [0,1,2,6,...](i.e. 4 is not inside), when the model is training we want every mention could have gt,
                    # so we do this.
                    if m['gold'][0] in gt_pos:
                        true_pos = gt_pos.index(m['gold'][0])
                        p = gt_p_e_m[true_pos]
                        true_pos = len(named_cands_t) + 1
                    else:
                        true_pos = -1

                content_tmp.append({'true_pos':true_pos, 'p_e_m':p_e_m, 'named_cands':named_cands}) # save preprocessing (cheng)

                #while is training change the last cand to gold
                if true_pos >= len(named_cands_t):
                    if not predict: 
                        true_pos = len(named_cands_t) - 1
                        p_e_m_t[-1] = p
                        named_cands_t[-1] = m['gold'][0]
                    else: # if is on predict then the ture_position is not exist 
                        true_pos = -1 

                cands = [self.model.entity_voca.get_id(wiki_prefix + c) for c in named_cands_t]
                mask = [1.] * len(cands)
                if len(cands) == 0 and not predict:
                    continue
                elif len(cands) < self.args.n_cands_before_rank: # if len(cands) < top 30 candidate, then padding unk candidate to array
                    cands += [self.model.entity_voca.unk_id] * (self.args.n_cands_before_rank - len(cands)) #cands represent candidate wili id 
                    named_cands_t += [Vocabulary.unk_token] * (self.args.n_cands_before_rank - len(named_cands_t)) #named_cands represent candidate's name
                    p_e_m_t += [1e-8] * (self.args.n_cands_before_rank - len(p_e_m_t))
                    mask += [0.] * (self.args.n_cands_before_rank - len(mask)) #if exist mask = 1 else =0

                if self.args.language == 'en':
                    lctx = m['context'][0].strip().split()
                    lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                    lctx_ids = [tid for tid in lctx_ids if tid != self.prerank_model.word_voca.unk_id] #drop unk id word
                    lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.args.ctx_window//2):] #if lctx len >50 then drop the word before 50 words

                    rctx = m['context'][1].strip().split()
                    rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                    rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                    rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window//2)]
                elif self.args.language == 'zh':
                    lctx = HanziConv.toSimplified(m['context'][0].strip())
                    lctx = jieba.lcut(lctx)
                    lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                    lctx_ids = [tid for tid in lctx_ids if tid != self.prerank_model.word_voca.unk_id]
                    lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.args.ctx_window//2):]

                    rctx = HanziConv.toSimplified(m['context'][1].strip())
                    rctx = jieba.lcut(rctx)
                    rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                    rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                    rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window//2)]                    

                ment = m['mention'].strip().split()
                ment_ids = [self.prerank_model.word_voca.get_id(t) for t in ment if utils.is_important_word(t)]
                ment_ids = [tid for tid in ment_ids if tid != self.prerank_model.word_voca.unk_id]

                m['sent'] = ' '.join(lctx + rctx)

                # secondary local context (for computing relation scores)
                #snd_local context only have small len(before '')
                if conll_doc is not None:
                    conll_m = m['conll_m']
                    sent = conll_doc['sentences'][conll_m['sent_id']]
                    start = conll_m['start']
                    end = conll_m['end']

                    snd_lctx = [self.model.snd_word_voca.get_id(t)
                                for t in sent[max(0, start - self.args.snd_local_ctx_window//2):start]]
                    snd_rctx = [self.model.snd_word_voca.get_id(t)
                                for t in sent[end:min(len(sent), end + self.args.snd_local_ctx_window//2)]]
                    snd_ment = [self.model.snd_word_voca.get_id(t)
                                for t in sent[start:end]]

                    if len(snd_lctx) == 0:
                        snd_lctx = [self.model.snd_word_voca.unk_id]
                    if len(snd_rctx) == 0:
                        snd_rctx = [self.model.snd_word_voca.unk_id]
                    if len(snd_ment) == 0:
                        snd_ment = [self.model.snd_word_voca.unk_id]
                else:
                    snd_lctx = [self.model.snd_word_voca.unk_id]
                    snd_rctx = [self.model.snd_word_voca.unk_id]
                    snd_ment = [self.model.snd_word_voca.unk_id]

                items.append({'context': (lctx_ids, rctx_ids),
                              'snd_ctx': (snd_lctx, snd_rctx),
                              'ment_ids': ment_ids,
                              'snd_ment': snd_ment,
                              'cands': cands,
                              'named_cands': named_cands_t,
                              'p_e_m': p_e_m_t,
                              'mask': mask,
                              'true_pos': true_pos,
                              'doc_name': doc_name,
                              'raw': m
                              })

            if len(items) > 0:
                # note: this shouldn't affect the order of prediction because we use doc_name to add predicted entities,
                # and we don't shuffle the data for prediction
                if len(items) > 100:
                    print(len(items)) #means this docs have >100 mentions
                    for k in range(0, len(items), 100):
                        data.append(items[k:min(len(items), k + 100)])
                else:
                    data.append(items)

            save_cands[doc_name] = content_tmp # save preprocessing (cheng)

        tEnd = time.time()
        print("It cost %.4f min" % ((tEnd - tStart)/60))

        with open(preprocessing_path + data_name + '.pickle', 'wb') as fp: # save preprocessing (cheng)
            pickle.dump(save_cands, fp, protocol=pickle.HIGHEST_PROTOCOL) 

        return self.prerank(data, predict)


    def get_data_items_load(self, dataset, data_name, predict=False):
        data = []
        cand_source = 'candidates'
        tStart = time.time()
        # load preprocessing pickle
        with open(preprocessing_path + data_name + '.pickle', 'rb') as fp:
            pre_data = pickle.load(fp)

        for doc_name, content in dataset.items():
            items = []
            conll_doc = content[0].get('conll_doc', None)

            count = 0 # for pickle
            for m in content:
                #load pre_data (cheng)
                named_cands = [] # avoid variable to change dict value
                named_cands += pre_data[doc_name][count]['named_cands']
                p_e_m = [] # avoid variable to change dict value
                p_e_m = pre_data[doc_name][count]['p_e_m']
                true_pos = pre_data[doc_name][count]['true_pos']
                gt_pos = [c[0] for c in m['candidates']]
                gt_p_e_m = [min(1., max(1e-3, c[1])) for c in m['candidates']]

                try:
                    true_pos = named_cands.index(m['gold'][0])
                    p = p_e_m[true_pos]
                except:
                    if m['gold'][0] in gt_pos:
                        true_pos = gt_pos.index(m['gold'][0])
                        p = gt_p_e_m[true_pos]
                        true_pos = len(named_cands) + 1
                    else:
                        true_pos = -1

                #while is training change the last cand to gold
                if true_pos >= len(named_cands):
                    if not predict: 
                        true_pos = len(named_cands) - 1
                        p_e_m[-1] = p
                        named_cands[-1] = m['gold'][0]
                    else: # if is on predict then the ture_position is not exist 
                        true_pos = -1 

                cands = [self.model.entity_voca.get_id(wiki_prefix + c) for c in named_cands]
                mask = [1.] * len(cands)
                if len(cands) == 0 and not predict:
                    continue
                elif len(cands) < self.args.n_cands_before_rank: # if len(cands) < top 30 candidate, then padding unk candidate to array
                    cands += [self.model.entity_voca.unk_id] * (self.args.n_cands_before_rank - len(cands)) #cands represent candidate wili id 
                    named_cands += [Vocabulary.unk_token] * (self.args.n_cands_before_rank - len(named_cands)) #named_cands represent candidate's name
                    p_e_m += [1e-8] * (self.args.n_cands_before_rank - len(p_e_m))
                    mask += [0.] * (self.args.n_cands_before_rank - len(mask)) #if exist mask = 1 else =0

                if self.args.language == 'en':
                    lctx = m['context'][0].strip().split()
                    lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                    lctx_ids = [tid for tid in lctx_ids if tid != self.prerank_model.word_voca.unk_id] #drop unk id word
                    lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.args.ctx_window//2):] #if lctx len >50 then drop the word before 50 words

                    rctx = m['context'][1].strip().split()
                    rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                    rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                    rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window//2)]
                elif self.args.language == 'zh':
                    lctx = HanziConv.toSimplified(m['context'][0].strip())
                    lctx = jieba.lcut(lctx)
                    lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                    lctx_ids = [tid for tid in lctx_ids if tid != self.prerank_model.word_voca.unk_id]
                    lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.args.ctx_window//2):]

                    rctx = HanziConv.toSimplified(m['context'][1].strip())
                    rctx = jieba.lcut(rctx)
                    rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                    rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                    rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window//2)]                    

                ment = m['mention'].strip().split()
                ment_ids = [self.prerank_model.word_voca.get_id(t) for t in ment if utils.is_important_word(t)]
                ment_ids = [tid for tid in ment_ids if tid != self.prerank_model.word_voca.unk_id]

                m['sent'] = ' '.join(lctx + rctx)

                # secondary local context (for computing relation scores)
                #snd_local context only have small len(before '')
                if conll_doc is not None:
                    conll_m = m['conll_m']
                    sent = conll_doc['sentences'][conll_m['sent_id']]
                    start = conll_m['start']
                    end = conll_m['end']

                    snd_lctx = [self.model.snd_word_voca.get_id(t)
                                for t in sent[max(0, start - self.args.snd_local_ctx_window//2):start]]
                    snd_rctx = [self.model.snd_word_voca.get_id(t)
                                for t in sent[end:min(len(sent), end + self.args.snd_local_ctx_window//2)]]
                    snd_ment = [self.model.snd_word_voca.get_id(t)
                                for t in sent[start:end]]

                    if len(snd_lctx) == 0:
                        snd_lctx = [self.model.snd_word_voca.unk_id]
                    if len(snd_rctx) == 0:
                        snd_rctx = [self.model.snd_word_voca.unk_id]
                    if len(snd_ment) == 0:
                        snd_ment = [self.model.snd_word_voca.unk_id]
                else:
                    snd_lctx = [self.model.snd_word_voca.unk_id]
                    snd_rctx = [self.model.snd_word_voca.unk_id]
                    snd_ment = [self.model.snd_word_voca.unk_id]

                items.append({'context': (lctx_ids, rctx_ids),
                              'snd_ctx': (snd_lctx, snd_rctx),
                              'ment_ids': ment_ids,
                              'snd_ment': snd_ment,
                              'cands': cands,
                              'named_cands': named_cands,
                              'p_e_m': p_e_m,
                              'mask': mask,
                              'true_pos': true_pos,
                              'doc_name': doc_name,
                              'raw': m
                              })
                count += 1

            if len(items) > 0:
                # note: this shouldn't affect the order of prediction because we use doc_name to add predicted entities,
                # and we don't shuffle the data for prediction
                if len(items) > 100:
                    print(len(items)) #means this docs have >100 mentions
                    for k in range(0, len(items), 100):
                        data.append(items[k:min(len(items), k + 100)])
                else:
                    data.append(items)


        tEnd = time.time()
        print("It cost %.4f min" % ((tEnd - tStart)/60))

        return self.prerank(data, predict)

    def train(self, org_train_dataset, org_dev_datasets, config):
        print('extracting training data')
        if self.args.language == 'en':
            train_dataset = self.get_data_items(org_train_dataset, data_name='aida-train', predict=False)
        elif self.args.language == 'zh':
            train_dataset = self.get_data_items(org_train_dataset, data_name='tackbp2015_train', predict=False)
        print('#train docs', len(train_dataset))

        dev_datasets = []
        for dname, data in org_dev_datasets:
            dev_datasets.append((dname, self.get_data_items(data,dname, predict=True)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

        print('creating optimizer')
        optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr']) #what is the model.parameters()?
        best_f1 = -1
        not_better_count = 0
        is_counting = False
        eval_after_n_epochs = self.args.eval_after_n_epochs

        for e in range(config['n_epochs']):
            shuffle(train_dataset)

            total_loss = 0
            for dc, batch in enumerate(train_dataset):  # each document is a minibatch
                self.model.train()
                optimizer.zero_grad() #change optimizer gradient to zero because the default gradient is not zero
                # convert data items to pytorch inputs
                token_ids = [m['context'][0] + m['context'][1]
                             if len(m['context'][0]) + len(m['context'][1]) > 0
                             else [self.model.word_voca.unk_id]
                             for m in batch]
                s_ltoken_ids = [m['snd_ctx'][0] for m in batch]
                s_rtoken_ids = [m['snd_ctx'][1] for m in batch]
                s_mtoken_ids = [m['snd_ment'] for m in batch]

                entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
                true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())
                p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
                entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())

                token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
                s_ltoken_ids, s_ltoken_mask = utils.make_equal_len(s_ltoken_ids, self.model.snd_word_voca.unk_id,
                                                                   to_right=False)
                s_rtoken_ids, s_rtoken_mask = utils.make_equal_len(s_rtoken_ids, self.model.snd_word_voca.unk_id)
                s_rtoken_ids = [l[::-1] for l in s_rtoken_ids]
                s_rtoken_mask = [l[::-1] for l in s_rtoken_mask]
                s_mtoken_ids, s_mtoken_mask = utils.make_equal_len(s_mtoken_ids, self.model.snd_word_voca.unk_id)

                token_ids = Variable(torch.LongTensor(token_ids).cuda())
                token_mask = Variable(torch.FloatTensor(token_mask).cuda())
                # too ugly but too lazy to fix it
                self.model.s_ltoken_ids = Variable(torch.LongTensor(s_ltoken_ids).cuda())
                self.model.s_ltoken_mask = Variable(torch.FloatTensor(s_ltoken_mask).cuda())
                self.model.s_rtoken_ids = Variable(torch.LongTensor(s_rtoken_ids).cuda())
                self.model.s_rtoken_mask = Variable(torch.FloatTensor(s_rtoken_mask).cuda())
                self.model.s_mtoken_ids = Variable(torch.LongTensor(s_mtoken_ids).cuda())
                self.model.s_mtoken_mask = Variable(torch.FloatTensor(s_mtoken_mask).cuda())

                scores = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m,
                                            gold=true_pos.view(-1, 1))
                loss = self.model.loss(scores, true_pos)

                loss.backward()
                optimizer.step()
                self.model.regularize(max_norm=100)

                loss = loss.cpu().data.numpy()
                total_loss += loss
                print('epoch', e, "%0.2f%%" % (dc/len(train_dataset) * 100), loss, end='\r')

            print('epoch', e, 'total loss', total_loss, total_loss / len(train_dataset))

            if (e + 1) % eval_after_n_epochs == 0:
                dev_f1 = 0
                for di, (dname, data) in enumerate(dev_datasets):
                    predictions = self.predict(data)
                    f1 = D.eval(org_dev_datasets[di][1], predictions)
                    print(dname, utils.tokgreen('micro F1: ' + str(f1)))

                    if dname == 'aida-A' or dname == 'tackbp2015_dev':
                        dev_f1 = f1

                if config['lr'] == 1e-4 and dev_f1 >= self.args.dev_f1_change_lr:
                    eval_after_n_epochs = 2
                    is_counting = True
                    best_f1 = dev_f1
                    not_better_count = 0

                    config['lr'] = 1e-5
                    print('change learning rate to', config['lr'])
                    if self.args.mulrel_type == 'rel-norm':
                        optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])
                    elif self.args.mulrel_type == 'ment-norm':
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = config['lr']

                if is_counting:
                    if dev_f1 < best_f1:
                        not_better_count += 1
                    else:
                        not_better_count = 0
                        best_f1 = dev_f1
                        print('save model to', self.args.model_path)
                        self.model.save(self.args.model_path)

                if not_better_count == self.args.n_not_inc:
                    break

                self.model.print_weight_norm()

    def predict(self, data):
        predictions = {items[0]['doc_name']: [] for items in data}
        self.model.eval()

        for batch in data:  # each document is a minibatch
            token_ids = [m['context'][0] + m['context'][1]
                         if len(m['context'][0]) + len(m['context'][1]) > 0
                         else [self.model.word_voca.unk_id]
                         for m in batch]
            s_ltoken_ids = [m['snd_ctx'][0] for m in batch]
            s_rtoken_ids = [m['snd_ctx'][1] for m in batch]
            s_mtoken_ids = [m['snd_ment'] for m in batch]

            lctx_ids = s_ltoken_ids
            rctx_ids = s_rtoken_ids
            m_ids = s_mtoken_ids

            entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
            p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
            entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())
            true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())

            token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
            s_ltoken_ids, s_ltoken_mask = utils.make_equal_len(s_ltoken_ids, self.model.snd_word_voca.unk_id,
                                                               to_right=False)
            s_rtoken_ids, s_rtoken_mask = utils.make_equal_len(s_rtoken_ids, self.model.snd_word_voca.unk_id)
            s_rtoken_ids = [l[::-1] for l in s_rtoken_ids]
            s_rtoken_mask = [l[::-1] for l in s_rtoken_mask]
            s_mtoken_ids, s_mtoken_mask = utils.make_equal_len(s_mtoken_ids, self.model.snd_word_voca.unk_id)

            token_ids = Variable(torch.LongTensor(token_ids).cuda())
            token_mask = Variable(torch.FloatTensor(token_mask).cuda())
            # too ugly, but too lazy to fix it
            self.model.s_ltoken_ids = Variable(torch.LongTensor(s_ltoken_ids).cuda())
            self.model.s_ltoken_mask = Variable(torch.FloatTensor(s_ltoken_mask).cuda())
            self.model.s_rtoken_ids = Variable(torch.LongTensor(s_rtoken_ids).cuda())
            self.model.s_rtoken_mask = Variable(torch.FloatTensor(s_rtoken_mask).cuda())
            self.model.s_mtoken_ids = Variable(torch.LongTensor(s_mtoken_ids).cuda())
            self.model.s_mtoken_mask = Variable(torch.FloatTensor(s_mtoken_mask).cuda())

            scores = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m,
                                        gold=true_pos.view(-1, 1))
            scores = scores.cpu().data.numpy()

            # print out relation weights
            if self.args.mode == 'eval' and self.args.print_rel:
                print('================================')
                weights = self.model._rel_ctx_ctx_weights.cpu().data.numpy()
                voca = self.model.snd_word_voca
                for i in range(len(batch)):
                    print(' '.join([voca.id2word[id] for id in lctx_ids[i]]),
                          utils.tokgreen(' '.join([voca.id2word[id] for id in m_ids[i]])),
                          ' '.join([voca.id2word[id] for id in rctx_ids[i]]))
                    for j in range(len(batch)):
                        if i == j:
                            continue
                        np.set_printoptions(precision=2)
                        print('\t', weights[:, i, j], '\t',
                              ' '.join([voca.id2word[id] for id in lctx_ids[j]]),
                              utils.tokgreen(' '.join([voca.id2word[id] for id in m_ids[j]])),
                              ' '.join([voca.id2word[id] for id in rctx_ids[j]]))

            pred_ids = np.argmax(scores, axis=1)
            pred_entities = [m['selected_cands']['named_cands'][i] if m['selected_cands']['mask'][i] == 1
                             else (m['selected_cands']['named_cands'][0] if m['selected_cands']['mask'][0] == 1 else 'NIL')
                             for (i, m) in zip(pred_ids, batch)]
            doc_names = [m['doc_name'] for m in batch]

            if self.args.mode == 'eval' and self.args.print_incorrect:
                gold = [item['selected_cands']['named_cands'][item['selected_cands']['true_pos']]
                        if item['selected_cands']['true_pos'] >= 0 else 'UNKNOWN' for item in batch]
                pred = pred_entities
                for i in range(len(gold)):
                    if gold[i] != pred[i]:
                        print('--------------------------------------------')
                        pprint(batch[i]['raw'])
                        print(gold[i], pred[i])

            if self.args.mode == 'eval' and self.args.print_correct:
                gold = [item['selected_cands']['named_cands'][item['selected_cands']['true_pos']]
                        if item['selected_cands']['true_pos'] >= 0 else 'UNKNOWN' for item in batch]
                pred = pred_entities
                for i in range(len(gold)):
                    if gold[i] == pred[i]:
                        print('--------------------------------------------')
                        pprint(batch[i]['raw'])
                        print(gold[i], pred[i])                        

            for dname, entity in zip(doc_names, pred_entities):
                predictions[dname].append({'pred': (entity, 0.)})

        return predictions
