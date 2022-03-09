from RECON.utils import embedding_utils, context_utils
from RECON.semanticgraph import io
from RECON.parsing import legacy_sp_models as sp_models
from RECON.models import baselines
import numpy as np
import json
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import *
import ast
from RECON.models.factory import get_model
import argparse
from copy import deepcopy
from requests import get
import os
import logging

logging.basicConfig()


import torch.nn.functional as F

try:
    from functools import reduce
except:
    pass

import os
def to_np(x):
    return x.data.cpu().numpy()


# load model and data
logging.info("loading")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152đ

logging.info(torch.cuda.device_count())
logging.info(torch.cuda.get_device_name(0))

np.random.seed(1)

p0_index = 1

CUDA = torch.cuda.is_available()

data_folder = "RECON_data/"
relation_descriptions = dict()
with open(data_folder + "properties-with-labels.txt", encoding="utf-8") as f:
    for line in f:
        relation_descriptions[line.strip().split("\t")[0]] = line.strip().split("\t")[1]


model_params = data_folder + "model_params.json"
word_embeddings = data_folder + "wiki.sl.vec"

context_data_file = data_folder + "knowledge_graph.json"
gat_embedding_file = data_folder + 'final_entity_embeddings.json'
gat_entity2id_file = data_folder + 'entity2id.txt'
gat_relation_embedding_file = data_folder + 'final_relation_embeddings.json'
gat_relation2id_file = data_folder + 'relation2id.txt'
w_ent2rel_all_rels_file = data_folder + '/W_ent2rel.json.npy'

with open(model_params) as f:
    model_params = json.load(f)
model_params['batch_size'] = 1

char_vocab_file = data_folder + "char_vocab.json"

sp_models.set_max_edges(model_params['max_num_nodes'] * (model_params['max_num_nodes'] - 1),
                        model_params['max_num_nodes'])

with open(context_data_file, 'r') as f:
    context_data = json.load(f)
with open(gat_embedding_file, 'r') as f:
    gat_embeddings = json.load(f)
with open(gat_relation_embedding_file, 'r') as f:
    gat_relation_embeddings = json.load(f)

W_ent2rel_all_rels = np.load(w_ent2rel_all_rels_file)
with open(gat_entity2id_file, 'r') as f:
    gat_entity2idx = {}
    data = f.read()
    lines = data.split('\n')
    for line in lines:
        line_arr = line.split(' ')
        if len(line_arr) == 2:
            gat_entity2idx[line_arr[0].strip()] = line_arr[1].strip()

with open(gat_relation2id_file, 'r') as f:
    gat_relation2idx = {}
    data = f.read()
    lines = data.split('\n')
    for line in lines:
        line_arr = line.split(' ')
        if len(line_arr) == 2:
            gat_relation2idx[line_arr[0].strip()] = line_arr[1].strip()

all_zeros = deepcopy(gat_embeddings["0"])
for i in range(len(all_zeros)):
    all_zeros[i] = 0
unknown = deepcopy(all_zeros)
for i in gat_embeddings:
    unknown = [sum(x) for x in zip(unknown, gat_embeddings[i])]
l = float(len(gat_embeddings))
unknown = [float(x) / l for x in unknown]
gat_embeddings[str(len(gat_embeddings))] = all_zeros
gat_embeddings[str(len(gat_embeddings))] = unknown

embeddings, word2idx = embedding_utils.load(word_embeddings)
logging.info("Loaded embeddings:", embeddings.shape)

logging.info("Reading the property index")
with open(data_folder + "RECON.property2idx") as f:
    property2idx = ast.literal_eval(f.read())
idx2property = {v: k for k, v in property2idx.items()}
logging.info("Reading the entity index")
with open(data_folder + "RECON.entity2idx") as f:
    entity2idx = ast.literal_eval(f.read())
idx2entity = {v: k for k, v in entity2idx.items()}
context_data['ALL_ZERO'] = {
        'desc': '',
        'label': 'ALL_ZERO',
        'instances': [],
        'aliases': []
    }

with open(char_vocab_file, 'r') as f:
    char_vocab = json.load(f)

max_sent_len = 36
logging.info("Max sentence length set to: {}".format(max_sent_len))

graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding

_, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)

training_data = None

n_out = len(property2idx)
logging.info("N_out:", n_out)

model = get_model("RECON")(model_params, embeddings, max_sent_len, n_out, char_vocab,
                               gat_relation_embeddings, W_ent2rel_all_rels, idx2property, gat_relation2idx)

#torch.cuda.set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model = model.cuda()
model.load_state_dict(torch.load(data_folder + "RECON-8.out"))


def predict(sentence, call_id):
    global model_params
    test_set = []
    test_set = get('http://0.0.0.0:8001/mark_entities/' + sentence, params={"call_id": call_id}).json()
    test_set, _ = io.load_relation_graphs_from_jsons(test_set, load_vertices=True, data='wikidata')
    test_as_indices = list(graphs_to_indices(test_set, word2idx, property2idx, max_sent_len, embeddings=embeddings,
                                             position2idx=position2idx, entity2idx=entity2idx))

    indices = np.arange(test_as_indices[0].shape[0])

    #result_file = open(os.path.join(result_folder, "_" + model_name + "_train_wiki"), "w", encoding="utf-8")
    test_f1 = 0.0
    results = []
    for i in tqdm(range(int(test_as_indices[0].shape[0] / model_params['batch_size']))):
        sentence_input = test_as_indices[0][
            indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
        entity_markers = test_as_indices[1][
            indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
        labels = test_as_indices[2][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
        entity_indices = test_as_indices[4][
                indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
        unique_entities, unique_entities_surface_forms, max_occurred_entity_in_batch_pos = context_utils.get_batch_unique_entities(
                test_as_indices[4][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
                test_as_indices[5][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]])
        unique_entities_context_indices = context_utils.get_context_indices(unique_entities,
                                                                                unique_entities_surface_forms,
                                                                                context_data, idx2entity, word2idx,
                                                                                char_vocab,
                                                                                model_params['conv_filter_size'],
                                                                                max_sent_len=32,
                                                                                max_num_contexts=32,
                                                                                max_char_len=10, data='wikidata')
        entities_position = context_utils.get_entity_location_unique_entities(unique_entities, entity_indices)
        gat_entity_embeddings, nonzero_gat_entity_embeddings, nonzero_entity_pos = context_utils.get_selected_gat_entity_embeddings(
                entity_indices, entity2idx, idx2entity, gat_entity2idx, gat_embeddings)
       
        output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(),
                               Variable(torch.from_numpy(entity_markers.astype(int))).cuda(),
                               test_as_indices[3][
                                   indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
                               Variable(torch.from_numpy(unique_entities.astype(np.long))).cuda(),
                               Variable(torch.from_numpy(entity_indices.astype(np.long))).cuda(),
                               Variable(torch.from_numpy(unique_entities_context_indices[0].astype(np.long))).cuda(),
                               Variable(torch.from_numpy(unique_entities_context_indices[1].astype(np.long))).cuda(),
                               Variable(torch.from_numpy(unique_entities_context_indices[2].astype(bool))).cuda(),
                               Variable(torch.from_numpy(entities_position.astype(int))).cuda(),
                               max_occurred_entity_in_batch_pos,
                               Variable(torch.from_numpy(nonzero_gat_entity_embeddings.astype(np.float32)),
                                        requires_grad=False).cuda(),
                               nonzero_entity_pos,
                               Variable(torch.from_numpy(gat_entity_embeddings.astype(np.float32)),
                                        requires_grad=False).cuda())



        labels_copy = labels.reshape(-1).tolist()
        p_indices = np.array(labels_copy) != 0

        score = F.softmax(output, dim=-1)
        score = to_np(score).reshape(-1, n_out)
        labels = labels.reshape(-1)
        p_indices = labels != 0
        score = score[p_indices].tolist()
        labels = labels[p_indices].tolist()
        pred_labels = r = np.argmax(score, axis=-1)
        indices2 = [i for i in range(len(p_indices)) if p_indices[i]]
        entity_pairs = test_as_indices[6][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        entity_pairs = reduce(lambda x, y: x + y, entity_pairs)
        
        input_edges = sum([sentence["edgeSet"] for sentence in test_set], [])
        start_idx = i * model_params['batch_size']
        for index, (i, j, entity_pair, edge) in enumerate(zip(score, labels, entity_pairs, input_edges)):
            sent = ' '.join(test_set[start_idx + indices2[index] // (
                        model_params['max_num_nodes'] * (model_params['max_num_nodes'] - 1))]['tokens']).strip()
            results.append({
                "sentence": edge["sentence"],
                "entity1": {
                	"text": edge["entity1_text"],
                	"sentence_position": edge["entity1_sentence_position"]
                },
                "entity2": {
                	"text": edge["entity2_text"],
                	"sentence_position": edge["entity2_sentence_position"]
                },
                "relation": {
                            "WikiData_tag": idx2property[pred_labels[index]], 
                            "description": relation_descriptions[idx2property[pred_labels[index]]]
                       },
                "score": score[index][pred_labels[index]]

            })
    return results

