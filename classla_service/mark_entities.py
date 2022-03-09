import json

import classla
from copy import deepcopy

from elasticsearch import Elasticsearch
import editdistance
import compress_fasttext
import requests
from time import sleep
import logging

logging.basicConfig()

classla.download('sl')
nlp1 = classla.Pipeline('sl', processors='tokenize,pos,ner,lemma', use_gpu=True)
model = compress_fasttext.models.CompressedFastTextKeyedVectors.load("CLASSLA/wiki.sl.small")
es = None



def entitySearch(query):
    global es
    if es is None:
    	try:
    		es = Elasticsearch(['http://elasticsearch:9200'])
    	except:
    		return []
    		
    indexName = "wikidataentityindex"
    results=[]
    ###################################################

    ###################################################
    try:
	    	
	    elasticResults=es.search(index=indexName, doc_type="_doc", body={
		    "query": {
			"match" : {
			    "label" : {
				"query": query,
				"fuzziness": "AUTO"
				
			    }
			}
		    },"size":100
			    }
			   )
    except Exception as e:
    	es = None
    	logging.info("failed to search es:")
    	logging.info(e)
    	return []
 
    for result in elasticResults['hits']['hits']:
        edit_distance=editdistance.eval(result["_source"]["label"].lower().replace('.','').strip(), query.lower().strip())
        if edit_distance<=1:
            results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"]*50,30])
        else:
            results.append([result["_source"]["label"],result["_source"]["uri"],result["_score"]*25,0])
            
    results= sorted(results, key = lambda x: (int(x[1][x[1].rfind("/")+2:-1]),-x[3],-x[2]))
    seen = set()
    results = [x for x in results if x[1] not in seen and not seen.add(x[1])]
    results=results[:20]
    results= sorted(results, key = lambda x: (-x[3],int(x[1][x[1].rfind("/")+2:-1])))
        
    return results[:15]


def get_wikidata_tag(entity):
    #global timer

    #t = time()
    #if t - timer < 0.4:
    #    sleep(0.4 - t + timer)

    #timer = t

    try:
            x = requests.get(
                'https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&props=sitelinks&sitefilter=slwiki&sites=slwiki&titles=' + entity)
            dom = parseString(x.text)
            tmp = dom.getElementsByTagName('entity')

            if len(tmp) == 1:
                val = tmp[0].attributes['id'].value
                if val != "-1":
                    return val
    except:
            pass
        
    try:
            x = requests.get(
                'https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&props=sitelinks&sitefilter=emwiki&sites=enwiki&titles=' + entity)
            dom = parseString(x.text)
            tmp = dom.getElementsByTagName('entity')

            if len(tmp) == 1:
                val = tmp[0].attributes['id'].value
                if val != "-1":
                    return val
    except:
            pass

    return None


def search_NE_elasticsearch(entity_words, entity_WV):
    global model
    annotations = entitySearch(entity_words)
    out = "None"
    bst_scr = 0.88
    bst_elastic_scr = 0
    for annotation in annotations:
        if annotation[2] > 250.0:
            candidate_WV = sum([model[word] for word in annotation[0].split(" ")])
            cosine_similarity = 1 - spatial.distance.cosine(entity_WV, candidate_WV)
            if cosine_similarity > bst_scr:
                out = annotations[0][1][32:-1]
                bst_scr = cosine_similarity
                bst_elastic_scr = annotation[2]
            elif cosine_similarity == bst_scr and annotation[2] > bst_elastic_scr:
                out = annotations[0][1][32:-1]
                bst_scr = cosine_similarity
                bst_elastic_scr = annotation[2]
    return out, bst_scr


def find_wiki_tag(tokens, lemmas):
    wiki_tag = get_wikidata_tag("_".join(lemmas))
    if wiki_tag is None:
        wiki_tag = get_wikidata_tag("_".join(tokens))
    if wiki_tag is None:
        wiki_tag = search_NE_elasticsearch(" ".join(lemmas), sum([model[word] for word in lemmas]))[0]
    return wiki_tag


def mark_entities_in_text(text, call_id):
    marked_doc = nlp1(text).to_dict()
    logging.info(f"call: {call_id} classla output :\n{json.dumps(marked_doc, indent = 4)}")
    out = []
    for sentence in marked_doc:
        vertex_set = []
        edge_set = []
        tokens = []
        lemmas = []
        token_pos = []
        cur_pos = 0
        sentence_pieces = []
        for index, word in enumerate(sentence[0]):
            tokens.append(word["text"])
            sentence_pieces.append(word["text"])
            token_pos.append(cur_pos)
            if not "misc" in word or "SpaceAfter=No" not in word["misc"]:
                sentence_pieces[-1] += " "
            cur_pos += len(sentence_pieces[-1])
            lemmas.append(word["lemma"])
            if word["ner"][0] == "B":
                vertex_set.append({
                    "kbID": "None",
                    "tokenpositions": [
                        index
                    ],
                })
            elif word["ner"][0] == "I":
                try:
                    vertex_set[-1]["tokenpositions"].append(index)
                except:
                    pass
        for i in range(len(vertex_set)):
            vertex_set[i]["kbID"] = find_wiki_tag(tokens[vertex_set[i]["tokenpositions"][0]:vertex_set[i]["tokenpositions"][-1] + 1],
                                                  lemmas[vertex_set[i]["tokenpositions"][0]:vertex_set[i]["tokenpositions"][-1] + 1])
        sentence_text = "".join(sentence_pieces).strip()
        for i in range(len(vertex_set)):
            for j in range(i+1, len(vertex_set)):
                edge_set.append({
                    "kbID": "P20",
                    "left": vertex_set[i]["tokenpositions"],
                    "right": vertex_set[j]["tokenpositions"],
                    "sentence": sentence_text,
                    "entity1_text": "".join(sentence_pieces[vertex_set[i]["tokenpositions"][0]:vertex_set[i]["tokenpositions"][-1] + 1]).strip(),
                    "entity2_text": "".join(sentence_pieces[vertex_set[j]["tokenpositions"][0]:vertex_set[j]["tokenpositions"][-1] + 1]).strip(),
                    "entity1_sentence_position": token_pos[vertex_set[i]["tokenpositions"][0]],
                    "entity2_sentence_position": token_pos[vertex_set[j]["tokenpositions"][0]]
                })
                edge_set.append({
                    "kbID": "P20",
                    "right": vertex_set[i]["tokenpositions"],
                    "left": vertex_set[j]["tokenpositions"],
                    "sentence": sentence_text,
                    "entity1_text": edge_set[-1]["entity2_text"],
                    "entity2_text": edge_set[-1]["entity1_text"],
                    "entity1_sentence_position": edge_set[-1]["entity2_sentence_position"],
                    "entity2_sentence_position": edge_set[-1]["entity1_sentence_position"]
                })
        out.append({"vertexSet": deepcopy(vertex_set), "edgeSet": deepcopy(edge_set), "tokens": deepcopy(tokens)})
    
    logging.info(f"call {call_id} input for RECON prediction :\n{json.dumps(out, indent = 4)}")
    return out
