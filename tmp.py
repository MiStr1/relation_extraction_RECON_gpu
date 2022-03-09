from elasticsearch import Elasticsearch
import editdistance 

es = None



def entitySearch(query):
    global es
    if es is None:
        try:
            es = Elasticsearch(['http://localhost:9200'])
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
    	print("failed to search es:", e)
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

 
 
print(entitySearch("pica"))
