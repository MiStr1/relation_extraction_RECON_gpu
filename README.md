# relation_extraction_RECON_gpu
fastapi relation extraction service in docker which uses a model built with method [RECON](https://github.com/ansonb/RECON) and GPU acceleration

---

This repository contains a model for relation extraction in the Slovenian language. 

## Project structure

- `classla_service/` contains code for finding entities in text and marking them with a WikiData tag.
- `relation_extraction/` contains scripts to extract relation from a text with marked entities.
- `RECON_data.zip` contains all the data used for relation extraction with RECON.
- `wikidataentity.zip` contains a dump of WikiData entities and is used for marking the entitites in text.


## Run with docker

First, we need to extract the folder contained in RECON_data.zip into the root of this project.

To run GPU accelerated docker containers you need to have an Nvidia GPU and [CUDA for WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) on Windows 10 or 11
or [The NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for Linux. You can also use the 
[CPU alternative for this service](https://github.com/MiStr1/relation_extraction_BERT_cpu).

This project can be run with docker-compose with the command `docker-compose up` in the root of the project.

When the elasticsearch service starts you need to fill it with a WikiData entity dump. To do this you first need to extract the `wikidataentity.zip`.
Then run 

`elasticdump  --output=http://localhost:9200/wikidataentityindex/  --input=wikidataentity.json  --type=data`

in the same folder as `wikidataentity.json`. To do this you need [elasticdump](https://www.npmjs.com/package/elasticdump) (and [npm](https://www.npmjs.com)).

**Note** This service consumes a lot of RAM (<20GB). You can omit the elasticsearch service and run `docker-compose up entity_relation_extractor` instead. This will lower RAM
consumption to 13GB at startup and 11GB after startup. You will however get less accurate relations because the entities will not be marked with a WikiData tag.

 
 ## Use
 
 Rest API is provided by FastAPI/uvicorn.
 
 After starting up the API, the OpenAPI/Swagger documentation will become accessible at http://localhost:8000/docs and http://localhost:8000/openapi.json.
 
 For extracting the relations in a sentence you can send get request to http://localhost:8000/find_relations/{text} where {text} represents the sentence.
 You can also send a get request to the http://localhost:8000/find_relations and add parameter `text` which contains the sentence. The return form for those
 two get requests can be found on http://localhost:8000/docs when the service is running.
 
