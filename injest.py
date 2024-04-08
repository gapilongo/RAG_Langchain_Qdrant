from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import logging
import os
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f'you are using {device}')
path = "./data"
isExist = os.path.exists(path)
if not isExist:
   logging.error(f"Path {path} does not exist")
   exit(1)

logging.info('Getting Documents ...')
filename_fn = lambda filename: {'file_name': filename}
documents = DirectoryLoader(path,glob="./*.pdf", show_progress=True,loader_cls=PyMuPDFLoader).load()

# with open('output.txt', 'w') as f:
#     # Iterate over the elements of the list and write each element to the file
#     for item in documents:
#         f.write("%s\n" % item)


text_splitter = RecursiveCharacterTextSplitter(
   chunk_size= 500,
   chunk_overlap= 30
)

texts = text_splitter.split_documents(documents)

#load the embedding modal

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}

logging.info(f"Loding the embedding model : {model_name}")
embeddings = HuggingFaceBgeEmbeddings(
   model_name = model_name,
   model_kwargs = model_kwargs,
   encode_kwargs = encode_kwargs
)

logging.info(f"Model loaded ...")

url = "http://localhost:6333"
collection_name = "test_collection"
logging.info(f"Creating and saving the index {collection_name}")
qdrant = Qdrant.from_documents(
   texts,
   embeddings,
   url=url,
   prefer_grpc = False,
   collection_name=collection_name
)
logging.info("Done !")
