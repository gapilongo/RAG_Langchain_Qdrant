from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import logging
import os
import torch
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f'you are using {device}')


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

url = "http://localhost:6333"
collection_name = "test_collection"

client = QdrantClient(
    url = url,
    prefer_grpc= False
)

logging.info(f"load {client}")

db = Qdrant(
    client = client,
    embeddings=embeddings,
    collection_name=collection_name
)

logging.info(f"load {db}")

query = "can ai take over human and how?"

docs = db.similarity_search_with_score(query=query, k=5)

# for i in docs:
#     doc,score = i
#     print({"score": score, "content": doc.page_content, "metadata": doc.metadata})

custom_prompt = """
Use the following pieces of context to answer the question at the end. Please provide
a short single-sentence summary answer only. If you don't know the answer or if it's 
not present in given context, don't try to make up an answer, but Say "I don't know the answer". 
Context: {context}
Question: {question}
Helpful Answer:
"""

custom_prompt_template = PromptTemplate(
    template=custom_prompt, input_variables=["context", "question"]
)
llm = OpenAI()

custom_qa = RetrievalQA.from_chain_type(
    llm, 
    chain_type="stuff", 
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt_template},
)

response = custom_qa.invoke(input={"context":docs,"query":query})
print(response)
