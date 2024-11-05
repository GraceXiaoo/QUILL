#pip install chromadb
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from chromadb import Documents, EmbeddingFunction, Embeddings


class MyVectorDBConnector:
    def __init__(self, path,collection_name):
        chroma_client = chromadb.PersistentClient(path=path)
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = get_embeddings

    def add_documents(self, documents,metadata):
        '''Adding documents and vectors to collection'''
        self.collection.add(
            documents=documents,
            embeddings=get_embeddings(documents),
            metadatas=metadata,
            ids=[f"id{i}" for i in range(len(documents))]
        )

    def search_author(self, query, top_n, author):
        '''Search Vector Database by author'''
        results = self.collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=top_n,
            where={"author":author}
        )
        return results

    def search_topic(self, query, top_n, topic):
        '''Search Vector Database bu topic'''
        results = self.collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=top_n,
            where={"topic":topic}
        )
        return results
    def search(self, query, top_n):
        '''Search Vector Database'''
        results = self.collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=top_n
        )
        return results
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        self.model=SentenceTransformer('/data1/bowei/QUILL/code/model_llm/acge_text_embedding')
        self.embeddings = self.model.encode(Documents, normalize_embeddings=False)
        self.matryoshka_dim = 1024
        self.embeddings = self.embeddings[..., :self.matryoshka_dim] 
        self.embeddings = normalize(self.embeddings, norm="l2", axis=1)
        reshaped_array = self.embeddings.reshape(-1, self.matryoshka_dim)
        list_of_lists = reshaped_array.tolist()
        return list_of_lists

def get_embeddings(Documents):
    model=SentenceTransformer('/data1/bowei/QUILL/code/model_llm/acge_text_embedding')
    embeddings = model.encode(Documents, normalize_embeddings=False)
    matryoshka_dim = 1024
    embeddings = embeddings[..., :matryoshka_dim] 
    embeddings = normalize(embeddings, norm="l2", axis=1)
    reshaped_array = embeddings.reshape(-1, matryoshka_dim)
    list_of_lists = reshaped_array.tolist()
    return list_of_lists








