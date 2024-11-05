from rag_module import MyVectorDBConnector,get_embeddings
import os,json
vector=MyVectorDBConnector(path='/QUILL/code/rag/model/quill_final',collection_name='quill_final')

documents=[]
metadata=[]

directory='/QUILL/data/rag'
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path=os.path.join(directory,filename)
        try:
            with open(file_path,'r',encoding='utf-8') as file:
                data=json.load(file)
                print(len(data))
                for dic in data:
                    if dic['quote'] not in documents:
                        documents.append(dic['quote'])
                        try:
                            metadata.append({'topic':dic['topic'],'author':dic['author'],'poem':dic['poem']})
                        except:
                            metadata.append({'topic':dic['topic'],'author':dic['author']})
        except Exception as e:
            print('error:',str(e))

print(len(documents))
vector.collection.add(
    documents=documents,
    metadatas=metadata,
    embeddings=get_embeddings(documents),
    ids=[f"id{i}" for i in range(len(documents))] 
)
print('done')