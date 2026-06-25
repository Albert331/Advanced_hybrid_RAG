from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langsmith import traceable
import os

load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'

embedding_model = OllamaEmbeddings(model='nomic-embed-text')

class RAG():
    
    class EnsembleRetriever:
        def __init__(self, retrievers, weights):
            self.retrievers = retrievers
            self.weights = weights
        
        def invoke(self, query: str):
            all_docs = []
            seen = set()
            
            for retriever in self.retrievers:
                docs = retriever.invoke(query)
                for doc in docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        all_docs.append(doc)
            
            return all_docs

    def __init__(self):
        self.all_docs = []
        vector_store = self.Kb('Knowledge_base')
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        bm_retriever = self.bm25('knowledge_base')
        
        self.ensemble_retriever = self.EnsembleRetriever(
            retrievers=[bm_retriever, retriever],
            weights=[0.5, 0.5]
        )

    def Kb(self, fileadress: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)

        
        if os.path.exists('Knowledge_base'):
            vector_store = Chroma(
                collection_name='efficiency_trials',
                embedding_function=embedding_model,
                persist_directory='./Knowledge_base'
            )
            
            for filename in os.listdir(fileadress):
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(os.path.join(fileadress, filename))
                    self.all_docs.extend(loader.load())
            print('loaded existing knowledge base')
            return vector_store

        
        for filename in os.listdir(fileadress):
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(fileadress, filename))
                self.all_docs.extend(loader.load())

        chunks = splitter.split_documents(self.all_docs)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name='efficiency_trials',
            persist_directory='./chroma_db'
        )
        print(f'imported {len(chunks)} chunks from {fileadress}')
        return vector_store


    def bm25(self,filepath:str):
        bm_retriever = BM25Retriever.from_documents(
            self.all_docs,
            k=3
            
            )
        return bm_retriever
    
    def invoke(self,query:str):
        return self.ensemble_retriever.invoke(query)

    



@traceable
def llm_call(query:str):
    
    rag = RAG()
    llm = ChatOllama(
        model='llama3.2:latest',
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_template(
        '''
    Answer the question only based on the following context.
    If the answer is not in the context, say "I don't know".
    Do not use any prior knowledge.

    Context:
    {context}

    Question: {question}
    Answer:

    
    '''
    )
    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    rag_chain =(
        {'context':RunnableLambda(rag.invoke) | format_docs,'question':RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

    print('RAG ANSWERS\n')
    print(answer)
    
    
llm_call("what optimizer did the authors use?")