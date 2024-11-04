# Import necessary modules
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
import numpy as np
from uuid import uuid4
from huggingface_hub import login
import pandas as pd
from typing import List, Dict, Any
import pickle


class CreateVectorStore():
    def __init__(self, hf_token):
        self.hf_token = hf_token

    def prepare_documents(self) -> List[Document]:
        """
        Prepare the documents for the vector store.

        This function reads the parquet file, renames the columns, and
        creates a list of Document objects with metadata. The metadata
        includes the source id, question, and answer.

        Returns
        -------
        List[Document]
            A list of Document objects with the metadata.
        """
        login(token=self.hf_token)
        # Read the parquet file
        df = pd.read_parquet("hf://datasets/PatronusAI/HaluBench/data/test-00000-of-00001.parquet")
        
        # Select the first 25 rows and the first 5 columns
        train_df = df.iloc[0:100, 0:5]
        
        # Rename the columns
        train_df.rename(columns={'passage': 'context'}, inplace=True)
        
        # Convert DataFrame to documents with metadata with source id
        documents = [
            Document(
                page_content=row['context'],
                metadata={'id': row['id'], 'question': row['question'], 'answer': row['answer']}
            ) for _, row in train_df.iterrows()
        ]
        
        return documents, train_df
    
    def create_chunks(self, documents):
        """
        Split documents into smaller chunks.

        This method uses a RecursiveCharacterTextSplitter to split the input
        documents into chunks of specified size with a given overlap.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects to be split into chunks.

        Returns
        -------
        List[Document]
            A list of Document objects representing the chunks.
        """
        # Initialize the text splitter with chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Split the documents into chunks using the text splitter
        chunks = text_splitter.split_documents(documents)

        return chunks
    
    def create_and_save_bm25_index(self, chunks):
        """
        Create a BM25 index from the document chunks.

        Parameters
        ----------
        chunks : List[Document]
            A list of Document objects representing the document chunks.

        Returns
        -------
        None
        """
        # Tokenize the document chunks
        tokenized_corpus = [chunk.page_content.split() for chunk in chunks]
        
        # Create the BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save the BM25 index to a file
        with open('bm25_index.pkl', 'wb') as f:
            pickle.dump(bm25, f)
        

    def create_and_save_semantic_and_hybrid_embeddings(self, chunks):
        """
        Create semantic embeddings from the document chunks using the given model.

        Parameters
        ----------
        chunks : List[Document]
            A list of Document objects representing the document chunks.

        Returns
        -------
        None
        """
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {'device': 'cpu'}
            )
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # Save the FAISS vector store to a file
        vectorstore.save_local('faiss_vectorstore')


class Retriever():
    def __init__(self, query):
        self.query = query

    def bm25_retriever(self, chunks, path_to_pkl_file, k=3):
        """
        Retrieve the top k chunks using the BM25 index.

        Parameters
        ----------
        chunks : List[Document]
            A list of Document objects representing the document chunks.
        k : int, optional
            The number of chunks to retrieve. Defaults to 3.

        Returns
        -------
        retrieved_chunks : List[Document]
            A list of Document objects representing the top k chunks.
        """
        # Load the BM25 index from the file
        with open(path_to_pkl_file, 'rb') as f:
            bm25 = pickle.load(f)
        # Get BM25 scores
        tokenized_query = self.query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top k chunks based on BM25 scores
        top_k_indices = bm25_scores.argsort()[-k:][::-1]
        retrieved_chunks = [chunks[i] for i in top_k_indices]
        
        return retrieved_chunks
    
    def semantic_retriever(self, path_to_vectorstore, k=3):
        """
        Retrieve the top k most semantically similar chunks to the query.

        Parameters
        ----------
        query : str
            The query string to search for.
        path_to_vectorstore : str
            The file path to the FAISS vector store.
        k : int, optional
            The number of chunks to retrieve. Defaults to 3.

        Returns
        -------
        List[Tuple[Document, float]]
            A list of tuples containing the retrieved Document objects and
            their corresponding cosine similarity scores.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {'device': 'cpu'}
            )

        # Load the FAISS vector store from the file
        vectorstore = FAISS.load_local(path_to_vectorstore, embeddings=embeddings,allow_dangerous_deserialization=True)

        # Retrieve similar documents using cosine similarity
        relevant_docs = vectorstore.similarity_search(self.query, k=k)
        
        return relevant_docs


    def reciprocal_rank_fusion(self, results_list, k=60):
        """
        Reciprocal rank fusion with source id with universal unique identifier.

        This function takes in a list of lists of Document objects and
        returns a list of Document objects with fused scores.

        Parameters
        ----------
        results_list : List[List[Document]]
            A list of lists of Document objects.
        k : int, optional
            The number of documents to retrieve. Defaults to 60.

        Returns
        -------
        List[Document]
            A list of Document objects with fused scores.
        """
        # Initialize a dictionary to store the fused scores
        fused_scores = {}
        # Iterate over the list of results
        for rank, results in enumerate(results_list):
            # Iterate over the documents in the results
            for doc in results:
                # Get the source id of the document
                doc_id = doc.metadata['id']
                # If the document id is not in the fused scores dictionary,
                # add it with a score of 0
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                # Add the reciprocal rank to the fused scores
                fused_scores[doc_id] += 1 / (rank + k)
        
        # Sort the documents by their fused scores in descending order
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        # Return the top k documents
        return [doc for doc in results_list[0] if doc.metadata['id'] in [doc_id for doc_id, _ in sorted_docs]]
    
    def hybrid_retriever(self, chunks, path_to_vectorstore, path_to_pkl_file, k=3):
        """
        Retrieve the top k chunks using both BM25 and semantic search.

        This method first retrieves the top k chunks using BM25 search,
        then uses semantic search to retrieve the top k chunks. The
        two lists of chunks are then fused using reciprocal rank fusion.

        Parameters
        ----------
        chunks : List[Document]
            A list of Document objects representing the document chunks.
        k : int, optional
            The number of chunks to retrieve. Defaults to 3.

        Returns
        -------
        List[Document]
            A list of Document objects representing the top k chunks.
        """
        # Retrieve the top k chunks using BM25 search
        bm25_results = self.bm25_retriever(chunks=chunks, path_to_pkl_file= path_to_pkl_file)
        # Retrieve the top k chunks using semantic search
        semantic_results = self.semantic_retriever(path_to_vectorstore=path_to_vectorstore)
        # Fuse the two lists of chunks using reciprocal rank fusion
        fused_results = self.reciprocal_rank_fusion([bm25_results, semantic_results])
        # Return the top k chunks
        return fused_results[:k]


class Generation():
    def __init__(self, retrieved_chunks, query):
        self.retrieved_chunks = retrieved_chunks
        self.query = query

    def get_llama_pipeline(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> HuggingFacePipeline:
        """
        Returns a pipeline for text generation using the Meta LLaMA model.

        The pipeline is initialized with the following parameters:

        - `max_length`: The maximum length of the generated text.
        - `max_new_tokens`: The maximum number of new tokens to generate.
        - `temperature`: The temperature parameter for the softmax function.
        - `pad_token_id`: The token ID for padding the generated text.

        The pipeline is then wrapped in a LangChain HuggingFacePipeline object.

        Parameters
        ----------
        model_name : str
            The name of the Meta LLaMA model to use.

        Returns
        -------
        HuggingFacePipeline
            A LangChain HuggingFacePipeline object for text generation.
        """
        # Load the tokenizer and model from the Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        
        # Initialize the text generation pipeline with the specified parameters
        pipe = pipeline("text-generation", 
            model=model,
            tokenizer=tokenizer,
            max_length=1024,  # Increased from 512
            max_new_tokens=512,  # Explicitly set number of new tokens to generate
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
            )
        
        # Wrap the pipeline in a LangChain HuggingFacePipeline object
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

    def generate_response(self) -> Dict[str, Any]:
        """
        Generates a response to the user's query based on the retrieved chunks.

        This method first creates a prompt template from the retrieved chunks,
        then uses the LLaMA model to generate a response based on the prompt.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the response and the retrieved chunks.
        """

        # Create a prompt template from the retrieved chunks
        context = "\n".join([chunk.page_content for chunk in self.retrieved_chunks])
        prompt_template = f"""
        You are a helpful AI assistant. Use the following pieces of context to answer the question.
        If you don't know the answer, just say that you don't know. Do not try to make up an answer.

        Context: {context}

        Question: {self.query}

        Provide a clear and concise answer based only on the given context.
        Do not include any information that is not supported by the context.

        Answer:
        """
        
        # Create a prompt object from the template
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Get the LLaMA pipeline
        llm = self.get_llama_pipeline()

        # Create a LangChain LLMChain object
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain to generate a response
        response = chain.run(context=context, question=self.query)

        # Return the response and the retrieved chunks
        return {"response": response, "retrieved_chunks": self.retrieved_chunks}

class Evaluation():
    def __init__(self, df_test, retriever:Retriever, generation:Generation, retriever_type= "bm25"):
        self.df_test = df_test
        self.retriever = retriever
        self.generation = generation
        self.retriever_type = retriever_type
        
        

    def evaluate_model(self):
        """
        Evaluate the model on the test data.

        Iterate over the test data and generate answers for each question.
        Calculate the accuracy of the model by comparing the generated answers
        with the true answers.

        Returns:
        A DataFrame with the results of the evaluation.
        """
        results = []
        for _, row in self.df_test.iterrows():
            query = row['question']
            context = row['context']
            true_answer = row['answer']
                   

            if self.retriever_type == "text_matching":
                retrieved_chunks = self.retriever.bm25_retriever()

            if self.retriever_type == "semantic_matching":
                retrieved_chunks = self.retriever.semantic_retriever()

            if self.retriever_type == "hybrid_matching":
                retrieved_chunks = self.retriever.hybrid_retriever()
            
            #model_pipeline = generator.get_llama_pipeline()
            generator = Generation(retrieved_chunks=retrieved_chunks, query=query)
            result = generator.generate_response()
            generated_answer = result["response"]
            retrieved_chunks = result["retrieved_chunks"]
            
            results.append({
                'query': query,
                'context': context,
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'retrieved_chunks': retrieved_chunks
            })
        results_df = pd.DataFrame(results)
        # Assuming your DataFrame is called 'df'
        results_df['extracted_answer'] = results_df['generated_answer'].str.split('Answer:').str[-1].str.strip()
        
        # If you want to remove the quotation marks at the beginning and end
        results_df['extracted_answer'] = results_df['extracted_answer'].str.strip("'")

        return results_df
     
if __name__ == "__main__":
    vector_store= CreateVectorStore(hf_token="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")
    documents, df_test = vector_store.prepare_documents()
    chunks = vector_store.create_chunks(documents)
    # vector_store.create_and_save_bm25_index(chunks=chunks)
    # vector_store.create_and_save_semantic_and_hybrid_embeddings(chunks=chunks)

    results = []
    for _, row in df_test.iterrows():
        query = row['question']
        context = row['context']
        true_answer = row['answer']
        # Retriever
        retriever_type = "semantic_matching"
        if retriever_type == "text_matching":
            retriever = Retriever(query=query)
            retrieved_chunks = retriever.bm25_retriever(chunks=chunks, path_to_pkl_file='bm25_index.pkl') 

        if retriever_type == "semantic_matching":
            retriever = Retriever(query=query)
            retrieved_chunks = retriever.semantic_retriever(path_to_vectorstore='faiss_vectorstore')

        if retriever_type == "hybrid_matching":
            retriever = Retriever(query=query)
            retrieved_chunks = retriever.hybrid_retriever(chunks=chunks, path_to_vectorstore='faiss_vectorstore', path_to_pkl_file='bm25_index.pkl' )
        
        generator = Generation(retrieved_chunks=retrieved_chunks, query=query)
        result = generator.generate_response()
        generated_answer = result["response"]
        retrieved_chunks = result["retrieved_chunks"]
        
        results.append({
            'query': query,
            'context': context,
            'true_answer': true_answer,
            'generated_answer': generated_answer,
            'retrieved_chunks': retrieved_chunks
        })
    results_df = pd.DataFrame(results)
    # Assuming your DataFrame is called 'df'
    results_df['extracted_answer'] = results_df['generated_answer'].str.split('Answer:').str[-1].str.strip()
    
    # If you want to remove the quotation marks at the beginning and end
    results_df['extracted_answer'] = results_df['extracted_answer'].str.strip("'")
    results_df.to_csv('SM_100_rows.csv', index=False)

    
    x=2

# load_df = pd.read_csv('SM_data.csv')
# load_df
