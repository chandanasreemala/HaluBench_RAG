from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
from huggingface_hub import login
import pandas as pd
from typing import List, Dict
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re 
from langchain.schema import Document  
import matplotlib.pyplot as plt  
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
import nltk


class BM25Retriever:
  def __init__(self, texts: List[str]):
      self.texts = texts
      tokenized_texts = [text.split() for text in texts]
      self.bm25 = BM25Okapi(tokenized_texts)
  
  def get_relevant_documents(self, query: str, k: int = 3) -> List[str]:
      tokenized_query = query.split()
      doc_scores = self.bm25.get_scores(tokenized_query)
      top_k_indices = np.argsort(doc_scores)[-k:][::-1]
      return [self.texts[i] for i in top_k_indices]
  

class SemanticRetriever:  
    def __init__(self, texts):    
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  
        documents = [Document(page_content=text) for text in texts]   
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)  

    def get_relevant_documents(self, query, k=3):    
        docs = self.vectorstore.similarity_search(query, k=k)  
        return [doc.page_content for doc in docs] 


class HybridRetriever:
  def __init__(self, texts: List[str]):
      self.bm25_retriever = BM25Retriever(texts)
      self.semantic_retriever = SemanticRetriever(texts)
      
  def reciprocal_rank_fusion(self, rankings: List[List[str]], k: float = 60) -> Dict[str, float]:
      scores = {}
      for rank_list in rankings:
          for rank, doc in enumerate(rank_list):
              if doc not in scores:
                  scores[doc] = 0
              scores[doc] += 1 / (rank + k)
      return scores
  
  def get_relevant_documents(self, query: str, k: int = 3) -> List[str]:
      bm25_docs = self.bm25_retriever.get_relevant_documents(query, k)
      semantic_docs = self.semantic_retriever.get_relevant_documents(query, k)
      fusion_scores = self.reciprocal_rank_fusion([bm25_docs, semantic_docs])
      sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
      return [doc for doc, _ in sorted_docs[:k]]
  


def average_precision(retrieved_chunks, ground_truth_chunk):   
    relevant_retrieved = 0  
    precision_at_k = 0.0  
 
    for k, chunk in enumerate(retrieved_chunks, start=1):  
        if chunk == ground_truth_chunk:  
            relevant_retrieved += 1  
            precision_at_k += relevant_retrieved / k  
 
    if relevant_retrieved == 0:  
        return 0.0  
    
    return precision_at_k / relevant_retrieved 

def calculate_ndcg(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
  relevance = []
  
  for doc in retrieved_docs[:k]: 
      similarity = 1 if doc in relevant_docs else 0
      relevance.append(similarity)
            
  dcg = sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance)])
  ideal_relevance = sorted(relevance, reverse=True)
  idcg = sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])
  return dcg / idcg if idcg > 0 else 0


def count_words(text):  
    if pd.isna(text):
        return 0  
    return len(str(text).split())

def main():
  
	login(token="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")

	df = pd.read_parquet("hf://datasets/PatronusAI/HaluBench/data/test-00000-of-00001.parquet")
	# df= df.iloc[0:5, :]
	df.rename(columns={'passage': 'context'}, inplace=True)
	cleaned_rows = df.drop_duplicates(subset=['context', 'question'], keep='first')  
	cleaned_rows['word_count'] = cleaned_rows['context'].apply(count_words)
	
	texts = cleaned_rows['context'].tolist()
	bm25_retriever = BM25Retriever(texts)
	semantic_retriever = SemanticRetriever(texts) 
	hybrid_retriever = HybridRetriever(texts)

	bm25_chunks = []
	semantic_chunks = []
	hybrid_chunks = []


	for index, row in cleaned_rows.iterrows(): 

		query = row['question']  

		bm25_rel_docs = bm25_retriever.get_relevant_documents(query)
		semantic_rel_docs = semantic_retriever.get_relevant_documents(query)
		hybrid_rel_docs = hybrid_retriever.get_relevant_documents(query)

		bm25_chunks.append(bm25_rel_docs)
		semantic_chunks.append(semantic_rel_docs)
		hybrid_chunks.append(hybrid_rel_docs)

		cleaned_rows['bm25_chunks'] = bm25_chunks
		cleaned_rows['semantic_chunks'] = semantic_chunks
		cleaned_rows['hybrid_chunks'] = hybrid_chunks

	for index, row in cleaned_rows.iterrows(): 

		ground_truth_chunk = row['context'] 
		retrieved_chunks_bm25 = row['bm25_chunks']  
		retrieved_chunks_semantic = row['semantic_chunks']  
		retrieved_chunks_hybrid = row['hybrid_chunks'] 

		bm25_ap = average_precision(retrieved_chunks_bm25, ground_truth_chunk) 
		bm25_ndcg = calculate_ndcg(retrieved_chunks_bm25, ground_truth_chunk) 
		cleaned_rows.at[index, 'bm25_ap'] = bm25_ap
		cleaned_rows.at[index, 'bm25_ndcg'] = bm25_ndcg
		
		semantic_ap = average_precision(retrieved_chunks_semantic, ground_truth_chunk) 
		semantic_ndcg = calculate_ndcg(retrieved_chunks_semantic, ground_truth_chunk) 
		cleaned_rows.at[index, 'semantic_ap'] = semantic_ap
		cleaned_rows.at[index, 'semantic_ndcg'] = semantic_ndcg
		
		hybrid_ap = average_precision(retrieved_chunks_hybrid, ground_truth_chunk) 
		hybrid_ndcg = calculate_ndcg(retrieved_chunks_hybrid, ground_truth_chunk) 
		cleaned_rows.at[index, 'hybrid_ap'] = hybrid_ap
		cleaned_rows.at[index, 'hybrid_ndcg'] = hybrid_ndcg

	cleaned_rows.to_csv('main_results.csv', index=False)
	return cleaned_rows

# Usage
if __name__ == "__main__":
  result_df = main() 

x=2