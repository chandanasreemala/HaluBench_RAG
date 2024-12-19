from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
import torch
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


dataframe = pd.read_csv('main_results.csv')
df_factual = dataframe[dataframe['label'] == 'PASS'].reset_index(drop=True)
df_factual =df_factual.iloc[:1, :].reset_index(drop=True)


class LlamaQAPipeline:
    def __init__(self):
        #self.model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_prompt(self, context: str, question: str) -> str:
        """
        Prompt for the model using context and question
        """
        prompt = f"""[INST] You are a helpful AI assistant. Based on the following context, 
        please only provide the answer to the question. Please answer the question accurately and concisely.

        Context: {context}

        Question: {question}

        Answer: [/INST]"""
        return prompt

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generate answer using the model
        """
        prompt = self.generate_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            inputs.input_ids,
            max_length=6144,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the answer part after "[/INST]"
        #extracted_answer = response.split("[/INST]")[-1].strip()
        return response
    
def clean_output(output: str) -> str:  
    """  
    Cleans the model's output to extract only the answer.  
    """  
    # Split the output by the "[/INST]" token and take the last part  
    if "[/INST]" in output:  
        output = output.split("[/INST]")[-1].strip()  

    # Remove any remaining special tokens or unnecessary text  
    output = output.replace("[INST]", "").strip()  

    return output  


# Initialize the pipeline and evaluator
qa_pipeline = LlamaQAPipeline()

# bm25_data = df_factual[["bm25_chunks", "question", "answer"]].to_dict('records')
# semantic_data = df_factual[["semantic_chunks", "question", "answer"]].to_dict('records')
# hybrid_data = df_factual[["hybrid_chunks", "question", "answer"]].to_dict('records')

data = df_factual[["bm25_chunks", "semantic_chunks","hybrid_chunks","question", "answer"]].to_dict('records')


# Generate predictions and evaluate
bm25_predictions = []
semantic_predictions = []
hybrid_predictions = []
ground_truths = []
results = pd.DataFrame()

for item in data:
    bm25_prediction = qa_pipeline.generate_answer(item['bm25_chunks'], item['question'])
    bm25_predictions.append(bm25_prediction)

    semantic_prediction = qa_pipeline.generate_answer(item['semantic_chunks'], item['question'])
    semantic_predictions.append(semantic_prediction)

    hybrid_prediction = qa_pipeline.generate_answer(item['hybrid_chunks'], item['question'])
    hybrid_predictions.append(hybrid_prediction)

    ground_truths.append(item['answer'])
    results = pd.concat([results, pd.DataFrame({
        'question': [item['question']],
        'answer': [item['answer']],
        'bm25_context': [item['bm25_chunks']],
        'bm25_prediction': [bm25_predictions],
        'semantic_context': [item['semantic_chunks']],
        'semantic_prediction': [semantic_predictions],
        'hybrid_context': [item['hybrid_chunks']],
        'hybrid_prediction': [hybrid_predictions]
        
    }, index = [0])])

# for item in bm25_data:
#     bm25_prediction = qa_pipeline.generate_answer(item['bm25_chunks'], item['question'])
#     bm25_predictions.append(bm25_prediction)
#     ground_truths.append(item['answer'])
#     results = pd.concat([results, pd.DataFrame({
#         'question': [item['question']],
#         'answer': [item['answer']],
#         'bm25_context': [item['bm25_chunks']],
#         'bm25_prediction': [bm25_predictions]
#         #'extracted_prediction': bm25_prediction.replace(r'\[.*?\]', '', regex=True)
#     }, index = [0])])

# for item in semantic_data:
#     semantic_prediction = qa_pipeline.generate_answer(item['semantic_chunks'], item['question'])
#     semantic_predictions.append(semantic_prediction)
#     # ground_truths.append(item['answer'])
#     results = pd.concat([results, pd.DataFrame({
#         # 'question': [item['question']],
#         # 'answer': [item['answer']],
#         'semantic_context': [item['semantic_chunks']],
#         'semantic_prediction': [semantic_predictions]
#         #'extracted_prediction': semantic_prediction.replace(r'\[.*?\]', '', regex=True)
#     }, index = [0])])


# for item in hybrid_data:
#     hybrid_prediction = qa_pipeline.generate_answer(item['hybrid_chunks'], item['question'])
#     hybrid_predictions.append(hybrid_prediction)
#     # ground_truths.append(item['answer'])
#     results = pd.concat([results, pd.DataFrame({
#         # 'question': [item['question']],
#         # 'answer': [item['answer']],
#         'hybrid_context': [item['hybrid_chunks']],
#         'hybrid_prediction': [hybrid_predictions]
#         #'extracted_prediction': hybrid_prediction.replace(r'\[.*?\]', '', regex=True)
#     }, index = [0])])


#save the results dataframe to csv
results.to_csv("generated_output.csv", index=False)
x = 2