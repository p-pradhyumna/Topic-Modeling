# Library Import
!pip install python-dotenv

import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import sys
import ipywidgets as widgets
from IPython.display import display
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import warnings
from collections import Counter
from bertopic import BERTopic
import random
import math
import configparser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings("ignore")
sys.path.append('')
from dotenv import load_dotenv
import os

# Initial Setup

## Set up the LLaMA Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_model_path = './Llama'  # Update if different
llama_model = AutoModelForCausalLM.from_pretrained(local_model_path)
llama_model = llama_model.to(device)
llama_tokenizer = AutoTokenizer.from_pretrained(local_model_path)

## Initial Settings (update config.ini file)
result_path = r'C:\Users\ppaloju\OneDrive - MetLife\Documents\MetIQ_result.xlsx'
data_file_path = r'C:\Users\ppaloju\OneDrive - MetLife\Documents\RMfile.xlsx'
text_column = 'interaction'
num_sample_data = 200
random_seed = 42
test_size = 100

ini_exampled_topic_set = {"Termination", "Bill Generation & Payment related", "Action Taken and Resolutions"}
Document_1 = "BA had retro terminations Terminations Processed"
topic_1 = "Termination"
Document_2 = "Billing issue No action required, the CSC placed a call with the group & provided a bill vs paid report for review and advise if any additional research is needed."
topic_2 = "Bill Generation & Payment related"
Document_3 = "Group was terminated in then instated and the ee are still showing terminated in GF"
topic_3 = "Action Taken and Resolutions"
Document_4 = ""
topic_4 = ""

config = configparser.ConfigParser()
config.read('./config.ini')
result_path = config.get('dataset', 'result_path')
data_file_path = config.get('dataset', 'data_file_path')
text_column = config.get('dataset', 'text_column')
num_sample_data = config.getint('dataset', 'num_sample_data')
random_seed = config.getint('dataset', 'random_seed')
test_size = config.getint('dataset', 'test_size')
use_examples = config.getboolean('Domain example', 'use_examples', fallback=True)
ini_exampled_topic_set = config.get('Domain example', 'ini_exampled_topic_set')
ini_exampled_topic_set = ast.literal_eval(ini_exampled_topic_set)
Document_1 = config.get('Domain example', 'Document_1')
topic_1 = config.get('Domain example', 'topic_1')
Document_2 = config.get('Domain example', 'Document_2')
topic_2 = config.get('Domain example', 'topic_2')
Document_3 = config.get('Domain example', 'Document_3')
topic_3 = config.get('Domain example', 'topic_3')
Document_4 = config.get('Domain example', 'Document_4')
topic_4 = config.get('Domain example', 'topic_4')

## Load the Data
raw_df = pd.read_excel(data_file_path)
raw_df[text_column] = raw_df['COMMENTS'] + ' ' + raw_df['ACTIONS']
raw_df[text_column] = raw_df[text_column].fillna("").astype(str)

## Preprocess for Call Center Transcripts (Optional)
def preprocess_transcript(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\b(um|uh|like|you know|hello|hi|good morning|thank you|bye)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(Agent|Customer)\s*:\s*', '', text, flags=re.MULTILINE)
    return ' '.join(text.split()).strip()

def segment_transcript(text, max_length=500):
    if not text:
        return []
    sentences = text.split('.')
    segments = []
    current_segment = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length:
            segments.append('. '.join(current_segment) + '.')
            current_segment = [sentence]
            current_length = sentence_length
        else:
            current_segment.append(sentence)
            current_length += sentence_length
    if current_segment:
        segments.append('. '.join(current_segment) + '.')
    return segments

# Apply preprocessing (uncomment for transcripts)
# raw_df[text_column] = raw_df[text_column].apply(preprocess_transcript)
# raw_df['segments'] = raw_df[text_column].apply(segment_transcript)
# raw_df = raw_df.explode('segments').reset_index(drop=True)
# raw_df = raw_df.rename(columns={'segments': text_column})
# raw_df = raw_df[raw_df[text_column].notna() & (raw_df[text_column] != '')]
  

## BERTopic for Sampling
emb_model = SentenceTransformer('./model/all-mpnet-base-v2')
topic_model = BERTopic(embedding_model=emb_model, min_topic_size=3)
cleaned_text = raw_df[text_column].str.strip()
cleaned_text = cleaned_text[cleaned_text != ""]
documents = cleaned_text.str.lower().drop_duplicates()
topics, probs = topic_model.fit_transform(documents)

topic_info = topic_model.get_topic_info()
topic_representations = {topic_id: ", ".join([word for word, _ in topic_model.get_topic(topic_id)]) for topic_id in topic_info["Topic"]}
doc_topic_df = pd.DataFrame({"doc_id": range(len(documents)), "text": documents, "topic": topics})
valid_topics = [t for t in topic_representations.keys() if t != -1]

if len(valid_topics) > num_sample_data:
    sample_per_topic = 1
else:
    sample_per_topic = math.ceil(num_sample_data / len(valid_topics))

selected_samples = []
random.seed(random_seed)
for topic_id in valid_topics:
    topic_docs = doc_topic_df[doc_topic_df["topic"] == topic_id]
    num_docs_in_topic = len(topic_docs)
    num_to_sample = min(sample_per_topic, num_docs_in_topic)
    sampled_docs = topic_docs.sample(num_to_sample, random_state=random_seed) if num_docs_in_topic > 0 else pd.DataFrame()
    for _, row in sampled_docs.iterrows():
        selected_samples.append((topic_id, row["text"], topic_representations[topic_id]))
sampled_df = pd.DataFrame(selected_samples, columns=["topic", "sampled_text", "key_words_representation"])

def create_genai_input(df):
    merged_df = df.groupby(["topic", "key_words_representation"])["sampled_text"].apply(lambda x: "\n***************************\n".join(x)).reset_index()
    merged_df["merged_text"] = merged_df["sampled_text"] + "\n\n[key words representation]\n" + merged_df["key_words_representation"]
    return merged_df[["merged_text"]]

genai_input_df = create_genai_input(sampled_df)

# Step 1: Topic Generation
example_document_list = []
example_topic_list = []
for i in range(1, 5):
    document = globals()[f"Document_{i}"]
    topic = globals()[f"topic_{i}"]
    if document != "" and topic != "":
        example_document_list.append(document)
        example_topic_list.append(topic)

def prepare_example(topic_set, example_documents, example_topic_list):
    if not use_examples:
        return ""
    result = "[Examples]\nExampled topic list: {}\n-----------------\n".format(list(topic_set))
    for i in range(len(example_documents)):
        if example_topic_list[i] not in topic_set:
            result += "Example: Generating a new topic: \"{}\"\nDocument:\n{}\nYour response:\n{}\n--------------------\n".format(
                example_topic_list[i], example_documents[i], example_topic_list[i])
        else:
            result += "Example: Using an existing topic: \"{}\" (only if no better new topic can be generated).\nDocument:\n{}\nYour response:\n{}\n--------------------\n".format(
                example_topic_list[i], example_documents[i], example_topic_list[i])
    return result

prompt1 = """
You will receive single/multiple documents and a set of topics. Your task is to identify topics within the document. 
Your priority is to generate NEW topics unless an existing topic in the provided topic list PERFECTLY captures the documents' essence.

[TOPICS]
{topic_list}
{examples}

[Instructions]
Step 1: Prioritize creating new topics whenever possible:
- Always attempt to generate a NOVEL topic (2-5 words) unless an exact match exists in the topic list.
- Avoid generic terms (e.g., "Issues", "Problems") or single letters.
- Ensure topics are DIVERSE and cover DISTINCT aspects (e.g., billing, terminations, resolutions).
- Avoid multiple topics for the same theme (e.g., do not generate "Billing Issues" and "Billing Problems").
- Return AT MOST 1-2 topics per document.
- Use the [key words representation] at the end of each document list for reference.

Step 2: Identify general topic(s) in the document:
- If a single dominant topic is identified, return that topic.
- If multiple strong topics emerge, return up to 2 topics separated by a comma.
- Strive for GENERIC and UNIQUE topics that represent all given documents.

[Document]
{document}

[Your response]
Return ONLY the relevant topics without any additional text or prefixes.
Example: Unacceptable: "Issues", "Billing Problems"
Document: The billing was incorrect.
Your response: Billing Errors
"""

prompt1 = prompt1.format(examples=prepare_example(ini_exampled_topic_set, example_document_list, example_topic_list))

prompt1_copy = prompt1[:]
topic_set_copy = ini_exampled_topic_set.copy()

def get_response(row):
    global topic_set_copy
    updated_topic_list = list(topic_set_copy.copy())
    exe_prompt1 = prompt1_copy.format(topic_list='\n'.join(updated_topic_list), document=row["merged_text"])
    inputs = llama_tokenizer(exe_prompt1, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.85,
            temperature=0.5,
            pad_token_id=llama_tokenizer.eos_token_id,
            eos_token_id=llama_tokenizer.eos_token_id
        )
    result = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if "[Your response]" in result:
        result = result.split("[Your response]")[-1].strip()
    topics = [t.strip() for t in result.split(',') if t.strip()]
    cleaned_topics = [t for t in topics if t.lower() not in ["general comment", "none", "", "issues", "problems"]]
    topic_set_copy.update(cleaned_topics)
    return ', '.join(cleaned_topics)

def apply_with_progress(df, func):
    tqdm.pandas()
    return df.progress_apply(func, axis=1)

genai_input_df["Llama_Predicted_Topic"] = apply_with_progress(genai_input_df, get_response)

topic_list = []
for item in genai_input_df["Llama_Predicted_Topic"]:
    if pd.notna(item) and item.strip():
        topics = [t.strip() for t in item.split(',') if t.strip()]
        topic_list.extend(topics)

clean_topic_list = [re.sub(r'[^A-Za-z0-9 ]+', '', x).strip().lower().capitalize() for x in topic_list]
clean_topic_list = list(set(clean_topic_list))

def clean_topics(topics):
    filtered_list = [t for t in topics if not (
        len(t.strip()) <= 2 or
        t.lower() in ['issues', 'problems', 'general', 'none']
    )]
    emb_model = SentenceTransformer('./model/all-mpnet-base-v2')
    embeddings = emb_model.encode(filtered_list)
    cleaned_list = []
    seen = set()
    for i, topic in enumerate(filtered_list):
        if topic.lower() in seen:
            continue
        topic_emb = embeddings[i]
        is_unique = True
        for seen_topic, seen_emb in cleaned_list:
            if cosine_similarity([topic_emb], [seen_emb])[0][0] > 0.8:
                is_unique = False
                break
        if is_unique:
            cleaned_list.append((topic, topic_emb))
            seen.add(topic.lower())
    return [t for t, _ in cleaned_list]

clean_topic_list = clean_topics(clean_topic_list)
print("Clean topic list after Step 1:", clean_topic_list)

# Step 2: Topic Refinement
from sklearn.cluster import AgglomerativeClustering

def cluster_topics(topics):
    if not topics:
        return []
    emb_model = SentenceTransformer('./model/all-mpnet-base-v2')
    embeddings = emb_model.encode(topics)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, linkage='average')
    labels = clustering.fit_predict(embeddings)
    clustered_topics = {}
    for topic, label in zip(topics, labels):
        if label not in clustered_topics:
            clustered_topics[label] = []
        clustered_topics[label].append(topic)
    return [max(cluster, key=len) for cluster in clustered_topics.values()]

clean_topic_list = cluster_topics(clean_topic_list)

prompt2 = """
You are an expert in topic clustering for insurance customer feedback. Your task is to analyze the provided list of topics and create EXACTLY 10 DISTINCT and MEANINGFUL categories.

[Instructions]:
1. Create EXACTLY 10 categories that are COMPLETELY DISTINCT - each must represent a different aspect of feedback.
2. DO NOT use variations of the same concept (e.g., avoid "Billing Issues" and "Billing Problems").
3. Consolidate similar topics into a single category (e.g., merge all billing-related topics into "Billing & Payments").
4. Use specific, descriptive names (2-5 words) relevant to insurance feedback.
5. Ensure categories cover ALL major feedback aspects (e.g., billing, claims, terminations, customer support).
6. DO NOT use generic placeholders (e.g., "Category1").
7. Return ONLY a Python list with EXACTLY 10 unique categories.

[Examples]:
- Unacceptable: ['Billing Issues', 'Billing Problems', 'Payment Issues']
- Acceptable: ['Billing & Payments', 'Claims Processing', 'Termination Issues', 'Customer Support']

[Input list]
{topics}

[Your response]
['Billing & Payments', 'Claims Processing', 'Termination Issues', 'Customer Support', 'Account Management', 'Enrollment Process', 'Technical Support', 'Plan Coverage', 'Policy Clarity', 'Communication Issues']
"""

inputs = llama_tokenizer(prompt2.format(topics="\n".join(clean_topic_list)), return_tensors="pt", truncation=True).to(device)
with torch.no_grad():
    outputs = llama_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.1,
        pad_token_id=llama_tokenizer.eos_token_id,
        eos_token_id=llama_tokenizer.eos_token_id
    )

result = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
if "[Your response]" in result:
    response_text = result.split("[Your response]")[-1].strip()
else:
    response_text = result.strip()

try:
    refined_clean_topic_list = ast.literal_eval(response_text)
    refined_clean_topic_list = list(dict.fromkeys(t for t in refined_clean_topic_list if not re.match(r'Category\d+', t)))
    if len(refined_clean_topic_list) < 10:
        topic_counts = Counter(clean_topic_list)
        top_topics = [t for t, _ in topic_counts.most_common(10 - len(refined_clean_topic_list))]
        refined_clean_topic_list.extend(t for t in top_topics if t not in refined_clean_topic_list)
    refined_clean_topic_list = refined_clean_topic_list[:10]
except:
    matches = re.findall(r"['\"]([^'\"]+?)['\"]", response_text)
    refined_clean_topic_list = list(dict.fromkeys(m for m in matches if not re.match(r'Category\d+', m)))
    if len(refined_clean_topic_list) < 10:
        topic_counts = Counter(clean_topic_list)
        top_topics = [t for t, _ in topic_counts.most_common(10 - len(refined_clean_topic_list))]
        refined_clean_topic_list.extend(t for t in top_topics if t not in refined_clean_topic_list)
    refined_clean_topic_list = refined_clean_topic_list[:10]

if len(set(refined_clean_topic_list)) != 10:
    raise ValueError("Step 2 did not produce exactly 10 unique categories")

print("Refined topic list after Step 2:", refined_clean_topic_list)

# Step 3: Topic Assignment
prompt3 = """
You will receive a document and a predefined topic list. Your task is to assign the document to the most relevant topic from the list based on the content.

[Instructions]
1. Only use topics from the provided list - DO NOT create new topics.
2. Return ONLY the topic selected from the list, without any additional text or prefix.
3. For multiple related topics, return the most dominant topic.
4. If the document does not relate to any topic, return "No topic".

Topic List:
{topic_list}

[Document]:
{document}

[Your response]
"""

prompt3 = prompt3.format(topic_list=", ".join(refined_clean_topic_list))

def get_response_step3(row):
    exe_prompt3 = prompt3.format(document=row[text_column])
    inputs = llama_tokenizer(exe_prompt3, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            temperature=0.1,
            pad_token_id=llama_tokenizer.eos_token_id,
            eos_token_id=llama_tokenizer.eos_token_id
        )
    result = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if "[Your response]" in result:
        result = result.split("[Your response]")[-1].strip()
    result = re.sub(r'\[.*?\]', '', result).strip()
    return result if result in refined_clean_topic_list else "No topic"

raw_df["Llama_Assigned_Topic"] = apply_with_progress(raw_df, get_response_step3)
raw_df.to_excel(result_path, index=False)

# Validation
print("Assigned topic distribution:", raw_df['Llama_Assigned_Topic'].value_counts())
sample_results = raw_df[[text_column, 'Llama_Assigned_Topic']].sample(5)
for _, row in sample_results.iterrows():
    print(f"Document: {row[text_column][:100]}...\nAssigned Topic: {row['Llama_Assigned_Topic']}\n")

