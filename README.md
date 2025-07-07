I implemented a three-step topic modeling pipeline that leverages both BERTopic and Generative AI (GenAI) to generate, refine, and assign topics to a large set of documents. The goal was to go beyond traditional static topic models and build a flexible, accurate, and human-aligned topic taxonomy. Here’s how the system works:


🔹 Step 1: Topic Generation – Creating Raw Topics from Document Clusters
🔸 Purpose:
To generate candidate topics directly from the content, allowing for novel and data-specific topics to emerge.
🔸 Process:
1. Clustering with BERTopic:
    * We start by clustering documents using BERTopic, which uses transformer embeddings and HDBSCAN to group similar documents.
    * Each cluster essentially represents a theme or subject across multiple documents.
2. Sampling Representative Documents:
    * From each cluster, we sample representative documents that best capture the cluster’s meaning.
3. Prompt Construction:
    * A detailed prompt (prompt1) is crafted and includes:
        * Instructions to the language model on how to generate topic labels.
        * Examples of good topics.
        * A guideline to create new topics unless there's an exact match to an existing one.
4. Topic Extraction using GenAI:
    * Each group of documents is fed into a GenAI model with prompt1.
    * The model returns interpreted topic names—often in natural language format.
    * These are stored in a new dataframe column called MetIQ_Predicted_Topic.
5. Post-processing & Cleanup:
    * If multiple topics are returned (comma-separated), we split and standardize them:
        * Convert to lowercase, remove special characters, and capitalize consistently.
    * Duplicate topics are also removed.
✅ Outcome:
This step creates an initial list of granular and human-readable topics, grounded in the document content.

🔹 Step 2: Topic Refinement – Merging and Normalizing Topics
🔸 Purpose:
To reduce redundancy and ensure coherence across the topic set while keeping meaningful distinctions.
🔸 Process:
1. Refinement Prompt Design (prompt2):
    * A prompt is designed to ask the GenAI model to group similar or overlapping topics into generalized categories.
    * The prompt includes examples, such as how "customer support" and "customer service" can be merged into one.
2. Merging Topics using GenAI:
    * The list of generated topics from Step 1 is passed to the model via prompt2.
    * The model identifies synonyms or near-duplicates and suggests a consolidated list of topics.
3. Result Parsing:
    * The model's response is cleaned:
        * Remove unnecessary characters (e.g., apostrophes).
        * Parse it into a valid Python list using ast.literal_eval().
✅ Outcome:
A refined and deduplicated list of topics is now ready to be used for assigning labels, ensuring consistency across the dataset.

🔹 Step 3: Topic Assignment – Tagging Documents with Refined Topics
🔸 Purpose:
To consistently label each document with the most relevant topic(s) from the refined list created in Step 2.
🔸 Process:
1. Assignment Prompt Design (prompt3):
    * This prompt includes the full list of refined topics.
    * The instructions clearly state:
        * Do not invent new topics.
        * Only choose the most relevant existing topic(s) from the list.
2. Filtering & Sampling:
    * Documents with fewer than 10 words are excluded to avoid noise.
    * A test subset is selected (based on test_size) for efficient evaluation.
3. Topic Assignment using GenAI:
    * Each document is passed to the model along with prompt3.
    * The model evaluates the content and assigns one or more most relevant topics.
    * If nothing fits, it returns "No topic".
4. Saving Results:
    * The results are saved in a new column MetIQ_Assigned_Topic.
    * Final output is exported to an Excel file at the specified result_path.
✅ Outcome:
Documents are tagged accurately and consistently, using a controlled vocabulary of refined topics.

🚀 Highlights of This Pipeline:
* Hybrid Approach: Combines unsupervised clustering (BERTopic) with human-aligned interpretation (GenAI).
* Content-Aware: Topics are derived from the data, not predefined.
* Generative Intelligence: GenAI plays a key role in semantic interpretation and consolidation.
* Scalability: Easily applicable to large datasets due to automation and sampling.
  
