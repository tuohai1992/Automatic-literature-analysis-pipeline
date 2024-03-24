import pandas as pd
import numpy as np
import argparse
from top2vec import Top2Vec
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
import matplotlib.pyplot as plt
import umap.plot
import gensim.corpora as corpora
from gensim.utils import tokenize
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objs as go
import matplotlib
from matplotlib.colors import to_rgb
from scipy.spatial.distance import cdist
from textwrap import wrap
import openai
import re
import math
import csv
import pdb

data_full = pd.read_csv("data/scopus_search_results.csv")

# remove papers without abstract and also conference proceeding front page
data_full =  data_full[data_full["description"].notna()]

data_full =data_full[~data_full["subtypeDescription"].isin(['Conference Review'])]

data_full.reset_index(drop=True,inplace= True)
data_full['published_year']= pd.DatetimeIndex(data_full["coverDate"]).year.map(str)
data_full['published_month']= pd.DatetimeIndex(data_full["coverDate"]).month.map(str)
data_full['published_month'] = data_full['published_month'].apply(lambda x: '0'+str(x) if len(str(x))==1 else str(x))
data_full['published_year_month'] = data_full[['published_year','published_month']].agg('/'.join,axis =1)
data_citedby_count = data_full["citedby_count"]
data_fr = data_full
# get document_ids
data_fr.rename(columns = {'Unnamed: 0':'document_ids'}, inplace = True)

data_fr['document_ids'] = data_fr.index
# Get citation
cit_num = np.array(data_fr['citedby_count'].copy())

data = list(str(data_fr["title"][index])+". "+ str(item) \
                for index, item in enumerate(data_fr["description"]))
Health_topic = ['no']
health_topic_summary =[]
No_healthcare_topic = []
check = 1
iteration = 0


def top2vec():

    """
    Function of employing top2vec for the dataet
    """
    global check
    global iteration
    global data
    global data_fr
    global data_full
    global data_citedby_count
    global Health_topic
    global health_topic_summary
    global No_healthcare_topic
    while check:
        if Health_topic == []:
            iteration+=1
            print('interation: '+str(iteration))
            # Some pretrained models can be chosen ""
            #top2vec_model = Top2Vec(data, embedding_model='universal-sentence-encoder')
            top2vec_model = Top2Vec(data, speed="deep-learn",embedding_model='doc2vec', workers = 8)
        
            topic_vectors = top2vec_model.topic_vectors


            # return of the the sizes,topic word and index for topic
                    
            topic_sizes, topic_idx = top2vec_model.get_topic_sizes()
            topic_words, word_scores, topic_nums = top2vec_model.get_topics(len(topic_idx))
            # Detect if the combination of keywords related to healthcare using gpt3
            print('topic numbers: '+str(len(topic_vectors)))
            pdb.set_trace()
            print('Detect if the combination of keywords related to healthcare using gpt3')
            for i in range(topic_words.shape[0]):
                list_words=topic_words[i][:30].tolist()
                words_string = ", ".join(list_words)
                # Set up OpenAI API key
                openai.api_key = "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"

                # Define the prompt
                prompt = "Classification of healthcare-related keywords:\n\n"

                # Define the combination of keywords
                keywords = words_string

                # Add the combination of keywords to the prompt
                prompt += f"The combination of keywords is: {keywords}\n\n"
                prompt += "the importance of words increases as they appear earlier in the combination"

                # Add the classification task to the prompt
                prompt += "Is the combination of keywords related to healthcare or medical?"


                # Use the GPT-3 model to generate the classification
                response1 = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=200,
                    n=1,
                    stop=None,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                result1 = response1.choices[0].text.strip()

                # Check if result1 contains "yes" or "no" and set result1 to just that word
                if "yes" in result1.lower():
                    result1 = "yes"
                elif "no" in result1.lower():
                    result1 = "no"
                    No_healthcare_topic.append(i)

                if result1.lower() == "yes":
                    # Add the topic summarization task to the prompt
                    prompt += "\n\nIf the answer is 'yes', please summarize the healthcare topic in a refined phrase with less than 10 words."
    
                    # Use the GPT-3 model to summarize the healthcare topic
                    response2 = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        temperature=0,
                        max_tokens=200,
                        n=1,
                        stop=None,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    result2 = response2.choices[0].text.strip()
                    # Extract only the keywords from result2
                    keywords = [word.strip() for word in result2.split()[-5:]]
                    result2 = ", ".join(keywords).replace(",", "")
                else:
                    result2 = "N/A"
                Health_topic.append(result1)
                health_topic_summary.append(result2)
                print('topic'+str(i) +'completed')
            print(Health_topic)
            print(health_topic_summary)
            pdb.set_trace()


            # Build a topic_list={"topic_id":[doc_id.doc_id,...]}
            topic_list = defaultdict()
            for i in topic_idx:
                if i in No_healthcare_topic:
                    continue
                else:
                    documents, document_scores, document_ids = top2vec_model.search_documents_by_topic(topic_num=i, num_docs=topic_sizes[i])
                    for ii in document_ids:
                        topic_list[ii] = str(i)

    
            #Make doc id in sequence
            sorted_dict = dict(sorted(topic_list.items()))
            docs_df = pd.DataFrame()
            docs_df['Topic'] = sorted_dict.values()
            docs_df['Doc_ID'] = sorted_dict.keys()
            docs_df['title'] = [data_fr['title'].tolist()[i] for i in docs_df['Doc_ID'].tolist()]
            docs_df['description'] = [data_fr['description'].tolist()[i] for i in docs_df['Doc_ID'].tolist()]
            docs_df['Doc'] = [data[i] for i in docs_df['Doc_ID'].tolist()]
            docs_df['publish_year_month']=[data_full['published_year_month'].tolist()[i] for i in docs_df['Doc_ID'].tolist()]
            docs_df['citeby_count']= [data_citedby_count[i] for i in docs_df['Doc_ID'].tolist()]
            #docs_df = pd.merge(docs_df,topic_words_save, on='Topic')
            docs_df.to_csv("output/Top2Vec_output/document_cluster_Top2Vec_doc2vec.csv",index=False)



            #Calculate Coherence score
            tokenized = [list(tokenize(s)) for s in data]
            id2word = corpora.Dictionary(tokenized)
            corpus = [id2word.doc2bow(text) for text in tokenized]
        
            cm = CoherenceModel(topics=[s.tolist() for s in topic_words] ,texts=tokenized, corpus=corpus, dictionary=id2word, coherence='c_v')
        
            print("Model Coherence C_V is:{0}".format(cm.get_coherence()))
            print(str(len(docs_df['title'].tolist()))+'paper left')

        elif 'no' in Health_topic:
            iteration+=1
            print('interation: '+str(iteration))

            data_full = pd.read_csv("output/Top2Vec_output/document_cluster_Top2Vec_doc2vec.csv")

            data_citedby_count = data_full["citeby_count"]
            data_fr = data_full
            # get document_ids

            data_fr['Doc_ID'] = data_fr.index
            # Get citation
            cit_num = np.array(data_fr['citeby_count'].copy())

            data = data_fr['Doc'].tolist()

            # Some pretrained models can be chosen ""
            #top2vec_model = Top2Vec(data, embedding_model='universal-sentence-encoder')
            top2vec_model = Top2Vec(data, speed="deep-learn",embedding_model='doc2vec', workers = 8)
        
            topic_vectors = top2vec_model.topic_vectors


            # return of the the sizes,topic word and index for topic
                    
            topic_sizes, topic_idx = top2vec_model.get_topic_sizes()
            topic_words, word_scores, topic_nums = top2vec_model.get_topics(len(topic_idx))
            # Detect if the combination of keywords related to healthcare using gpt3
            print('topic numbers: '+str(len(topic_vectors)))
            pdb.set_trace()
            print('Detect if the combination of keywords related to healthcare using gpt3')

            Health_topic = []
            health_topic_summary =[]
            No_healthcare_topic = []
            for i in range(topic_words.shape[0]):
                list_words=topic_words[i][:30].tolist()
                words_string = ", ".join(list_words)
                # Set up OpenAI API key
                openai.api_key = "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"

                # Define the prompt
                prompt = "Classification of healthcare-related keywords:\n\n"

                # Define the combination of keywords
                keywords = words_string

                # Add the combination of keywords to the prompt
                prompt += f"The combination of keywords is: {keywords}\n\n"
                prompt += "the importance of words increases as they appear earlier in the combination"
                
                # Add the classification task to the prompt
                prompt += "Is the combination of keywords related to healthcare or medical?"

                # Use the GPT-3 model to generate the classification
                response1 = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=200,
                    n=1,
                    stop=None,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                result1 = response1.choices[0].text.strip()

                # Check if result1 contains "yes" or "no" and set result1 to just that word
                if "yes" in result1.lower():
                    result1 = "yes"
                elif "no" in result1.lower():
                    result1 = "no"
                    No_healthcare_topic.append(i)

                if result1.lower() == "yes":
                    # Add the topic summarization task to the prompt
                    prompt += "\n\nIf the answer is 'yes', please summarize the healthcare topic in a refined phrase with less than 10 words."
    
                    # Use the GPT-3 model to summarize the healthcare topic
                    response2 = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        temperature=0,
                        max_tokens=200,
                        n=1,
                        stop=None,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    result2 = response2.choices[0].text.strip()
                    # Extract only the keywords from result2
                    keywords = [word.strip() for word in result2.split()[-5:]]
                    result2 = ", ".join(keywords).replace(",", "")
                else:
                    result2 = "N/A"
                Health_topic.append(result1)
                health_topic_summary.append(result2)
                print('topic'+str(i) +'completed')
            print(Health_topic)
            print(health_topic_summary)
            pdb.set_trace()




            # Build a topic_list={"topic_id":[doc_id.doc_id,...]}
            topic_list = defaultdict()
            for i in topic_idx:
                if i in No_healthcare_topic:
                    continue
                else:
                    documents, document_scores, document_ids = top2vec_model.search_documents_by_topic(topic_num=i, num_docs=topic_sizes[i])
                    for ii in document_ids:
                        topic_list[ii] = str(i)

    
            #Make doc id in sequence
            sorted_dict = dict(sorted(topic_list.items()))
            docs_df = pd.DataFrame()
            docs_df['Topic'] = sorted_dict.values()
            docs_df['Doc_ID'] = sorted_dict.keys()
            docs_df['title'] = [data_fr['title'].tolist()[i] for i in docs_df['Doc_ID'].tolist()]
            docs_df['description'] = [data_fr['description'].tolist()[i] for i in docs_df['Doc_ID'].tolist()]
            docs_df['Doc'] = [data[i] for i in docs_df['Doc_ID'].tolist()]
            docs_df['publish_year_month']=[data_full['publish_year_month'].tolist()[i] for i in docs_df['Doc_ID'].tolist()]
            docs_df['citeby_count']= [data_citedby_count[i] for i in docs_df['Doc_ID'].tolist()]
            #docs_df = pd.merge(docs_df,topic_words_save, on='Topic')
            docs_df.to_csv("output/Top2Vec_output/document_cluster_Top2Vec_doc2vec.csv",index=False)


            #Calculate Coherence score
            tokenized = [list(tokenize(s)) for s in data]
            id2word = corpora.Dictionary(tokenized)
            corpus = [id2word.doc2bow(text) for text in tokenized]
        
            cm = CoherenceModel(topics=[s.tolist() for s in topic_words] ,texts=tokenized, corpus=corpus, dictionary=id2word, coherence='c_v')
        
            print("Model Coherence C_V is:{0}".format(cm.get_coherence()))
            print(str(len(docs_df['title'].tolist()))+'paper left')
        
        else:
            check = 0
            print('Outlier topic check finished')

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_path",
        type = str,
        default = "data/scopus_search_results.csv",
        help = "path of the dataset",
    )


    args = parser.parse_args()
    top2vec()