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



def top2vec(path_):
    """
    Function of employing top2vec for the dataet
    """
    embeddings = ['doc2vec']
    for embedding in embeddings:
        # Read data
        data_full = pd.read_csv(path_)

        data_citedby_count = data_full["citeby_count"]
        data_fr = data_full
        # get document_ids

        data_fr['Doc_ID'] = data_fr.index
        # Get citation
        cit_num = np.array(data_fr['citeby_count'].copy())

        data = data_fr['Doc'].tolist()

        # Some pretrained models can be chosen ""
        #top2vec_model = Top2Vec(data, embedding_model='universal-sentence-encoder')
        top2vec_model = Top2Vec(data, speed="deep-learn",embedding_model= embedding, workers = 8)
        
        topic_vectors = top2vec_model.topic_vectors


        # return of the the sizes,topic word and index for data_full
                    
        topic_sizes, topic_idx = top2vec_model.get_topic_sizes()
        topic_words, word_scores, topic_nums = top2vec_model.get_topics(len(topic_idx))
       
        

        # return of the the sizes,topic word and index for topic(continue)
        topic_words2 = topic_words[:,:20].tolist()


        #calculate distance matrix and create dendrogram

        num_topics = len(topic_vectors)
        global topics_nums
        topics_nums = len(topic_vectors)



        topic_sizes_dict={}
        topic_idx_dict={}
        topic_words_dict={}
        word_scores_dict={}
        topic_nums_dict={}
        topic_words2_dict={}
        topic_mapping_dict ={}

        topic_sizes_dict['topic_sizes_'+str(num_topics)] = topic_sizes
        topic_idx_dict['topic_idx_'+str(num_topics)]=topic_idx
        topic_words_dict['topic_words_'+str(num_topics)] = topic_words
        word_scores_dict['word_scores_'+str(num_topics)] = word_scores
        topic_nums_dict['topic_nums_'+str(num_topics)] = topic_nums_dict
        topic_words2_dict['topic_words2_'+str(num_topics)] = topic_words2
        topic_mapping_dict['topic_mapping_'+str(num_topics)] = [[i] for i in range(0,num_topics)]

        merged_topic_words = []

        for i in range (1,topic_sizes.shape[0]):
            topic_sizes_key = 'topic_sizes_'+str(num_topics-i)
            topic_idx_key = 'topic_idx_'+str(num_topics-i)
            topic_words_key = 'topic_words_'+str(num_topics-i)
            word_scores_key = 'word_scores_'+str(num_topics-i)
            topic_nums_key = 'topic_nums_'+str(num_topics-i)
            topic_words2_key = 'topic_words2_'+str(num_topics-i)
            topic_mapping_key = 'topic_mapping_'+str(num_topics-i)
            topic_mapping_dict[topic_mapping_key] = top2vec_model.hierarchical_topic_reduction(num_topics=num_topics-i)
            topic_sizes_dict[topic_sizes_key],topic_idx_dict[topic_idx_key]=top2vec_model.get_topic_sizes(reduced=True)
            topic_words_dict[topic_words_key],word_scores_dict[word_scores_key],topic_nums_dict[topic_nums_key] = top2vec_model.get_topics(len(topic_idx_dict[topic_idx_key]),reduced=True)       
            topic_words2_dict[topic_words2_key]= topic_words_dict[topic_words_key][:,:20].tolist()

            for topic_mapping in topic_mapping_dict['topic_mapping_'+str(num_topics-i)]:
                if topic_mapping not in topic_mapping_dict['topic_mapping_'+str(num_topics-i+1)]:
                    merge_index = topic_mapping_dict['topic_mapping_'+str(num_topics-i)].index(topic_mapping)
                    merged_topic_word = topic_words_dict[topic_words_key].tolist()[merge_index]
                    merged_topic_word.insert(0,'topic_num:'+str(num_topics-i))
                    merged_topic_words.append(merged_topic_word)

        # Initialize an empty distance matrix
        distance_matrix = np.empty((num_topics, num_topics))

        # Iterate over the topic vectors and calculate the cosine similarity between each pair of topics
        for i in range(num_topics):
            for j in range(num_topics):
                # Calculate the cosine similarity between the ith and jth topic vectors
                similarity = np.dot(topic_vectors[i], topic_vectors[j]) / (np.linalg.norm(topic_vectors[i]) * np.linalg.norm(topic_vectors[j]))
       
                # Convert the similarity to a distance by subtracting it from 1
                distance = 1 - similarity
       
                # Store the distance in the distance matrix
                distance_matrix[i, j] = distance

        mergings = linkage(distance_matrix, method='average',metric='cosine')
        top_10_elements = [sublist[:7] for sublist in topic_words2]
        matplotlib.rcParams['lines.linewidth'] = 5
        dn =dendrogram(mergings, labels=top_10_elements,leaf_rotation = 90)


        num_topics2 = len(topic_vectors)

        icoords = [0.5 * sum(icoord[1:3]) for icoord in dn['icoord']]
        dcoords = [dcoord[1] for dcoord in dn['dcoord'] ]
        combine = sorted(zip(icoords,dcoords),key=lambda x:x[1])

        

      
        for j, (icoord, dcoord) in enumerate(combine):
            num_topics2 -= 1
            plt.annotate("TN:{}".format(num_topics2), (icoord,dcoord), va='top', ha='center',fontsize=20)

        # Label the axes and save dendrogram
        plt.xlabel('Topic',fontsize=30)
        plt.ylabel('Similarity',fontsize=30)
        plt.xticks(fontsize=40)  # Increase x-axis content font size
        plt.yticks(fontsize=30)  # Increase y-axis content font size
        fig = plt.gcf()
        fig.set_size_inches(20, 10) 
        # Modify the line width of the dendrogram
        plt.savefig(f"dendrogram.eps",dpi=400,bbox_inches='tight',format='eps')
        plt.savefig(f"dendrogram.png",dpi=400,bbox_inches='tight')
        pdb.set_trace()


        # input reduced topic number
        print('Reduced topic number after checking dendrogram:')
        target_num_topics=int(input())
        topic_mapping = top2vec_model.hierarchical_topic_reduction(num_topics=target_num_topics)
        topic_sizes, topic_idx = top2vec_model.get_topic_sizes(reduced=True)
        topic_words, word_scores, topic_nums = top2vec_model.get_topics(len(topic_idx),reduced=True)
        # Detect if the combination of keywords related to healthcare using gpt3
        Health_topic = []
        health_topic_summary =[]
        for i in range(topic_words.shape[0]):
            list_words=topic_words[i][:20].tolist()
            words_string = ", ".join(list_words)
            # Set up OpenAI API key
            openai.api_key = "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"

            # Define the prompt
            prompt = "Classification of healthcare-related keywords:\n\n"

            # Define the combination of keywords
            keywords = words_string

            # Add the combination of keywords to the prompt
            prompt += f"The combination of keywords is: {keywords}\n\n"

            # Add the classification task to the prompt
            prompt += "Is the combination of keywords related to healthcare or medical? output should be yes or no"

            # Use the GPT-3 model to generate the classification
            response1 = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=10,
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

            if result1.lower() == "yes":
                # Add the topic summarization task to the prompt
                prompt += "\n\nIf the answer is 'yes', please summarize the healthcare or medical topic in a refined phrase with less than 10 words."
    
                # Use the GPT-3 model to summarize the healthcare topic
                response2 = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=10,
                    n=1,
                    stop=None,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                result2 = response2.choices[0].text.strip()
            else:
                result2 = "N/A"
            Health_topic.append(result1)
            health_topic_summary.append(result2)

        topic_words_short = topic_words[:,:5].tolist()
        topic_words_save = pd.DataFrame(list(zip(topic_nums,topic_words,topic_words_short,word_scores,Health_topic,health_topic_summary)),columns=['Topic','topic_words','topic_words_short','word_scores','Health_topic','health_topic_summary'])
        topic_words_save['Topic'] = topic_words_save['Topic'].astype(str)
        topic_words_save.to_csv("output/Top2Vec_output/document_cluster_Top2Vec_"+str(embedding)+"_topic_words.csv",index=False)
        print('topic'+str(i)+'completed')
        print(Health_topic)
        print(health_topic_summary)
        pdb.set_trace()

        ## Document vector
        print("len(model.document_vectors): ", len(top2vec_model.document_vectors))
        ## Topic vector
        print("len(model.topic_vectors[0]): ", len(top2vec_model.topic_vectors_reduced[0]))
        top_num = len(top2vec_model.topic_vectors_reduced)
        print("len(model.topic_vectors): ", top_num)
        topic_sizes, topic_nums = top2vec_model.get_topic_sizes(reduced=True)

        # 1. Get 2d representation from UMAP including documents and topic, by umap_model.embedding_
        # 2. For-loop to draw circle with certain percentage

        ### reference code for ploting: 
        ### def points() and def def _matplotlib_points() from ummap https://github.com/lmcinnes/umap/blob/615cb1adf3611d6c138f3794a8202bbf339587a2/umap/plot.py#L152
        ### def def _matplotlib_points()

        all_vectors= np.concatenate((top2vec_model.document_vectors, top2vec_model.topic_vectors_reduced))
        all_top = np.concatenate((top2vec_model.doc_top_reduced, np.arange(top_num)))

        print("len(all_vectors): ", len(all_vectors))
        print("len(all_top): ", len(all_top))

        umap_args_model = {
            "n_neighbors": 10,
            "n_components": 2, # 5 -> 2 for plotting 
            "metric": "cosine",
        }
        umap_model = umap.UMAP(**umap_args_model).fit(all_vectors)
        # umap.plot.points(umap_model, labels = all_top)


        # Larger quantile -> larger circle
        quantile = 0.7
        # Top n most hghest paper cared
        high_number= 3

        umap_document_vectors = umap_model.embedding_[:-top_num,:]
        umap_topic_vectors = umap_model.embedding_[-top_num::,:]
        topic_words, _, _ = top2vec_model.get_topics(num_topics=top_num,reduced=True)

        # find color schema
        colors = list(matplotlib.colors.CSS4_COLORS.keys())
        rgb_colors = [to_rgb(color) for color in colors]
        euclidean_distance = cdist(rgb_colors, rgb_colors, 'euclidean')

        # Set the diagonal to infinity so colors are not compared to themselves
        np.fill_diagonal(euclidean_distance, np.inf)

        # Find the indices of the colors with the biggest color difference
        top= np.argpartition(euclidean_distance.ravel(), -40)[-40:]

        top_colors = [colors[i // len(colors)] for i in top]

        color_to_remove=['mintcream','mistyrose','moccasin','navajowhite','oldlace','navy','whitesmoke','linen','lightyellow','lightslategray']

        top_colors = [x for x in top_colors if x not in color_to_remove]
        top_colors
        fig = go.Figure()
        for i, num in enumerate(topic_sizes[:top_num]):
            _, _, topic_document_ids = top2vec_model.search_documents_by_topic(topic_num=i, num_docs=num,reduced=True)

            # Get citation ranking and find high cited paper
            citation_ranking = np.argsort(cit_num[topic_document_ids])[-high_number::]
            high_cit_docs = topic_document_ids[citation_ranking]
    
            #Get the distance from certain quantile docu to corresponding topic, based on 2d embedding
            all_dist = [math.dist(umap_document_vectors[id], umap_topic_vectors[i]) for id in topic_document_ids]
            quantile_dist = np.quantile(all_dist, quantile)
    
            # The topic circle
            fig.add_shape(type="circle",
                xref="x", yref="y",
                fillcolor=top_colors[i],
                x0=umap_topic_vectors[i][0]-quantile_dist, y0=umap_topic_vectors[i][1]-quantile_dist, x1=umap_topic_vectors[i][0]+quantile_dist, y1=umap_topic_vectors[i][1]+quantile_dist,
                line_color=top_colors[i],opacity = 0.3)

    
            # Get high citation doc plot

            fig.add_trace(go.Scatter(
                x = np.array(umap_document_vectors[high_cit_docs, 0]),
                y = np.array(umap_document_vectors[high_cit_docs, 1]),
                mode = 'markers',
                marker = dict(size = 18, color = top_colors[i], opacity = 1,symbol='square'),
                hovertemplate ='<b>%{text}</b><extra></extra>',
                text = ['topic:{}<br>Paper_ID:{}<br>Title:{}<br>publish_year:{}<br>citation_count:{}<br>Abstract:{}'.format(str(i),str(a),data_fr.loc[a,'title'],data_fr.loc[a,'publish_year_month'],data_fr.loc[a,'citeby_count'],re.sub(r"(.{100})", r"\1<br>", data_fr.loc[a,'description'], 0, re.DOTALL)) for a in high_cit_docs],
                showlegend = False
            ))
                # scatter plot of all documents
            fig.add_trace(go.Scatter(
                x = np.array(umap_document_vectors[topic_document_ids, 0]),
                y = np.array(umap_document_vectors[topic_document_ids, 1]),
                mode = 'markers',
                marker = dict(size = 2, color = top_colors[i], opacity = 1,symbol='circle'),
                hoverinfo='skip',showlegend = False
            ))
            # The topic location (central of the circle) 
            fig.add_trace(go.Scatter(
                x = np.array(umap_topic_vectors[i][0]),
                y = np.array(umap_topic_vectors[i][1]),
                mode = 'markers+text',
                marker = dict(size = 25, color = top_colors[i], opacity = 1, symbol='triangle-up'),
                hovertemplate ='<b>%{text}</b><extra></extra>',
                text = ['topic '+str(i)+ str(topic_words[i][:6])],
                name = 'topic '+str(i)+ str(topic_words[i][:6]),
                textposition='bottom center',
                textfont=dict(size=16)

            ))
            fig.update_xaxes(zeroline=False,visible=False, showticklabels=False)
            fig.update_yaxes(visible=False, showticklabels=False,scaleanchor = 'x',scaleratio = 1)
            fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)'
            },
            legend=dict(
                font=dict(
                    size=28,  # Increase legend font size
                )
            ),
            width=2300,
            height =1600,
            hoverlabel=dict(
                bgcolor="white",
                font=dict(size=18),
            ),
            font=dict(family = 'Arial',size = 13,color ='black'),


        )
        fig.show()
        pdb.set_trace()
        #rest codes


        # Build a topic_list={"topic_id":[doc_id.doc_id,...]}
        topic_list = defaultdict()
        for i in topic_idx:
            documents, document_scores, document_ids = top2vec_model.search_documents_by_topic(topic_num=i,reduced=True, num_docs=topic_sizes[i])
            for ii in document_ids:
                topic_list[ii] = str(i)
        #Make doc id in sequence
        sorted_dict = dict(sorted(topic_list.items()))
        #pdb.set_trace()
        docs_df = pd.DataFrame()
        docs_df['Topic'] = sorted_dict.values()
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_df['title'] = data_fr['title']
        docs_df['description'] = data_fr['description']
        docs_df['Doc'] = data
        docs_df['publish_year_month']=data_full['publish_year_month']
        docs_df['citeby_count']= data_citedby_count
        #docs_df = pd.merge(docs_df,topic_words_save, on='Topic')
        docs_df.to_csv("output/Top2Vec_output/document_cluster_Top2Vec_"+str(embedding)+".csv",index=False)


        #Calculate Coherence score
        tokenized = [list(tokenize(s)) for s in data]
        id2word = corpora.Dictionary(tokenized)
        corpus = [id2word.doc2bow(text) for text in tokenized]
        
        cm = CoherenceModel(topics=[s.tolist() for s in topic_words] ,texts=tokenized, corpus=corpus, dictionary=id2word, coherence='c_v')
        
        print("Model Coherence C_V is:{0}".format(cm.get_coherence()))


        print('embedding'+str(embedding)+'coherence_score'+str(cm.get_coherence()))
        print(str(len(docs_df['title'].tolist()))+'paper')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_path",
        type = str,
        default = "output/Top2Vec_output/document_cluster_Top2Vec_doc2vec.csv",
        help = "path of the dataset",
    )


    args = parser.parse_args()
    top2vec(path_ = args.data_path)
