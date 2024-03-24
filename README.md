# Automatic-literature-analysis-pipeline
Code repository for paper High-performance computing in healthcare: An automatic literature analysis perspective

## Paper extraction:

Filling Keywords and api_key in file config.yaml under folder Scopus_literature_extraction_v2 and execute elsevier_api_search_issn_v2.py for paper extraction.

## Initial topic modeling and outlier detection:

Execute Top2Vec_Initial_modeling+outlier_detection.py to conduct first-round topic modeling and detect and remove outlier topic (healthcare unrelated) using GPT-3 API

## Topic remodeling after removing outliers:

Execute Top2Vec_remodling_afteroutlier_detection_visualization.py to conduct Topic remodeling again after removing outliers, defining optimal topic number using dendrogram and generate bubble chart visualization

## Top modeling results visualizations:

The Jupyter notebook (Top modeling_results_visualizations.ipynb) is designed for the results visualization using words cloud, area chart, stacked area charts and violin chart.
