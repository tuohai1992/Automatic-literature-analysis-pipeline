# Automatic-literature-analysis-pipeline
This code repository supports the paper "High-performance Computing in Healthcare: An Automatic Literature Analysis Perspective."

## Paper extraction:

Fill in the required fields (Keywords and api_key) in the config.yaml file located under the Scopus_literature_extraction_v2 folder.
Execute elsevier_api_search_issn_v2.py.

## Initial topic modeling and outlier detection:

For the initial round of topic modeling and outlier detection:

Run Top2Vec_Initial_modeling+outlier_detection.py.
This step involves conducting first-round topic modeling to detect and remove outliers (topics unrelated to healthcare) using the GPT-3 API.

## Topic remodeling after removing outliers:

After the outliers have been removed:

Execute Top2Vec_remodeling_after_outlier_detection_visualization.py to conduct topic remodeling.
This process includes defining the optimal number of topics using a dendrogram and generating a bubble chart visualization.

## Top modeling results visualizations:

The Jupyter notebook Top_modeling_results_visualizations.ipynb is designed for the visualization of the modeling results. It includes:

Word Clouds, Area Chart, Stacked Area Chart, Violin Charts
