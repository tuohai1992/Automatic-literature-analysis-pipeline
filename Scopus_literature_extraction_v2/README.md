<!--
 Copyright 2020 Ujjwal Sharma. All rights reserved.
 Use of this source code is governed by a BSD-style
 license that can be found in the LICENSE file.
-->

# Search SCOPUS for your relevant literature. Easily.Quickly. #

## Credits ##

This script was originally developed by Stevan Rudinac and contains parts of his original work.

# Usage     

This script allows you to query SCOPUS for relevant journal articles (not conference proceedings) and dump the results to a CSV file which can be later examined.

## Setup ##

You will require Python 3.6 or higher to run this script. Additionally, you will also need the following libraries.

1. `pandas`
2. `pyyaml`
3. `elsapy`

You can install these libraries using the python package manager `pip` with the instruction:

```python
pip install elsapy pandas pyyaml
```

The usage of Anaconda as your Python distribution is strongly recommended.

## Execution ##

Once you have the libraries setup, please edit the supplied `config.yaml` file to indicate your search parameters. 

1. `api_key` - Your API key, issued by Elsevier.
2. `HPC keywords` - A list (starts and ends with [] brackets) of HPC related terms you want to query SCOPus for.
3. `Healthcare keywords` - A list (starts and ends with [] brackets) of Healthcare related terms you want to query SCOPus for.
4. `publication_list_path`(optional) - Path for a CSV file containing list of publications to search for. Please ensure your file looks similar to the supplied file.

Once this is done, please run:

```python
python elsevier_api_search_issn_v2.py
```

Your results will be dumped to the file `search_results.csv` located in the `save` directory.