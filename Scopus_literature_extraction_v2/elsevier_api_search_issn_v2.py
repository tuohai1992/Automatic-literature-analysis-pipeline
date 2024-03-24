# Copyright 2021 Ujjwal Sharma and Stevan Rudinac. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os

import yaml

import pandas as pd
from elsapy.elsclient import ElsClient
from elsapy.elsdoc import AbsDoc, FullDoc
from elsapy.elsprofile import ElsAffil, ElsAuthor
from elsapy.elssearch import ElsSearch
from pybliometrics.scopus import ScopusSearch
import pdb


def load_config(path):
    """Loads a YAML configuration file from a supplied address.
    Arguments:
        config {path} -- Path for configuration file.
    """

    try:
        with open(path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.SafeLoader)

        return config

    except FileNotFoundError:
        print("Missing YAML configuration at '%s'" % (path))
        exit(0)


# Load configuration.
config = load_config("./config.yaml")

client = ElsClient(config["api_key"])

HPC_keywords = 'OR'.join('"{0}"'.format(x) for x in config["HPC keywords"])
Healthcare_keywords = 'OR'.join('"{0}"'.format(x) for x in config["Healthcare keywords"])

df = pd.read_csv(config["publication_list_path"], header=None)
journals = df[2].tolist()
journal_names = df[1].tolist()

results_df = pd.DataFrame()


print(f"Searching keyword : '{HPC_keywords}' ")
# pdb.set_trace()

# query = 'TITLE-ABS-KEY (("high performance computing" OR "high-performance computing" OR "high performance computer" OR "supercomputer*" OR "supercomputing")AND ("healthcare" OR "health" OR "clinic*" OR "Medicine" OR  "disease*" OR "Medication" OR "pharmaceutical" OR "Medical" OR "patient*"OR "diagnosis" OR "diagnostic" OR "drug" OR "treatment" OR"surgery" OR "genomics" OR "health care" OR "genes" OR "genetics" OR "biomedical" OR "pathology" OR  "public health"OR "neuroimaging " OR  "epidemiology"OR "genome"  OR "therapy"))'  # any query that works in the Advanced Search on scopus.com
query = 'TITLE-ABS-KEY (('+ HPC_keywords + ')AND('+ Healthcare_keywords + '))'
# pdb.set_trace()

try:

    doc_srch = ScopusSearch(query)

    results_df = pd.DataFrame(doc_srch.results)
    # pdb.set_trace()

except BaseException as e:
    print("Error on_data: %s" % str(e))

    with open(os.path.join(config["save_dir"], "error_log_issn.txt"), "a") as f:
        f.write("Error on_data: %s\n" % str(e))

finally:
    results_df.to_csv("./save/search_results.csv")
