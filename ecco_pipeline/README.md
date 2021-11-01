# ecco-preprocessing

## Preprocessing Pipeline 
The general workflow for preprocessing a dataset to be added to the ECCO model is: [1] harvest the dataset, [2] run preprocessing on each file, and [3] aggregate data into yearly files. Currently the pipeline works on a local machine, although in the future it will be able to run on the cloud using AWS. 

## Logging/Tracking Metadata
The pipeline utilizes a Solr online database for logging. The Solr database is updated at each step of the pipeline with the following information: metadata about a dataset as a whole, metadata about specific files, and metadata about the pipeline process itself. This metadata is used to automatically control the flow of the pipeline. Entries in the database fall into one of seven “types”: Grid, Dataset, Field, Harvested, Transformation, Aggregation and Descendants. 

## Pipeline Structure 
Dataset specific harvesting, transformation, and aggregation configuration files are used to provide the necessary information to run the pipeline from generalized code. The run_pipeline.py file provides the user with options for how to run the pipeline, what steps of the pipeline to run and what datasets to send through the pipeline. 

## Cloud Development 
The current cloud computing functionality of this pipeline has not been tested or verified. As a result, the pipeline can currently only be run via the run_pipeline.py tool script on a local machine.

## More Information
More detailed information can be found at the following wiki pages:
  - [Documentation](https://github.com/ECCO-GROUP/ECCO-ACCESS/wiki/Documentation)
  - [Guide to the pipeline](https://github.com/ECCO-GROUP/ECCO-ACCESS/wiki/Preprocessing-Pipeline-Guide)
  - [Guide to the pipeline harvesters](https://github.com/ECCO-GROUP/ECCO-ACCESS/wiki/Preprocessing-Harvester-Guide)
