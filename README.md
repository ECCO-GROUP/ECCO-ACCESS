## Description

`ecco_access` is a Python package with utilities for accessing output from the Estimating the Circulation and Climate of the Ocean (ECCO) state estimate. These tools enable the user to query and download ECCO output to their local machine, or directly access the ECCO output when working in the AWS Cloud.

Currently access to ECCO version 4, releases 4 and 5 are supported. Additional versions will be made available as they are released through the Physical Oceanography Distributed Active Archive Center (PO.DAAC), or made available on the ECCO S3 bucket in the AWS Cloud (s3://ecco-model-granules).

There is some overlap in functionality with the [earthaccess](https://earthaccess.readthedocs.io/en/stable/) Python package for accessing NASA Earth science data, and datasets on PO.DAAC/NASA Earthdata can be accessed with both packages. `ecco_access` has some features that enable:

- querying of the variables in ECCO datasets
- spatial subsetting of ECCO output in the 13-tile Lat-Lon-Cap (LLC) native grid
- access on the AWS Cloud to ECCO release(s) not yet available through NASA Earthdata

## Documentation

Documentation on how to use `ecco_access`, including installation/setup requirements and instructions, can be found on ReadtheDocs: https://ecco-access.readthedocs.io

## Support

If you have questions or notice a bug, please reach out to andrewdelman@ucla.edu.

## Contributors

If you have Python code that would be helpful for accessing ECCO data or adding new features, we'd love to include your contributions! Feel free to [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of this repository and after including your changes, submit a pull request to the `main` branch of the original repository. Or contact andrewdelman@ucla.edu to discuss.
