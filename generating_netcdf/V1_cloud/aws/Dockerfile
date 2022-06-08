FROM python:3.10.4

# set a directory for the app
WORKDIR /usr/src/generating_netcdf

# copy all the files to the container
# COPY . .

# copy all but data files
RUN mkdir ./configs
COPY ./configs ./configs
RUN mkdir ./ecco_cloud_utils
COPY ./ecco_cloud_utils ./ecco_cloud_utils
RUN mkdir ./ecco_grids
COPY ./ecco_grids ./ecco_grids
RUN mkdir ./ecco_v4_py
COPY ./ecco_v4_py ./ecco_v4_py
RUN mkdir ./metadata
COPY ./metadata ./metadata
RUN mkdir ./mapping_factors
COPY ./mapping_factors ./mapping_factors

COPY ./Dockerfile .
COPY ./eccov4r4_gen_for_podaac_cloud.py .
COPY ./gen_netcdf_utils.py .
COPY ./mapping_factors.py .
COPY ./master_script.py .
COPY ./requirements.txt .
COPY ./upload_S3.py .
COPY ./aws-login.linux.amd64 .

# RUN mkdir ./temp_output

# copy data files
# RUN mkdir ./diags_all
# COPY ./diags_all ./diags_all

# install dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r ./requirements.txt

# install valgrind for mem testing
# RUN apt-get update
# RUN apt-get upgrade -y
# RUN apt-get install g++ valgrind -y

# run test command
CMD ["chmod", "+x", "aws-login.linux.amd64"]