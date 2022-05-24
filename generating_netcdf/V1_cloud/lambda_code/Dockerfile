FROM python:3.10.4

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

# create volume copy of necessary files
# VOLUME /usr/src/ECCO_GROUP

# ADD ~/Users/bark/Documents/ECCO_GROUP/ECCOv4-py/ecco_v4_py/ecco_utils.py /usr/src/ECCO_GROUP/
# COPY /Users/bark/Documents/ECCO_GROUP/ usr/src/ECCO_GROUP
# COPY ./ECCO_GROUP/ .
# COPY ../../../../ /usr/src/ECCO_GROUP

# run this before copying requirements for cache efficiency
# RUN pip3 install --upgrade pip

# install dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r ./requirements.txt                                                                                                                                                                                                
# RUN python3 -m pip install -r ./generating_netcdf/V1_cloud/lambda_code/requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt
# RUN python3 -m pip install geos
# RUN python3 -m pip install proj
# RUN python3 -m pip install cartopy

# run test command
# CMD ["python3", "eccov4r4_gen_for_podaac_cloud.py"]