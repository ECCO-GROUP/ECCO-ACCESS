FROM python:3.10.4

# set a directory for the app
# WORKDIR /usr/src/test_lambda
ARG FUNCTION_DIR="/home/app/"

WORKDIR ${FUNCTION_DIR}

# copy code file
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
COPY ./requirements.txt .
COPY ./app.py .
# COPY . ${FUNCTION_DIR}
# COPY ./app.py .
# COPY ./requirements.txt .

# install dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r ${FUNCTION_DIR}requirements.txt --target ${FUNCTION_DIR}

RUN python3 -m pip install awslambdaric --target ${FUNCTION_DIR}



# COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Basic command
# ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]
# CMD [ "app.handler" ]

# ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
COPY entry.sh /
# RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
RUN chmod 755 /entry.sh
#ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]
ENTRYPOINT [ "/entry.sh" ]
CMD [ "app.handler" ]