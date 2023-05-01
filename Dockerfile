FROM ubuntu:18.04
LABEL author="Raul Murillo"
LABEL org.opencontainers.image.authors="ramuri01@ucm.es"
LABEL version="1.0"
LABEL description="A Deep Learning Framework \
for the Posit Number System."

RUN apt update
RUN yes | apt install python3 python3-pip vim
RUN python3 -m pip install pip --upgrade

# create and use user deeppns
RUN useradd -ms /bin/bash deeppns
USER deeppns

# install dependencies
RUN pip install requests numpy==1.15.2 softposit
RUN pip install numpy-posit
RUN pip install https://s3-ap-southeast-1.amazonaws.com/posit-speedgo/tensorflow_posit-1.11.0.0.0.1.dev1-cp36-cp36m-linux_x86_64.whl
# optional packages
RUN pip install scikit-learn

# sample scripts
COPY --chown=deeppns:deeppns src/ /home/deeppns/examples/

WORKDIR /home/deeppns

# add a command that when you run the container without a command, it produces something meaningful
ENV CONTAINER_ID "Deep PeNSieve"
CMD ["/usr/bin/env", "bash"]