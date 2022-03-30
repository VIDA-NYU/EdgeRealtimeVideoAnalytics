FROM redislabs/redisedge:latest

RUN set -ex; \
    apt-get --allow-releaseinfo-change update && \
    apt-get update && \
    apt-get install -y --no-install-recommends python python3-pip python3-opencv ibglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 && \
    apt-get install -y python3-opencv && \
    apt-get remove -y python3-opencv && \
    python3 -m pip install --upgrade pip setuptools pipenv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/redislabs/lib/modules/python3

ADD ./requirements.txt /tmp/requirements.txt
RUN pipenv run pip install -r /tmp/requirements.txt
ADD ./redisedge.conf /usr/local/etc/redisedge.conf

ADD init.py .

RUN mkdir -p /app
ADD app /app

CMD ["/usr/local/etc/redisedge.conf"]
