FROM python:3.6
LABEL MAINTAINER="shes990129@gmail.com"

RUN apt-get -y update && apt-get install -y \
	libsm6 \
	libxext6 \
	libxrender1 \
	libmagic-dev \
        libgl1-mesa-dev

COPY requirements.txt .
RUN pip3 --no-cache-dir install -r requirements.txt

WORKDIR /usr/local/src/work

COPY face ./face
COPY codes/templates ./codes/templates
COPY codes/dot_foo.py ./codes
COPY codes/foo.py ./codes
COPY codes/file_utils.py ./codes
COPY codes/entrypoint.py ./codes

EXPOSE 5000

CMD ["python3", "./codes/entrypoint.py"]
