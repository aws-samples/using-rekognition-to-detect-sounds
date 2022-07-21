FROM public.ecr.aws/lambda/python:3.8

ENV NUMBA_CACHE_DIR=/tmp/NUMBA_CACHE_DIR/

ENV MPLCONFIGDIR=/tmp/MPLCONFIGDIR/

RUN mkdir -m 777 /tmp/NUMBA_CACHE_DIR /tmp/MPLCONFIGDIR

RUN rm -rfv /var/cache/yum
RUN yum clean all

RUN yum install -y libsndfile
RUN rm -rf /var/cache/yum

COPY *.py requirements.txt ./

RUN python3.8 -m pip install -r requirements.txt -t .

CMD ["app.lambda_handler"]