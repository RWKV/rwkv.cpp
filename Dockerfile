FROM python:3.8 as builder

RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list
RUN apt-get update && apt-get install -y g++ cmake

ADD . /work

RUN cd /work && cmake . && cmake --build . --config Release


FROM python:3.8

COPY --from=builder /work/librwkv.so /librwkv.so

ADD rwkv/rwkv_cpp_model.py /rwkv/rwkv_cpp_model.py
ADD rwkv/rwkv_cpp_shared_library.py /rwkv/rwkv_cpp_shared_library.py
ADD rwkv/rwkv_tokenizer.py /rwkv/rwkv_tokenizer.py
ADD rwkv/sampling.py /rwkv/sampling.py
ADD rwkv/20B_tokenizer.json /rwkv/20B_tokenizer.json
ADD rwkv/rwkv_vocab_v20230424.txt /rwkv/rwkv_vocab_v20230424.txt
ADD rwkv/api.py /rwkv/api.py

RUN pip3 install uvicorn numpy tokenizers fastapi==0.92.0 sse_starlette -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-cache-dir
RUN  pip3 install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /rwkv

CMD ["python", "api.py"]

