FROM ultralytics/yolov5:v7.0

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y build-essential make cmake
RUN apt install -y zsh

RUN pip install transformers tensorboard pandas pandas-profiling ipywidgets seaborn matplotlib scikit-learn scipy Pillow
RUN pip install yacs pyyaml Cython
RUN pip install cython_bbox 
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda110
RUN pip install fvcore sympy onnxoptimizer onnxsim
RUN pip install pydicom joblib dicomsdl python-gdcm pylibjpeg
RUN pip install pytorch-ignite exhaustive-weighted-random-sampler setproctitle
RUN pip install wandb numba tensorrt openpyxl onnxruntime-gpu
RUN pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
RUN pip install wandb
#### EXTERNAL DEPENDENCIES #####
RUN mkdir /workspace/libs
RUN mkdir /workspace/input
RUN mkdir /workspace/output
RUN mkdir /workspace/kaggle_rsna_breast_cancer
# YOLOX dependencies    
WORKDIR  /workspace/libs
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git
WORKDIR YOLOX
RUN pip install -v -e .  # or  python3 setup.py develop
RUN pip install flask

# Torch2trt
WORKDIR  /workspace/libs
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
WORKDIR torch2trt
RUN python3 setup.py install
# COPY infer.py /workspace
# COPY 2ddfad7286c2b016931ceccd1e2c7bbc.dcm /workspace
# COPY best_ensemble_convnext_small_batch2_fp32.engine /workspace
WORKDIR /workspace/
CMD ["/usr/bin/zsh"]
