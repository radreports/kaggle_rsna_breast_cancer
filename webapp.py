from flask import Flask, request, jsonify
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from torchvision import transforms
import torch
from torch.nn import functional as F

app = Flask(__name__)

# Function to load the TensorRT model
def load_model(model_path):
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(model_path, 'rb') as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

# A function to perform inference using TensorRT
def infer(engine, image):
    context = engine.create_execution_context()
    input_shape = engine.get_binding_shape(0)  # Assuming index 0 for input
    dtype = trt.nptype(engine.get_binding_dtype(0))
    output_shape = engine.get_binding_shape(1)  # Assuming index 1 for output
    d_input = cuda.mem_alloc(1 * image.nbytes)
    d_output = cuda.mem_alloc(1 * np.product(output_shape) * image.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, image, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    output = np.empty(output_shape, dtype=dtype)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    return output

# Preprocess image based on submit.py's patterns
def preprocess_image(image):
    input_size = [2048, 1024]
    input_h, input_w = input_size
    ori_h, ori_w = image.shape[:2]
    ratio = min(input_h / ori_h, input_w / ori_w)
    resized_image = cv2.resize(image, (int(ori_w * ratio), int(ori_h * ratio)))
    return resized_image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    processed_img = preprocess_image(img)
    engine = load_model('your_model_path.trt')
    output = infer(engine, processed_img)
    # Add post-processing here as needed
    save_path = "path_to_save_directory"
    cv2.imwrite(os.path.join(save_path, 'result.png'), output)
    return jsonify({'message': 'Prediction complete'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
