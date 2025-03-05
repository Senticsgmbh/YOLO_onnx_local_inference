# YOLO ONNX Inference Outside Deepstream

This repository contains python scripts for performing ONNX inference for YOLO v8 and v10 models.

## Prerequisites

The following Python packages are required:

```bash
pip install opencv-python numpy onnxruntime matplotlib
```

## Usage

1. Place your ONNX model in the model directory
2. Put your image/video/dataset in the input directory
3. Run the script:

You can run the script using command-line arguments:

```bash
# Basic usage with required arguments
python onnx_inference_V8.py --model /path/to/model.onnx --source /path/to/data
```

## Contact

a.thura@sentics.de
