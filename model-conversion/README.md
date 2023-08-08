# PWCNet Pytorch->ONNX conversion

This directory contains the conversion code to ONNX in `convert_onnx.py`.
Our pytorch PWCNet implementation is located in the ./pwc_pytorch directory. 

- To convert a model, follow the readme instructions in pwc_pytorch to download the models, and then execute `convert_onnx.py` which will convert the `pwcnet-my-light4dec-sepref-chairs-things-sintel` by default.
- To enable easy and fast flow visualization in onnxruntime, we also trace the color coding visualization code as a seperate model (see `flowvis.py`).
- Onnx conversion of such complex models including custom layers can still be unstable, that can often results in unexpected, slow or even failing onnx models especially when tracing certain torchscript operations.
  - We used onnx opset=17 and pytorch=1.10 and do constant folding. Using different versions might not produce the expected results 
- We implement C++ CUDA and CPU versions of the warp and correlation layers and register them as a library of onnxruntime operators, see the ../src/ort_custom_ops directory. During the onnx tracing phase, we simply insert a symbolic g.op (e.g. [correlation](https://github.com/MaxReimann/video-stream-consistency/blob/main/model-conversion/pwc_pytorch/pwcnet.py#L137)) with the registered name into the model tree.
- To quickly test models in python-onnxruntime, we provide a model runner in `modelrunner.py`, that loads the shared library of custom ORT Ops into the ORT runtime (provided they have been built)
- `test.py` provides various tests for the custom layers as well as the complete onnx model using the model runner
