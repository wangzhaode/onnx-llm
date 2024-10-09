![onnx-llm](resource/logo.png)

# onnx-llm
[![License](https://img.shields.io/github/license/wangzhaode/onnx-llm)](LICENSE.txt)

## Build

### Steps

1. Download `onnxruntime` release package from [here](https://github.com/microsoft/onnxruntime/releases).
2. Extract the package to `onnx-llm/third_party/onnxruntime`.
3. Compile the project.

### Example
```base
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-arm64-1.19.2.tgz
tar -xvf onnxruntime-osx-arm64-1.19.2.tgz
mv onnxruntime-osx-arm64-1.19.2 onnx-llm/third_party/onnxruntime
mkdir build && cd build
cmake ..
make -j
```

## Usage

Same as [mnn-llm](https://github.com/wangzhaode/mnn-llm)

```base
(base) ➜  build git:(main) ✗ ./cli_demo qwen2-0.5b-instruct/config.json ../resource/prompt.txt
model path is ../../llm-export/model/config.json
load tokenizer
tokenizer_type = 3
load tokenizer Done
load ../../llm-export/model/llm.onnx ... Load Module Done!
prompt file is ../resource/prompt.txt
Hello! How can I assist you today?
我是来自阿里云的超大规模语言模型，我叫通义千问。
很抱歉，作为AI助手，我无法实时获取和显示当前的天气信息。建议您查看当地的气象预报或应用中的天气查询功能来获取准确的信息。

#################################
prompt tokens num  = 36
decode tokens num  = 64
prefill time = 0.32 s
 decode time = 2.00 s
prefill speed = 112.66 tok/s
 decode speed = 32.07 tok/s
##################################
```

## Reference
- [mnn-llm](https://github.com/wangzhaode/mnn-llm)
- [onnxruntime](https://github.com/microsoft/onnxruntime)
