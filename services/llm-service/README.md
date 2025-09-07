# InvestIQ LLM Service

基于llama.cpp的LLM推理服务，提供OpenAI兼容的HTTP API。

## 🎯 功能

- **高性能推理**: 基于llama.cpp，支持GPU加速
- **OpenAI兼容**: 支持`/v1/chat/completions`接口
- **模型支持**: Qwen3, Llama, ChatGLM等GGUF格式模型
- **Jetson优化**: 针对Jetson AGX Orin优化

## 🚀 使用

### 模型准备
```bash
# 下载Qwen3-8B-Instruct GGUF模型
wget -O models/qwen3-8b-instruct.gguf \
  https://huggingface.co/Qwen/Qwen3-8B-Instruct-GGUF/resolve/main/qwen3-8b-instruct.gguf
```

### 启动服务
```bash
docker build -t investiq-llm .
docker run -p 8001:8001 -v ./models:/models investiq-llm
```

## 📡 API端点

- `POST /v1/inference` - InvestIQ格式推理
- `POST /v1/chat/completions` - OpenAI格式聊天
- `GET /v1/models` - 列出可用模型
- `GET /health` - 健康检查

## ⚙️ 配置

- `MODEL_PATH`: 模型文件路径
- `LLAMA_CPP_PORT`: llama.cpp端口 (8080)
- `PYTHON_SERVICE_PORT`: HTTP API端口 (8001)