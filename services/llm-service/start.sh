#!/bin/bash
set -e

# InvestIQ LLM Service启动脚本
# 同时启动llama.cpp服务器和Python HTTP包装器

MODE=${1:-prod}

echo "Starting InvestIQ LLM Service in $MODE mode"

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "Please mount the model file or update MODEL_PATH environment variable"
fi

# 启动llama.cpp服务器（后台）
echo "Starting llama.cpp server on port $LLAMA_CPP_PORT..."
llama-server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$LLAMA_CPP_PORT" \
    --n-gpu-layers 35 \
    --ctx-size 8192 \
    --threads 8 \
    --mlock \
    --verbose &

LLAMA_PID=$!

# 等待llama.cpp启动
echo "Waiting for llama.cpp to start..."
for i in {1..30}; do
    if curl -s http://localhost:$LLAMA_CPP_PORT/health >/dev/null; then
        echo "llama.cpp server started successfully"
        break
    fi
    echo "Attempt $i/30: Waiting for llama.cpp..."
    sleep 2
done

# 启动Python HTTP包装器
echo "Starting Python HTTP wrapper on port $PYTHON_SERVICE_PORT..."
if [ "$MODE" = "dev" ]; then
    python3 -m uvicorn app.server:app \
        --host 0.0.0.0 \
        --port "$PYTHON_SERVICE_PORT" \
        --reload
else
    python3 -m uvicorn app.server:app \
        --host 0.0.0.0 \
        --port "$PYTHON_SERVICE_PORT" \
        --workers 2
fi

# 清理：如果Python服务退出，也停止llama.cpp
kill $LLAMA_PID 2>/dev/null || true