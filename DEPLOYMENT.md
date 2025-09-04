# InvestIQ Platform 部署指南

## 🎯 部署概述

本文档详细介绍如何在NVIDIA Jetson Orin AGX上部署InvestIQ Platform，实现GPU+2DLA+CPU四重并行计算架构。

## 📋 前置要求

### 硬件要求
- **设备**: NVIDIA Jetson Orin AGX
- **内存**: 64GB LPDDR5
- **存储**: 2TB+ NVMe SSD
- **网络**: 千兆以太网 (用于模型下载)

### 软件要求
- **操作系统**: Ubuntu 20.04 LTS (JetPack 36.4.4)
- **容器运行时**: Docker 24.0+ + nvidia-container-runtime
- **Python**: 3.10+ (通过容器提供)

### 环境验证
```bash
# 检查JetPack版本
sudo apt show nvidia-jetpack

# 检查Docker和NVIDIA运行时
docker --version
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# 检查存储空间
df -h

# 检查内存
free -h
```

## 🚀 部署步骤

### 第一步: 环境准备

#### 1. 安装Docker和nvidia-container-runtime
```bash
# 安装Docker (如果未安装)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装nvidia-container-runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker

# 配置Docker默认运行时
sudo tee /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

sudo systemctl restart docker
```

#### 2. 克隆项目
```bash
# 克隆项目到本地
git clone https://github.com/firetoss/investiq-platform.git
cd investiq-platform

# 检查项目结构
ls -la
```

### 第二步: 模型文件准备

#### 1. 创建模型目录
```bash
# 创建模型存储目录
mkdir -p models
chmod 755 models
```

#### 2. 下载AI模型
```bash
# 下载Qwen3-8B INT8模型 (~6GB)
echo "下载Qwen3-8B INT8模型..."
wget -O models/Qwen3-8B-INT8.gguf \
  "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-INT8.gguf"

# 验证模型文件
ls -lh models/
file models/Qwen3-8B-INT8.gguf
```

**注意**: RoBERTa和PatchTST模型会在服务启动时自动从HuggingFace下载，确保网络连接正常。

### 第三步: 配置环境变量

#### 1. 复制环境配置
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
nano .env
```

#### 2. 关键配置项
```env
# Jetson硬件配置
NVIDIA_VISIBLE_DEVICES=0
DLA_CORES=2
CPU_CORES=12

# AI服务URL
LLM_SERVICE_URL=http://llm-service:8001
SENTIMENT_SERVICE_URL=http://sentiment-service:8002
TIMESERIES_SERVICE_URL=http://timeseries-service:8003
CPU_TIMESERIES_SERVICE_URL=http://cpu-timeseries:8004

# 数据库配置
DATABASE_URL=postgresql://investiq:investiq123@postgres:5432/investiq
REDIS_URL=redis://redis:6379/0

# 日志级别
LOG_LEVEL=INFO
FASTAPI_ENV=production
```

### 第四步: 启动服务

#### 1. 拉取基础镜像
```bash
# 拉取dustynv优化镜像
docker pull dustynv/llama.cpp:r36.4.4
docker pull dustynv/pytorch:2.1-r36.4.4
docker pull dustynv/tensorrt:8.6-r36.4.4

# 验证镜像
docker images | grep dustynv
```

#### 2. 构建和启动服务
```bash
# 构建主应用镜像
docker-compose -f docker-compose.jetson.yml build investiq-app

# 启动所有服务
docker-compose -f docker-compose.jetson.yml up -d

# 查看服务状态
docker-compose -f docker-compose.jetson.yml ps
```

#### 3. 查看启动日志
```bash
# 查看主应用日志
docker-compose -f docker-compose.jetson.yml logs -f investiq-app

# 查看AI服务日志
docker-compose -f docker-compose.jetson.yml logs -f llm-service
docker-compose -f docker-compose.jetson.yml logs -f sentiment-service
docker-compose -f docker-compose.jetson.yml logs -f timeseries-service
```

### 第五步: 验证部署

#### 1. 健康检查
```bash
# 检查主应用
curl -f http://localhost:8000/health

# 检查AI服务
curl -f http://localhost:8001/health  # LLM服务
curl -f http://localhost:8002/health  # 情感分析服务
curl -f http://localhost:8003/health  # 时序预测服务
curl -f http://localhost:8004/health  # CPU时序服务

# 检查数据库连接
curl -f http://localhost:8000/api/v1/system/health
```

#### 2. 功能测试
```bash
# 测试LLM推理
curl -X POST http://localhost:8000/api/v1/ai/llm/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "分析当前A股市场的投资机会"}'

# 测试情感分析
curl -X POST http://localhost:8000/api/v1/ai/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["市场表现良好，投资者信心增强"]}'

# 测试时序预测
curl -X POST http://localhost:8000/api/v1/ai/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{"data": [100, 102, 98, 105, 110], "horizon": 10}'
```

#### 3. 性能监控
```bash
# 查看性能仪表板
curl http://localhost:8000/api/v1/performance/dashboard

# 查看硬件利用率
curl http://localhost:8000/api/v1/performance/hardware/utilization

# 查看模型性能
curl http://localhost:8000/api/v1/performance/models/performance
```

## 📊 性能调优

### Jetson性能优化

#### 1. 功耗模式设置
```bash
# 设置最大性能模式
sudo nvpmodel -m 0

# 设置最大CPU频率
sudo jetson_clocks

# 验证设置
sudo nvpmodel -q
```

#### 2. 内存优化
```bash
# 增加swap空间 (如果需要)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久启用
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 3. 存储优化
```bash
# 启用NVMe优化
echo 'ACTION=="add|change", KERNEL=="nvme[0-9]*", ATTR{queue/scheduler}="none"' | sudo tee /etc/udev/rules.d/60-nvme-scheduler.rules

# 重启udev
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 容器优化

#### 1. Docker配置优化
```bash
# 编辑Docker配置
sudo nano /etc/docker/daemon.json
```

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    }
}
```

#### 2. 服务资源限制调整
根据实际性能表现，可以调整docker-compose.jetson.yml中的资源限制：

```yaml
# 示例: 调整LLM服务内存限制
llm-service:
  deploy:
    resources:
      limits:
        memory: 16G  # 根据实际使用调整
      reservations:
        memory: 12G
```

## 🔧 故障排除

### 常见问题

#### 1. 模型下载失败
```bash
# 检查网络连接
ping huggingface.co

# 手动下载模型
wget --continue -O models/Qwen3-8B-INT8.gguf \
  "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-INT8.gguf"
```

#### 2. GPU不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查容器GPU访问
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# 重启nvidia-container-runtime
sudo systemctl restart nvidia-container-runtime
sudo systemctl restart docker
```

#### 3. DLA不可用
```bash
# 检查DLA状态
cat /proc/device-tree/model

# 检查DLA驱动
ls /dev/nvhost-*

# 重启相关服务
sudo systemctl restart nvargus-daemon
```

#### 4. 内存不足
```bash
# 检查内存使用
free -h
docker stats

# 清理Docker缓存
docker system prune -a

# 重启服务
docker-compose -f docker-compose.jetson.yml restart
```

### 日志分析

#### 1. 查看服务日志
```bash
# 主应用日志
docker-compose -f docker-compose.jetson.yml logs investiq-app

# AI服务日志
docker-compose -f docker-compose.jetson.yml logs llm-service
docker-compose -f docker-compose.jetson.yml logs sentiment-service
docker-compose -f docker-compose.jetson.yml logs timeseries-service

# 实时日志
docker-compose -f docker-compose.jetson.yml logs -f --tail=100
```

#### 2. 性能监控
```bash
# 查看硬件使用情况
sudo jtop

# 查看GPU使用情况
nvidia-smi -l 1

# 查看容器资源使用
docker stats --no-stream
```

## 🔄 维护操作

### 日常维护

#### 1. 更新服务
```bash
# 拉取最新镜像
docker-compose -f docker-compose.jetson.yml pull

# 重启服务
docker-compose -f docker-compose.jetson.yml up -d
```

#### 2. 备份数据
```bash
# 备份数据库
docker exec investiq-postgres pg_dump -U investiq investiq > backup_$(date +%Y%m%d).sql

# 备份配置
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.jetson.yml
```

#### 3. 清理空间
```bash
# 清理Docker缓存
docker system prune -a

# 清理日志文件
sudo find /var/lib/docker/containers/ -name "*.log" -exec truncate -s 0 {} \;

# 清理临时文件
rm -rf tmp/* logs/*.log
```

### 性能调优

#### 1. 监控关键指标
- **GPU利用率**: 目标85-95%
- **DLA利用率**: 目标80-90%
- **CPU利用率**: 目标60-80%
- **内存使用**: 目标<90%
- **推理延迟**: LLM<2s, 情感分析<100ms, 时序预测<500ms

#### 2. 调优参数
根据监控数据调整以下参数：
- **批处理大小**: 在docker-compose.jetson.yml中调整BATCH_SIZE
- **GPU层数**: 调整LLAMA_N_GPU_LAYERS (默认35)
- **线程数**: 调整OMP_NUM_THREADS (默认12)
- **内存限制**: 调整各服务的memory limits

## 🔒 安全配置

### 1. 网络安全
```bash
# 配置防火墙 (仅开放必要端口)
sudo ufw enable
sudo ufw allow 8000/tcp  # 主应用
sudo ufw allow 22/tcp    # SSH
sudo ufw deny 8001:8004/tcp  # 内部AI服务端口
```

### 2. 容器安全
```yaml
# 在docker-compose.jetson.yml中启用安全配置
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

### 3. 数据安全
```bash
# 设置数据目录权限
sudo chown -R 1000:1000 data/
sudo chmod -R 750 data/

# 加密敏感配置
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)" >> .env
```

## 📈 监控和告警

### 1. 访问监控界面
- **Grafana**: http://localhost:3000 (admin/investiq123)
- **Prometheus**: http://localhost:9090
- **API监控**: http://localhost:8000/api/v1/performance/dashboard

### 2. 关键监控指标
- **硬件利用率**: GPU/DLA/CPU使用率
- **服务健康**: 各AI服务响应状态
- **推理性能**: 延迟和吞吐量
- **资源使用**: 内存、存储、网络

### 3. 告警配置
在Grafana中配置以下告警：
- GPU利用率 > 95% (持续5分钟)
- 内存使用率 > 90% (持续3分钟)
- 服务响应时间 > 5秒 (持续1分钟)
- 磁盘使用率 > 85%

## 🔄 升级和回滚

### 升级流程
```bash
# 1. 备份当前配置
cp docker-compose.jetson.yml docker-compose.jetson.yml.backup

# 2. 拉取最新代码
git pull origin main

# 3. 重新构建镜像
docker-compose -f docker-compose.jetson.yml build

# 4. 滚动更新服务
docker-compose -f docker-compose.jetson.yml up -d
```

### 回滚流程
```bash
# 1. 停止服务
docker-compose -f docker-compose.jetson.yml down

# 2. 恢复配置
cp docker-compose.jetson.yml.backup docker-compose.jetson.yml

# 3. 重新启动
docker-compose -f docker-compose.jetson.yml up -d
```

## 📞 技术支持

### 问题报告
如遇到部署问题，请提供以下信息：
1. Jetson设备型号和JetPack版本
2. 错误日志 (`docker-compose logs`)
3. 硬件状态 (`nvidia-smi`, `jtop`)
4. 系统资源 (`free -h`, `df -h`)

### 联系方式
- **GitHub Issues**: https://github.com/firetoss/investiq-platform/issues
- **技术讨论**: https://github.com/firetoss/investiq-platform/discussions

---

**🚀 部署成功后，您将拥有一个充分利用Jetson硬件潜力的智能投资分析平台！**
