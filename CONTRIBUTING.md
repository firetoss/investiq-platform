# InvestIQ Platform 贡献指南

## 🤝 欢迎贡献

感谢您对InvestIQ Platform的关注！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 错误报告和修复
- ✨ 新功能开发
- 📚 文档改进
- 🔧 性能优化
- 🧪 测试用例添加

## 📋 贡献流程

### 1. 准备工作

#### Fork项目
```bash
# 1. 在GitHub上Fork项目到您的账户
# 2. 克隆您的Fork到本地
git clone https://github.com/YOUR_USERNAME/investiq-platform.git
cd investiq-platform

# 3. 添加上游仓库
git remote add upstream https://github.com/firetoss/investiq-platform.git
```

#### 设置开发环境
```bash
# 安装uv包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync --all-extras

# 安装开发工具
uv run pre-commit install

# 验证环境
uv run python --version
uv run pytest --version
```

### 2. 开发流程

#### 创建功能分支
```bash
# 同步最新代码
git fetch upstream
git checkout main
git merge upstream/main

# 创建功能分支
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

#### 开发和测试
```bash
# 启动开发环境
docker-compose -f docker-compose.jetson.yml up -d postgres redis minio

# 启动主应用 (开发模式)
uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# 运行测试
uv run pytest backend/tests/ -v

# 代码格式化
uv run black backend/
uv run isort backend/

# 类型检查
uv run mypy backend/
```

### 3. 提交代码

#### 提交规范
使用[Conventional Commits](https://www.conventionalcommits.org/)格式：

```bash
# 功能开发
git commit -m "feat(ai): add RoBERTa sentiment analysis model"

# 错误修复
git commit -m "fix(docker): resolve DLA core allocation issue"

# 文档更新
git commit -m "docs(readme): update deployment instructions"

# 性能优化
git commit -m "perf(jetson): optimize GPU memory usage"

# 代码重构
git commit -m "refactor(services): migrate to microservices architecture"
```

#### 提交类型
- `feat`: 新功能
- `fix`: 错误修复
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 4. 创建Pull Request

#### PR准备
```bash
# 确保代码质量
uv run pre-commit run --all-files

# 运行完整测试套件
uv run pytest backend/tests/ --cov=backend --cov-report=html

# 推送分支
git push origin feature/your-feature-name
```

#### PR模板
在GitHub上创建PR时，请包含以下信息：

```markdown
## 📝 变更描述
简要描述您的更改内容和原因。

## 🔧 变更类型
- [ ] 错误修复
- [ ] 新功能
- [ ] 性能优化
- [ ] 文档更新
- [ ] 代码重构

## 🧪 测试
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 在Jetson设备上测试通过
- [ ] 性能基准测试通过

## 📋 检查清单
- [ ] 代码遵循项目规范
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 通过了所有CI检查

## 📸 截图/演示
如果适用，请添加截图或演示视频。

## 🔗 相关Issue
Closes #(issue number)
```

## 📚 开发指南

### 代码规范

#### Python代码规范
- **格式化**: 使用black (line-length=88)
- **导入排序**: 使用isort (profile=black)
- **类型注解**: 所有公共函数必须有类型注解
- **文档字符串**: 使用Google风格的docstring
- **命名规范**: 
  - 变量和函数: snake_case
  - 类名: PascalCase
  - 常量: UPPER_CASE

#### 示例代码
```python
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AIServiceClient:
    """AI服务客户端
    
    提供与AI服务的HTTP通信接口。
    
    Args:
        base_url: 服务基础URL
        timeout: 请求超时时间
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.timeout = timeout
    
    async def predict(
        self, 
        data: List[float], 
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行预测
        
        Args:
            data: 输入数据
            model_name: 模型名称
            
        Returns:
            预测结果字典
            
        Raises:
            ValueError: 当输入数据无效时
            httpx.HTTPError: 当HTTP请求失败时
        """
        if not data:
            raise ValueError("输入数据不能为空")
        
        # 实现逻辑...
        return {"predictions": []}
```

### 测试规范

#### 单元测试
```python
import pytest
from unittest.mock import AsyncMock, patch

from backend.app.services.ai_clients import LLMServiceClient


class TestLLMServiceClient:
    """LLM服务客户端测试"""
    
    @pytest.fixture
    def client(self):
        return LLMServiceClient("http://test-service:8001")
    
    @pytest.mark.asyncio
    async def test_inference_success(self, client):
        """测试推理成功场景"""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.json.return_value = {
                "choices": [{"text": "测试回答"}]
            }
            
            result = await client.inference("测试提示")
            
            assert result["text"] == "测试回答"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_inference_failure(self, client):
        """测试推理失败场景"""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = Exception("网络错误")
            
            result = await client.inference("测试提示")
            
            assert "error" in result
```

#### 集成测试
```python
import pytest
import httpx

@pytest.mark.integration
class TestAPIIntegration:
    """API集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self):
        """测试完整分析流水线"""
        async with httpx.AsyncClient() as client:
            # 测试政策分析
            response = await client.post(
                "http://localhost:8000/api/v1/analysis/policy",
                json={"policy_text": "测试政策内容"}
            )
            assert response.status_code == 200
            
            # 测试情感分析
            response = await client.post(
                "http://localhost:8000/api/v1/ai/sentiment/analyze",
                json={"texts": ["测试新闻内容"]}
            )
            assert response.status_code == 200
```

### 性能测试

#### 基准测试
```python
import time
import asyncio
from typing import List

async def benchmark_llm_inference(prompts: List[str]) -> Dict[str, float]:
    """LLM推理性能基准测试"""
    start_time = time.time()
    
    tasks = []
    for prompt in prompts:
        task = ai_clients.llm_client.inference(prompt)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    throughput = len(prompts) / total_time
    
    return {
        "total_time": total_time,
        "throughput": throughput,
        "avg_latency": total_time / len(prompts)
    }
```

## 🔧 开发工具

### 推荐IDE配置

#### VSCode配置 (.vscode/settings.json)
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### 调试配置

#### 本地调试
```bash
# 启动调试模式
export FASTAPI_ENV=development
export LOG_LEVEL=DEBUG

# 启动主应用
uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# 在另一个终端启动AI服务 (用于调试)
cd backend/app/services
python sentiment_server.py  # 端口8002
python timeseries_server.py  # 端口8003
python cpu_timeseries_server.py  # 端口8004
```

#### 容器调试
```bash
# 进入容器调试
docker-compose -f docker-compose.jetson.yml exec investiq-app bash

# 查看容器内部状态
docker-compose -f docker-compose.jetson.yml exec llm-service nvidia-smi
docker-compose -f docker-compose.jetson.yml exec sentiment-service python -c "import torch; print(torch.cuda.is_available())"
```

## 📊 性能优化指南

### AI模型优化

#### 1. 模型量化
```python
# 示例: 自定义模型量化
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("model-name")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### 2. 批处理优化
```python
# 优化批处理大小
OPTIMAL_BATCH_SIZES = {
    "sentiment_analysis": 32,  # DLA0优化
    "timeseries_prediction": 16,  # DLA1优化
    "llm_inference": 1,  # GPU优化
}
```

### Jetson硬件优化

#### 1. DLA优化
```python
# DLA模型部署示例
import torch

# 设置DLA设备
device = torch.device("cuda:0")  # DLA通过CUDA接口访问
model = model.to(device)
model.half()  # 使用FP16精度

# 启用DLA优化
torch.backends.cudnn.benchmark = True
```

#### 2. 内存优化
```python
# 内存管理最佳实践
import gc
import torch

def optimize_memory():
    """优化GPU内存使用"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

## 🐛 问题报告

### Bug报告模板
```markdown
**描述问题**
简要描述遇到的问题。

**复现步骤**
1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

**期望行为**
描述您期望发生的情况。

**实际行为**
描述实际发生的情况。

**环境信息**
- 设备: Jetson Orin AGX
- JetPack版本: 36.4.4
- Docker版本: 24.0.x
- 项目版本: v1.0.0

**日志信息**
```
粘贴相关的错误日志
```

**截图**
如果适用，添加截图来帮助解释问题。
```

### 功能请求模板
```markdown
**功能描述**
简要描述您希望添加的功能。

**使用场景**
描述这个功能的使用场景和价值。

**建议实现**
如果有具体的实现建议，请描述。

**替代方案**
描述您考虑过的其他解决方案。

**优先级**
- [ ] 低
- [ ] 中
- [ ] 高
- [ ] 紧急
```

## 📚 学习资源

### 相关技术文档
- [NVIDIA Jetson开发指南](https://developer.nvidia.com/jetson)
- [dustynv容器项目](https://github.com/dusty-nv/jetson-containers)
- [FastAPI文档](https://fastapi.tiangolo.com/)
- [PyTorch文档](https://pytorch.org/docs/)

### 金融AI相关
- [Qwen模型文档](https://github.com/QwenLM/Qwen)
- [PatchTST论文](https://arxiv.org/abs/2211.14730)
- [金融情感分析最佳实践](https://huggingface.co/models?pipeline_tag=text-classification&search=financial)

## 🏆 贡献者认可

### 贡献类型
我们认可以下类型的贡献：
- 💻 **代码贡献**: 新功能、错误修复、性能优化
- 📖 **文档贡献**: 文档改进、教程编写
- 🐛 **测试贡献**: 测试用例、性能基准
- 🎨 **设计贡献**: UI/UX设计、架构设计
- 💡 **想法贡献**: 功能建议、架构建议

### 贡献者列表
感谢所有为项目做出贡献的开发者！贡献者将在项目README中得到认可。

## 📞 联系方式

### 技术讨论
- **GitHub Discussions**: https://github.com/firetoss/investiq-platform/discussions
- **技术问题**: 通过GitHub Issues提交

### 代码审查
- 所有PR都需要至少一个维护者的审查
- 重大变更需要两个维护者的审查
- 自动化测试必须通过

## 🔒 安全政策

### 安全漏洞报告
如果发现安全漏洞，请：
1. **不要**在公开Issue中报告
2. 发送邮件到安全团队 (如果有)
3. 等待安全团队的响应和修复

### 安全最佳实践
- 不要在代码中硬编码密钥或密码
- 使用环境变量管理敏感配置
- 定期更新依赖包
- 遵循最小权限原则

## 📋 开发检查清单

### 提交前检查
- [ ] 代码通过所有测试
- [ ] 代码符合格式规范
- [ ] 添加了必要的文档
- [ ] 更新了相关的API文档
- [ ] 性能测试通过 (如果适用)
- [ ] 安全检查通过

### PR检查清单
- [ ] PR标题清晰描述变更
- [ ] PR描述包含变更原因和影响
- [ ] 链接了相关的Issue
- [ ] 添加了适当的标签
- [ ] 请求了合适的审查者

---

**🙏 再次感谢您的贡献！每一个贡献都让InvestIQ Platform变得更好。**
