# InvestIQ Common Library

InvestIQ平台的共享库，包含：

## 📦 组件

### Models
- 数据模型定义（Pydantic）
- 请求/响应schemas
- 数据库模型接口

### Clients  
- HTTP客户端封装
- 服务间通信接口
- 重试和错误处理

### Utils
- 通用工具函数
- 日志配置
- 验证函数

## 🚀 使用

```python
from investiq_common.models import EquityModel
from investiq_common.clients import LLMClient
from investiq_common.utils import setup_logging
```