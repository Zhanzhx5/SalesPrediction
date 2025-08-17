import torch
import sys

import pytorch_lightning as pl  # 先正常导入旧版
# 将 "lightning.pytorch" 指向已导入的旧版 pytorch_lightning
sys.modules["lightning.pytorch"] = pl  
sys.modules["lightning.pytorch.callbacks"] = pl.callbacks
sys.modules["lightning.pytorch.loggers"] = pl.loggers
# 显式替换 LightningModule 和 Trainer（关键！）
from pytorch_lightning import LightningModule, Trainer
sys.modules["lightning.pytorch.core"] = pl.core
sys.modules["lightning.pytorch.trainer"] = pl.trainer
# === 1.b 统一 LightningModule 类，解决旧版/新版不一致的问题 ===
try:
    import lightning.pytorch as _lp
    # 让 pl.LightningModule 与 lightning.pytorch.LightningModule 指向同一对象，避免 Trainer 类型校验失败
    if _lp.LightningModule is not pl.LightningModule:
        pl.LightningModule = _lp.LightningModule
        # patch 内部引用
        if hasattr(pl, 'core') and hasattr(pl.core, 'module'):
            pl.core.module.LightningModule = _lp.LightningModule
except ImportError:
    pass
# === 2. 正常导入其他依赖（包括 TemporalFusionTransformer）===

from pytorch_forecasting.models import TemporalFusionTransformer
import pytorch_forecasting as ptf

print("--- 诊断脚本开始 ---")

# 1. 再次确认我们当前环境中的版本
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Lightning Version: {pl.__version__}")
print(f"PyTorch Forecasting Version: {ptf.__version__}")

# 2. 检查核心类的继承关系
# 这是决定性的测试：我们直接检查TFT类本身，是否被Python环境识别为LightningModule的子类
is_subclass = issubclass(TemporalFusionTransformer, pl.LightningModule)

print(f"\nTemporalFusionTransformer的类型是: {TemporalFusionTransformer}")
print(f"LightningModule的类型是: {pl.LightningModule}")
print(f"\n核心诊断：TemporalFusionTransformer 是否是 LightningModule 的子类? -> {is_subclass}")

if is_subclass:
    print("\n✅ 诊断结果：环境兼容性正常。问题非常奇怪，可能出在您主脚本的某个地方。")
else:
    print("\n❌ 诊断结果：环境存在根本性的不兼容！这证实了两个库未能正确连接。")

print("--- 诊断脚本结束 ---")
