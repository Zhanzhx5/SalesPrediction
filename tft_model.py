#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销量预测项目 - TFT模型

严格按照指导实现的Temporal Fusion Transformer模型
基于PyTorch Forecasting框架
"""

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, NaNLabelEncoder
from pytorch_forecasting.metrics import NegativeBinomialDistributionLoss, PoissonLoss
from pytorch_forecasting.data.encoders import TorchNormalizer
from baseline_model import calculate_wape, calculate_mae, calculate_rmse
from pytorch_lightning.strategies import DDPSpawnStrategy
import warnings
import random
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
def set_random_seed(seed=42):
    """设置所有随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)
    print(f"✅ 随机种子已设置为: {seed}")

# 修复numpy兼容性问题
# 为旧版本的pytorch-forecasting提供np.float兼容性
if not hasattr(np, 'float'):
    np.float = float

class TFTModel:
    """Temporal Fusion Transformer模型"""
    
    def __init__(self, 
                 prediction_length=30,
                 encoder_length=90,
                 learning_rate=0.0002,
                 hidden_size=64,
                 attention_head_size=8,
                 dropout=0.2,
                 hidden_continuous_size=32,
                 batch_size=1024,
                 max_epochs=30,
                 patience=5,
                 random_seed=42,
                 optuna_pruning_callback=None):
        """
        初始化TFT模型
        
        Args:
            prediction_length: 预测长度（天数）
            encoder_length: 编码器长度（回看天数）
            learning_rate: 学习率
            hidden_size: 隐藏层大小
            attention_head_size: 注意力头数量
            dropout: Dropout比例
            hidden_continuous_size: 连续特征隐藏层大小
            batch_size: 批次大小
            max_epochs: 最大训练轮数
            patience: 早停耐心值
        """
        self.prediction_length = prediction_length
        self.encoder_length = encoder_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_seed = random_seed
        self.optuna_pruning_callback = optuna_pruning_callback
        
        # 设置基础随机种子（不触发多进程）
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 模型和数据集
        self.model = None
        self.training_dataset = None
        self.validation_dataset = None
        self.trainer = None
        
        print("🚀 TFT模型初始化完成")
        print(f"   - 预测长度: {prediction_length}天")
        print(f"   - 编码器长度: {encoder_length}天")
        print(f"   - 批次大小: {batch_size}")
        print(f"   - 最大训练轮数: {max_epochs}")
        print(f"   - 随机种子: {random_seed}")
    
    def load_and_preprocess_data(self, file_path):
        """
        步骤1: 数据加载与基础预处理
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            预处理后的数据框
        """
        print("📊 步骤1: 数据加载与基础预处理")
        
        # 1. 加载您已经处理好的CSV文件
        print(f"   加载数据文件: {file_path}")
        df = pd.read_csv(file_path)
        
        # 2. 转换日期格式 (非常重要)
        df['dtdate'] = pd.to_datetime(df['dtdate'])
        
        # 3. 创建整数时间索引 `time_idx` (库的硬性要求)
        # 它将日期转换为从0开始的连续整数，这是TFT工作的核心。
        df['time_idx'] = (df['dtdate'] - df['dtdate'].min()).dt.days
        
        # 4. 统一转换所有分类特征的数据类型 (推荐步骤)
        # 这一步可以防止后续出现类型推断错误，并能节省内存
        all_categorical_feature_names = [
            'store_id', 'item_id', 'year', 'month', 'day', 'day_of_week', 'week_of_year',
            'is_weekend', 'is_holiday', 'skc_sgspztdesc', 'shbd', 'sgoodtype', 'sfabric',
            'sfabricdesc', 'sdeptname', 'sstyle', 'supcolorno', 'sserie', 'susedof',
            'sseason', 'scolordesc', 'scomponet4', 'syear', 'scategorydesc', 'sstoredesc',
            'sdevelopment', 'qydz', 'ywqy', 'sprovince', 'scity', 'saddress', 'szone',
            'schanneltype', 'sstoresize', 'scitylevelid', 'sscmsalelevel', 'sscmarealevel',
            'sscmtotallevel', 'is_bus', 'sleveltype', 'stimage', 'ssalelevel',
            # 新添加的10个商品属性列
            'sbartype', 'scollar', 'scottoncupmaterial', 'secseries', 'smoldcupprocess',
            'spattern', 'sshoulderpro', 'sshoulderwidth', 'ssteelwheel', 'swaistband'
        ]
        
        for col in all_categorical_feature_names:
            if col in df.columns:
                df[col] = df[col].astype(str).astype("category")
        
        # 5. 确保数值特征使用兼容的数据类型 (解决numpy版本兼容性问题)
        numerical_features = [
            'sales', 'zd_kc', 'nsaleprice', 'nstorearea', 'number_assistant', 
            'nvipprice', 'sgprice', 'item_age', 'store_age',
            'sales_lag_1', 'sales_lag_3', 'sales_lag_7', 'sales_lag_14', 
            'sales_lag_28', 'sales_lag_364', 'rolling_mean_7', 'rolling_std_7',
            'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30', 'rolling_max_30',
            'rolling_min_30', 'last_year_rolling_mean_7', 'last_year_rolling_std_7',
            'YoY_growth_rate_7', 'days_since_last_sale'
        ]
        
        for col in numerical_features:
            if col in df.columns:
                # 使用float32而不是np.float32，确保兼容性
                df[col] = df[col].astype('float32')
        
        # 6. 确保time_idx是整数类型 (TFT库的硬性要求)
        df['time_idx'] = df['time_idx'].astype('int32')
        
        print("✅ 数据加载与基础预处理完成")
        return df
    
    def define_feature_groups(self):
        """
        步骤2: 特征最终分组定义
        
        Returns:
            特征分组字典
        """
        print("🔧 步骤2: 特征分组定义")
        
        # --- 静态特征 (Static Features) ---
        # 对于一个商品-店铺组合，这些特征是固定不变的。模型会学习它们对整个序列的长期影响。
        static_categoricals_list = [
            'store_id', 'item_id', 'skc_sgspztdesc', 'shbd', 'sgoodtype', 'sfabric',
            'sfabricdesc', 'sdeptname', 'sstyle', 'supcolorno', 'sserie', 'susedof',
            'sseason', 'scolordesc', 'scomponet4', 'syear', 'scategorydesc', 'sstoredesc',
            'sdevelopment', 'qydz', 'ywqy', 'sprovince', 'scity', 'saddress', 'szone',
            'schanneltype', 'sstoresize', 'scitylevelid', 'sscmsalelevel', 'sscmarealevel',
            'sscmtotallevel', 'is_bus', 'sleveltype', 'stimage', 'ssalelevel',
            # 添加缺失的10个商品属性列
            'sbartype', 'scollar', 'scottoncupmaterial', 'secseries', 'smoldcupprocess',
            'spattern', 'sshoulderpro', 'sshoulderwidth', 'ssteelwheel', 'swaistband'
        ]
        
        # 根据您的澄清，VIP价和吊牌价是静态的
        static_reals_list = [
            'nstorearea', 'number_assistant', 'nvipprice', 'sgprice'
        ]

        # --- 动态且未来已知的特征 (Time-Varying Known Features) ---
        # 对于未来的每一天，我们都能提前知道这些特征的值。模型会利用它们来做"计划"。
        time_varying_known_categoricals_list = [
            'year', 'month', 'day', 'day_of_week', 'week_of_year', 'is_weekend', 'is_holiday'
        ]
        time_varying_known_reals_list = [
            'time_idx', 'item_age', 'store_age' 
        ]

        # --- 动态且未来未知的特征 (Time-Varying Unknown Features) ---
        # 这些是需要预测或只能观测到的历史数据。在预测未来时，模型无法获取这些特征的未来值。
        # 我们保留所有lag和rolling特征，它们能为TFT提供强大的信号"快捷方式"，弥补回看窗口的视野局限。
        time_varying_unknown_reals_list = [
            'sales', 'zd_kc', 'nsaleprice', 'sales_lag_1', 'sales_lag_3', 'sales_lag_7',
            'sales_lag_14', 'sales_lag_28', 'sales_lag_364', 'rolling_mean_7',
            'rolling_std_7', 'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30',
            'rolling_max_30', 'rolling_min_30', 'last_year_rolling_mean_7',
            'last_year_rolling_std_7', 'YoY_growth_rate_7', 'days_since_last_sale'
        ]
        
        feature_groups = {
            'static_categoricals': static_categoricals_list,
            'static_reals': static_reals_list,
            'time_varying_known_categoricals': time_varying_known_categoricals_list,
            'time_varying_known_reals': time_varying_known_reals_list,
            'time_varying_unknown_reals': time_varying_unknown_reals_list
        }
        
        print("✅ 特征分组定义完成")
        return feature_groups
    
    def create_timeseries_dataset(self, df, feature_groups):
        """
        步骤3: 创建 TimeSeriesDataSet
        
        Args:
            df: 预处理后的数据框
            feature_groups: 特征分组字典
            
        Returns:
            训练和验证数据集
        """
        print("🔧 步骤3: 创建TimeSeriesDataSet")
        
        # 根据您的数据划分，训练集到2025年4月30日
        training_cutoff_date = pd.to_datetime("2025-04-30")
        training_cutoff_idx = df[df['dtdate'] == training_cutoff_date]['time_idx'].iloc[0]
        
        # 过滤掉不存在的特征
        for group_name, features in feature_groups.items():
            feature_groups[group_name] = [f for f in features if f in df.columns]

        # --- 创建训练数据集 ---
        self.training_dataset = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff_idx],
            time_idx="time_idx",
            target="sales",
            group_ids=["store_id", "item_id"],
            max_encoder_length=self.encoder_length,
            max_prediction_length=self.prediction_length,
            min_encoder_length=0,                   # <-- 关键修正：允许最短0天的历史
            static_categoricals=feature_groups['static_categoricals'],
            static_reals=feature_groups['static_reals'],
            time_varying_known_categoricals=feature_groups['time_varying_known_categoricals'],
            time_varying_known_reals=feature_groups['time_varying_known_reals'],
            time_varying_unknown_reals=feature_groups['time_varying_unknown_reals'],

            # 目标不做任何归一化/标准化：identity（无需逆变换）
            target_normalizer=TorchNormalizer(method="identity", center=False),
            # 保留您必需的分类编码器，以处理未见过的类别
            categorical_encoders={
                col: NaNLabelEncoder(add_nan=True) for col in (
                    feature_groups['static_categoricals'] + feature_groups['time_varying_known_categoricals']
                )
            },

            allow_missing_timesteps=True,
            add_relative_time_idx=True
        )
        
        # 创建验证数据集 (只预测5月份，用于训练时的验证)
        # 找到验证集预测所需历史数据的截止点
        val_encoder_cutoff_date = pd.to_datetime("2025-04-30")
        val_encoder_cutoff_idx = df[df['dtdate'] == val_encoder_cutoff_date]['time_idx'].iloc[0]
        
        # 限制验证集只包含5月份的预测窗口
        val_end_date = pd.to_datetime("2025-05-31")
        val_end_idx = df[df['dtdate'] == val_end_date]['time_idx'].iloc[0]
        
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, 
            df[df['time_idx'] <= val_end_idx], # 只传入到5月31日的数据
            predict=True, 
            stop_randomization=True,
            min_prediction_idx=val_encoder_cutoff_idx + 1,  # 从5月1日开始预测
            min_encoder_length=0,  # 显式设置min_encoder_length=0
            add_relative_time_idx=True
        )
        
        print(f"✅ TimeSeriesDataSet创建完成")
        print(f"   - 训练集样本数: {len(self.training_dataset)}")
        print(f"   - 验证集样本数: {len(self.validation_dataset)}")
        
        return self.training_dataset, self.validation_dataset
    
    def create_model(self):
        """
        步骤4: 模型配置与实例化
        """
        print("🔧 步骤4: 创建TFT模型")
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=1,  # Poisson 只需要 1 个参数
            loss=PoissonLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        print("✅ TFT模型创建完成")
    
    def train_model(self):
        """
        步骤5: 模型训练
        """
        print("🔧 步骤5: 训练TFT模型")
        
        # 在训练开始时设置PyTorch Lightning的随机种子（会触发多进程）
        pl.seed_everything(self.random_seed, workers=True)

        # Optional: 利用 Tensor Cores，加速 matmul
        try:
            torch.set_float32_matmul_precision('medium')  # or 'high'
        except Exception:
            pass
        
        # 创建DataLoader
        train_dataloader = self.training_dataset.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 2,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        
        # 早停回调
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=self.patience, verbose=False, mode="min"
        )
        lr_logger = pl.callbacks.LearningRateMonitor()
        
        # 模型检查点回调 - 使用PyTorch Lightning的version机制
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=None,  # 让PyTorch Lightning自动管理路径到version目录
            filename="tft_model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,  # 每个version只保留最好的一个模型
            mode="min",
            save_last=False  # 不保存最后一个epoch
        )
        
        # 1. 显式创建一个TensorBoard Logger
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger("lightning_logs", name="sales_forecasting_tft")
        
        # 2. 检查GPU并自动使用所有可用GPU
        if torch.cuda.is_available():
            accelerator_config = 'gpu'
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                # 多GPU：使用 DDPSpawnStrategy 并关闭 find_unused_parameters 以避免性能开销警告
                devices_config = "auto"  # 使用所有可用GPU
                strategy_config = DDPSpawnStrategy(find_unused_parameters=False)
                print(f"✅ 检测到 {gpu_count} 个GPU，启用多GPU训练")
            else:
                devices_config = [0]
                strategy_config = None
                print(f"🖥️ 使用单GPU训练")
        else:
            accelerator_config = 'cpu'
            devices_config = 'auto'
            strategy_config = None
            print("⚠️ 未检测到可用GPU，将使用CPU进行训练。")

        # 3. 创建训练器 (使用兼容最新版的参数)
        callbacks_list = [lr_logger, early_stop_callback, checkpoint_callback]
        if getattr(self, 'optuna_pruning_callback', None) is not None:
            callbacks_list.append(self.optuna_pruning_callback)

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator_config,
            devices=devices_config,
            strategy=strategy_config,
            gradient_clip_algorithm="norm",
            gradient_clip_val=0.1,
            callbacks=callbacks_list,
            enable_progress_bar=True,
            logger=logger,
        )
        
        # 训练模型
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        print("✅ TFT模型训练完成")
    
    def predict_and_evaluate(self, df):
        """
        步骤6: 预测、后处理与评估 (修正版)
        
        Args:
            df: 包含所有数据的完整预处理后DataFrame
            
        Returns:
            评估结果字典
        """
        print("🔮 步骤6: 预测与评估")

        # 1. 加载最佳模型
        if self.trainer is not None and hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback.best_model_path:
            # 如果已经训练过，使用训练器中的最佳模型
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            print(f"   加载已训练模型: {best_model_path}")
        else:
            # 如果只是评估，从最新的version目录加载最佳模型
            import os
            import glob
            
            lightning_logs_dir = "lightning_logs/sales_forecasting_tft"
            if os.path.exists(lightning_logs_dir):
                # 查找所有版本目录
                version_dirs = glob.glob(os.path.join(lightning_logs_dir, "version_*"))
                if version_dirs:
                    # 按版本号排序，选择最新的
                    version_dirs.sort(key=lambda x: int(x.split('version_')[-1]), reverse=True)
                    latest_version_dir = version_dirs[0]
                    checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
                    
                    if os.path.exists(checkpoint_dir):
                        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
                        if checkpoint_files:
                            # 选择最新的checkpoint文件
                            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
                            best_model_path = checkpoint_files[0]
                            print(f"   加载最新version的最佳模型: {best_model_path}")
                        else:
                            raise FileNotFoundError("未找到已训练的模型文件，请先运行训练步骤")
                    else:
                        raise FileNotFoundError("未找到version目录中的checkpoints文件夹，请先运行训练步骤")
                else:
                    raise FileNotFoundError("未找到模型版本目录，请先运行训练步骤")
            else:
                raise FileNotFoundError("未找到lightning_logs目录，请先运行训练步骤")
        
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        # 为避免在大型数据评估时出现设备不一致问题，统一在CPU上进行推理
        best_tft.eval()
        best_tft.cpu()
        print("   模型已加载到 CPU 用于推理")

        # 2. 验证数据集存在
        if self.validation_dataset is None:
            raise ValueError("validation_dataset不存在，请先运行create_timeseries_dataset方法")
        
        # 3. 验证df参数
        if df is None:
            raise ValueError("必须传入包含真实值的完整DataFrame以进行评估")
        
        required_columns = ['time_idx', 'store_id', 'item_id', 'sales']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必需的列: {missing_columns}")

        results = {}

        # 4. 预测验证集 (5月份)
        print("   预测验证集 (5月份)...")
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False, batch_size=self.batch_size * 2, num_workers=4
        )
        
        # 最终修正: 根据调试结果，使用两个变量直接解包
        val_raw_output, val_index_df = best_tft.predict(val_dataloader, mode="raw", return_index=True)

        # 最终修正: 原始输出为类字典对象，直接通过键 "prediction" 取核心预测张量
        val_param_tensor = val_raw_output["prediction"]

        if isinstance(val_param_tensor, np.ndarray):
            val_param_tensor = torch.from_numpy(val_param_tensor)
        if isinstance(val_param_tensor, torch.Tensor):
            val_param_tensor = val_param_tensor.cpu()

        # 基于泊松分布参数进行采样，得到计数型样本（目标采用 identity，无需逆变换）
        # 1. 首先，应用激活函数将原始输出(logits)转换为非负的率参数(rate)
        #    这与训练时PoissonLoss内部的操作一致
        positive_rate_tensor = torch.exp(val_param_tensor)
        
        # 2. 然后，使用转换后的正率参数进行泊松采样
        val_samples_tensor = torch.poisson(positive_rate_tensor)
        val_samples_np = val_samples_tensor.detach().cpu().numpy()

        # 非负裁剪 + 0.1 阈值置零（保证与业务规则一致）
        # 注：泊松采样理论上输出非负整数，但保留阈值以防未来切换为连续预测
        val_predictions_non_negative = np.maximum(0, val_samples_np)
        val_final_predictions = np.where(val_predictions_non_negative < 0.1, 0, val_predictions_non_negative)
        
        # 处理验证集预测结果
        val_prediction_df = pd.DataFrame()
        for i in range(self.prediction_length):
            temp_df = val_index_df.copy()
            temp_df['prediction'] = val_final_predictions[:, i]
            temp_df['time_idx'] = temp_df['time_idx'] + i
            val_prediction_df = pd.concat([val_prediction_df, temp_df])
        
        # 合并验证集真实值与预测值
        val_actual_df = df[required_columns + ['sdeptname']].copy()
        val_evaluation_df = pd.merge(
            val_actual_df, 
            val_prediction_df, 
            on=['time_idx', 'store_id', 'item_id'], 
            how='inner'
        )
        
        if len(val_evaluation_df) == 0:
            raise ValueError("验证集预测结果与真实值无法匹配")
        
        print(f"   验证集匹配到 {len(val_evaluation_df)} 条评估记录")
        
        # 补充dtdate字段用于可视化
        val_evaluation_df = val_evaluation_df.merge(
            df[['store_id', 'item_id', 'time_idx', 'dtdate']],
            on=['store_id', 'item_id', 'time_idx'],
            how='left'
        )
        
        # 按天评估验证集指标
        val_daily_actual = val_evaluation_df['sales'].values
        val_daily_pred = val_evaluation_df['prediction'].values
        val_daily_non_zero_mask = val_daily_actual > 0
        val_daily_non_zero_wape = calculate_wape(val_daily_actual[val_daily_non_zero_mask], val_daily_pred[val_daily_non_zero_mask]) if val_daily_non_zero_mask.sum() > 0 else 0
        
        # 按月评估验证集指标
        # 按商品-店铺-月份分组求和
        val_evaluation_df['year_month'] = val_evaluation_df['dtdate'].dt.to_period('M')
        val_monthly = val_evaluation_df.groupby(['store_id', 'item_id', 'year_month']).agg({
            'sales': 'sum',
            'prediction': 'sum'
        }).reset_index()
        
        val_monthly_actual = val_monthly['sales'].values
        val_monthly_pred = val_monthly['prediction'].values
        val_monthly_non_zero_mask = val_monthly_actual > 0
        val_monthly_non_zero_wape = calculate_wape(val_monthly_actual[val_monthly_non_zero_mask], val_monthly_pred[val_monthly_non_zero_mask]) if val_monthly_non_zero_mask.sum() > 0 else 0
        
        results['validation'] = {
            # 按天指标
            'daily_wape': calculate_wape(val_daily_actual, val_daily_pred),
            'daily_mae': calculate_mae(val_daily_actual, val_daily_pred),
            'daily_rmse': calculate_rmse(val_daily_actual, val_daily_pred),
            'daily_total_actual': np.sum(val_daily_actual),
            'daily_total_predicted': np.sum(val_daily_pred),
            'daily_prediction_bias': (np.sum(val_daily_pred) - np.sum(val_daily_actual)) / np.sum(val_daily_actual) if np.sum(val_daily_actual) > 0 else 0,
            'daily_non_zero_wape': val_daily_non_zero_wape,
            'daily_non_zero_count': val_daily_non_zero_mask.sum(),
            'daily_record_count': len(val_daily_actual),
            
            # 按月指标
            'monthly_wape': calculate_wape(val_monthly_actual, val_monthly_pred),
            'monthly_mae': calculate_mae(val_monthly_actual, val_monthly_pred),
            'monthly_rmse': calculate_rmse(val_monthly_actual, val_monthly_pred),
            'monthly_total_actual': np.sum(val_monthly_actual),
            'monthly_total_predicted': np.sum(val_monthly_pred),
            'monthly_prediction_bias': (np.sum(val_monthly_pred) - np.sum(val_monthly_actual)) / np.sum(val_monthly_actual) if np.sum(val_monthly_actual) > 0 else 0,
            'monthly_non_zero_wape': val_monthly_non_zero_wape,
            'monthly_non_zero_count': val_monthly_non_zero_mask.sum(),
            'monthly_record_count': len(val_monthly_actual),
            
            # 详细预测数据
            'detailed_predictions': val_evaluation_df  # 添加详细的预测数据（包含dtdate）
        }
        
        print(f"✅ 验证集评估完成 - 按天WAPE: {results['validation']['daily_wape']:.4f}, 按月WAPE: {results['validation']['monthly_wape']:.4f}")

        # 5. 预测测试集 (6月份)
        print("   预测测试集 (6月份)...")
        
        # 找到测试集预测所需历史数据的截止点
        test_encoder_cutoff_date = pd.to_datetime("2025-05-31")
        test_encoder_cutoff_idx = df[df['dtdate'] == test_encoder_cutoff_date]['time_idx'].iloc[0]
        
        # 从原始训练集结构出发，创建一个新的数据集，包含5月份数据
        test_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,  # 传入包含所有（直到6月30日）数据的完整DataFrame
            predict=True,
            stop_randomization=True,
            min_prediction_idx=test_encoder_cutoff_idx + 1,
            min_encoder_length=0,  # 显式设置min_encoder_length=0
            add_relative_time_idx=True
        )
        
        # 创建测试集的dataloader
        test_dataloader = test_dataset.to_dataloader(
            train=False, batch_size=self.batch_size * 2, num_workers=4
        )
        
        # 进行测试集预测
        # 使用原始输出以获取分布参数（最终修正：两个返回值）
        test_raw_output, test_index_df = best_tft.predict(test_dataloader, mode="raw", return_index=True)

        # 最终修正: 原始输出为类字典对象，直接通过键 "prediction" 取核心预测张量
        test_param_tensor = test_raw_output["prediction"]

        if isinstance(test_param_tensor, np.ndarray):
            test_param_tensor = torch.from_numpy(test_param_tensor)
        if isinstance(test_param_tensor, torch.Tensor):
            test_param_tensor = test_param_tensor.cpu()

        # 基于泊松分布参数进行采样，得到计数型样本（目标采用 identity，无需逆变换）
        # 1. 同样地，对测试集的原始输出应用激活函数
        positive_rate_tensor_test = torch.exp(test_param_tensor)
        
        # 2. 使用转换后的正率参数进行采样
        test_samples_tensor = torch.poisson(positive_rate_tensor_test)
        test_samples_np = test_samples_tensor.detach().cpu().numpy()

        # 非负裁剪 + 0.1 阈值置零
        # 注：泊松采样理论上输出非负整数，但保留阈值以防未来切换为连续预测
        test_predictions_non_negative = np.maximum(0, test_samples_np)
        test_final_predictions = np.where(test_predictions_non_negative < 0.1, 0, test_predictions_non_negative)
        
        # 处理测试集预测结果
        test_prediction_df = pd.DataFrame()
        for i in range(self.prediction_length):
            temp_df = test_index_df.copy()
            temp_df['prediction'] = test_final_predictions[:, i]
            temp_df['time_idx'] = temp_df['time_idx'] + i
            test_prediction_df = pd.concat([test_prediction_df, temp_df])
        
        # 合并测试集真实值与预测值
        test_actual_df = df[required_columns + ['sdeptname']].copy()
        test_evaluation_df = pd.merge(
            test_actual_df, 
            test_prediction_df, 
            on=['time_idx', 'store_id', 'item_id'], 
            how='inner'
        )
        
        if len(test_evaluation_df) == 0:
            raise ValueError("测试集预测结果与真实值无法匹配")
        
        print(f"   测试集匹配到 {len(test_evaluation_df)} 条评估记录")
        
        # 补充dtdate字段用于可视化
        test_evaluation_df = test_evaluation_df.merge(
            df[['store_id', 'item_id', 'time_idx', 'dtdate']],
            on=['store_id', 'item_id', 'time_idx'],
            how='left'
        )
        
        # 按天评估测试集指标
        test_daily_actual = test_evaluation_df['sales'].values
        test_daily_pred = test_evaluation_df['prediction'].values
        test_daily_non_zero_mask = test_daily_actual > 0
        test_daily_non_zero_wape = calculate_wape(test_daily_actual[test_daily_non_zero_mask], test_daily_pred[test_daily_non_zero_mask]) if test_daily_non_zero_mask.sum() > 0 else 0
        
        # 按月评估测试集指标
        # 按商品-店铺-月份分组求和
        test_evaluation_df['year_month'] = test_evaluation_df['dtdate'].dt.to_period('M')
        test_monthly = test_evaluation_df.groupby(['store_id', 'item_id', 'year_month']).agg({
            'sales': 'sum',
            'prediction': 'sum'
        }).reset_index()
        
        test_monthly_actual = test_monthly['sales'].values
        test_monthly_pred = test_monthly['prediction'].values
        test_monthly_non_zero_mask = test_monthly_actual > 0
        test_monthly_non_zero_wape = calculate_wape(test_monthly_actual[test_monthly_non_zero_mask], test_monthly_pred[test_monthly_non_zero_mask]) if test_monthly_non_zero_mask.sum() > 0 else 0
        
        results['test'] = {
            # 按天指标
            'daily_wape': calculate_wape(test_daily_actual, test_daily_pred),
            'daily_mae': calculate_mae(test_daily_actual, test_daily_pred),
            'daily_rmse': calculate_rmse(test_daily_actual, test_daily_pred),
            'daily_total_actual': np.sum(test_daily_actual),
            'daily_total_predicted': np.sum(test_daily_pred),
            'daily_prediction_bias': (np.sum(test_daily_pred) - np.sum(test_daily_actual)) / np.sum(test_daily_actual) if np.sum(test_daily_actual) > 0 else 0,
            'daily_non_zero_wape': test_daily_non_zero_wape,
            'daily_non_zero_count': test_daily_non_zero_mask.sum(),
            'daily_record_count': len(test_daily_actual),
            
            # 按月指标
            'monthly_wape': calculate_wape(test_monthly_actual, test_monthly_pred),
            'monthly_mae': calculate_mae(test_monthly_actual, test_monthly_pred),
            'monthly_rmse': calculate_rmse(test_monthly_actual, test_monthly_pred),
            'monthly_total_actual': np.sum(test_monthly_actual),
            'monthly_total_predicted': np.sum(test_monthly_pred),
            'monthly_prediction_bias': (np.sum(test_monthly_pred) - np.sum(test_monthly_actual)) / np.sum(test_monthly_actual) if np.sum(test_monthly_actual) > 0 else 0,
            'monthly_non_zero_wape': test_monthly_non_zero_wape,
            'monthly_non_zero_count': test_monthly_non_zero_mask.sum(),
            'monthly_record_count': len(test_monthly_actual),
            
            # 详细预测数据
            'detailed_predictions': test_evaluation_df  # 添加详细的预测数据（包含dtdate）
        }
        
        print(f"✅ 测试集评估完成 - 按天WAPE: {results['test']['daily_wape']:.4f}, 按月WAPE: {results['test']['monthly_wape']:.4f}")
        
        # 持久化保存TFT测试集详细预测结果
        cols_to_save = ['store_id', 'item_id', 'sales', 'dtdate', 'time_idx', 'prediction', 'sdeptname']
        test_evaluation_df[cols_to_save].to_csv('tft_test_predictions.csv', index=False)
        print("✅ 已保存TFT测试集详细预测结果到 tft_test_predictions.csv")

        return results
    
    def fit(self, file_path):
        """
        完整的TFT模型训练流程
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            训练好的模型
        """
        print("🚀 开始TFT模型完整训练流程")
        
        # 步骤1: 数据加载与预处理
        df = self.load_and_preprocess_data(file_path)
        
        # 步骤2: 特征分组定义
        feature_groups = self.define_feature_groups()
        
        # 步骤3: 创建TimeSeriesDataSet
        self.create_timeseries_dataset(df, feature_groups)
        
        # 步骤4: 创建模型
        self.create_model()
        
        # 步骤5: 训练模型
        self.train_model()
        
        print("🎉 TFT模型训练流程完成")
        return self.model

if __name__ == "__main__":
    # 测试TFT模型
    from data_analysis import load_and_analyze_data
    
    # 加载数据
    df, train_mask, val_mask, test_mask = load_and_analyze_data()
    
    # 训练TFT模型
    tft_model = TFTModel(
        prediction_length=30,
        encoder_length=90,
        learning_rate=0.0002,
        hidden_size=64,
        attention_head_size=8,
        dropout=0.2,
        hidden_continuous_size=32,
        batch_size=1024,
        max_epochs=30,  
        patience=5
    )
    
    # 训练模型
    model = tft_model.fit('model_data_top10percent.csv')
    
    # 获取预处理后的数据用于评估
    df = tft_model.load_and_preprocess_data('model_data_top10percent.csv')
    
    # 评估模型
    results = tft_model.predict_and_evaluate(df)
    
    print("\n📋 TFT模型评估结果:")
    for dataset, metrics in results.items():
        print(f"\n{dataset.upper()} 集:")
        print("按天指标:")
        print(f"  daily_wape: {metrics['daily_wape']:.4f}")
        print(f"  daily_mae: {metrics['daily_mae']:.4f}")
        print(f"  daily_rmse: {metrics['daily_rmse']:.4f}")
        print(f"  daily_prediction_bias: {metrics['daily_prediction_bias']:.4f}")
        print(f"  daily_non_zero_wape: {metrics['daily_non_zero_wape']:.4f}")
        print("按月指标:")
        print(f"  monthly_wape: {metrics['monthly_wape']:.4f}")
        print(f"  monthly_mae: {metrics['monthly_mae']:.4f}")
        print(f"  monthly_rmse: {metrics['monthly_rmse']:.4f}")
        print(f"  monthly_prediction_bias: {metrics['monthly_prediction_bias']:.4f}")
        print(f"  monthly_non_zero_wape: {metrics['monthly_non_zero_wape']:.4f}") 