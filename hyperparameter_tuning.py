#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Optuna 对 TFTModel 进行超参数搜索

功能:
- 以验证集的 val_loss 作为优化目标，配合 Optuna 的 PyTorch Lightning 剪枝回调进行早停剪枝
- 训练完成后打印/保存最佳参数，并可视化优化历史与参数重要性（若安装了 plotly）
"""

import os
import json
import gc
import argparse
from typing import Dict, Any

import optuna
import optuna.logging as optuna_logging
from optuna.integration import PyTorchLightningPruningCallback

import torch

from tft_model import TFTModel


def build_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """定义超参数搜索空间。"""
    # 固定分布（避免 Optuna 对同名参数的分布不一致错误）
    hidden_size = trial.suggest_categorical("hidden_size", [16, 24, 32, 48, 64, 96])

    params = {
        "prediction_length": 30,
        "encoder_length": trial.suggest_categorical("encoder_length", [60, 90, 120]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_size": hidden_size,
        # 注意：这里固定 choices，实际使用时会按 hidden_size 映射到可整除的有效值
        "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4, 8, 16]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [8, 16, 24, 32]),
        "batch_size": trial.suggest_categorical("batch_size", [128,256]),
    }

    return params


def objective(trial: optuna.Trial, data_file: str, max_epochs: int, patience: int, seed: int) -> float:
    """Optuna 目标函数：返回最小化目标 val_loss。"""
    params = build_search_space(trial)

    # 在每个试验开始时打印当前参数（打印前做一次 head 修正，避免不合法组合）
    requested_heads = params["attention_head_size"]
    if params["hidden_size"] % requested_heads != 0:
        candidates = [h for h in [1, 2, 4, 8, 16] if params["hidden_size"] % h == 0]
        if not candidates:
            candidates = [1]
        corrected_heads = min(candidates, key=lambda h: abs(h - requested_heads))
    else:
        corrected_heads = requested_heads
    params["attention_head_size"] = corrected_heads

    print(f"\n[Trial {trial.number}] 开始，参数: "
          f"lr={params['learning_rate']:.2e}, hidden_size={params['hidden_size']}, "
          f"heads={params['attention_head_size']} (req={requested_heads}), dropout={params['dropout']:.2f}, "
          f"hidden_cont={params['hidden_continuous_size']}, batch={params['batch_size']}, "
          f"encoder_len={params['encoder_length']}, pred_len={params['prediction_length']}")

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    model = None
    best_score = None
    try:
        model = TFTModel(
            prediction_length=params["prediction_length"],
            encoder_length=params["encoder_length"],
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"],
            dropout=params["dropout"],
            hidden_continuous_size=params["hidden_continuous_size"],
            batch_size=params["batch_size"],
            max_epochs=max_epochs,
            patience=patience,
            random_seed=seed,
            optuna_pruning_callback=pruning_callback,
        )

        # 训练（内部会进行验证并记录 val_loss）
        model.fit(data_file)

        # 读取最佳验证损失
        try:
            # Lightning 提供的最佳分数对象 -> Tensor/float
            best_score = model.trainer.checkpoint_callback.best_model_score
            best_score = float(best_score.detach().cpu().item() if hasattr(best_score, "detach") else float(best_score))
        except Exception:
            # 兜底：从 callback_metrics 读取
            if model.trainer is not None and hasattr(model.trainer, "callback_metrics"):
                metric = model.trainer.callback_metrics.get("val_loss", None)
                if metric is not None:
                    best_score = float(metric.detach().cpu().item() if hasattr(metric, "detach") else float(metric))
    except RuntimeError as e:
        # 典型如 CUDA out of memory，返回一个很差的值以继续搜索
        print(f"[Trial {trial.number}] 运行时错误: {e}. 该试验将返回一个较差的目标值并继续。")
        best_score = float("inf")
    except Exception as e:
        # 其他异常同样不阻断整体搜索
        print(f"[Trial {trial.number}] 异常: {e}. 该试验将返回一个较差的目标值并继续。")
        best_score = float("inf")
    finally:
        # 主动释放显存与内存，避免多 trial 累积
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        del model
        gc.collect()

    if best_score is None or best_score == float("inf"):
        print(f"[Trial {trial.number}] 完成，val_loss=inf")
        return float("inf")

    print(f"[Trial {trial.number}] 完成，val_loss={best_score:.6f}")
    return best_score


def run_study(
    data_file: str,
    n_trials: int,
    timeout: int,
    study_name: str,
    storage: str,
    max_epochs: int,
    patience: int,
    seed: int,
    show_plots: bool,
):
    # 采样器与剪枝器
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage if storage else None,
        load_if_exists=bool(storage),
    )

    def _print_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        print(f"[Trial {trial.number}] 状态={trial.state.name}, 值={trial.value}, 参数={trial.params}")

    study.optimize(
        lambda t: objective(t, data_file=data_file, max_epochs=max_epochs, patience=patience, seed=seed),
        n_trials=n_trials if n_trials > 0 else None,
        timeout=timeout if timeout > 0 else None,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[_print_callback],
        catch=(RuntimeError,),
    )

    print("\n========== 最佳结果 ==========")
    print(f"最佳 val_loss: {study.best_value:.6f}")
    print("最佳超参数:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")

    # 保存最佳参数
    with open("optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, ensure_ascii=False, indent=2)
    print("已保存最佳参数到 optuna_best_params.json")

    if show_plots:
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )

            print("显示优化历史...（需要图形界面/浏览器）")
            plot_optimization_history(study).show()
            print("显示参数重要性...")
            plot_param_importances(study).show()
            print("显示并行坐标图...")
            plot_parallel_coordinate(study).show()
        except Exception as e:
            print(f"可视化失败（可忽略）: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna 超参数搜索 - TFTModel")
    parser.add_argument("--data_file", type=str, default="model_data_top10percent.csv", help="数据文件路径")
    parser.add_argument("--n_trials", type=int, default=20, help="搜索轮数（<=0 表示不限）")
    parser.add_argument("--timeout", type=int, default=0, help="搜索超时（秒，0 表示不限）")
    parser.add_argument("--study_name", type=str, default="tft_hyperparam_opt", help="Study 名称")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage，如 sqlite:///optuna.db；为空则内存")
    parser.add_argument("--max_epochs", type=int, default=30, help="每个 trial 的最大训练轮数")
    parser.add_argument("--patience", type=int, default=3, help="早停 patience")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--show_plots", action="store_true", help="是否显示 plotly 可视化图表")
    return parser.parse_args()


def main():
    args = parse_args()

    # 基本存在性检查
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"数据文件不存在: {args.data_file}")

    # Optuna 日志级别
    optuna_logging.set_verbosity(optuna_logging.INFO)

    run_study(
        data_file=args.data_file,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()


