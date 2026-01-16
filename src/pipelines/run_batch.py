"""
批量运行流水线脚本
Batch run pipeline script

本脚本整合了批量运行功能，支持：
1. 使用预定义配置运行实验
2. 运行完整的模型对比实验（实验1 + 实验2）
3. 自定义参数批量运行所有合金类型
4. 断点续传，支持从上次中断处继续执行

This script integrates batch run functionality, supporting:
1. Run experiments using predefined configurations
2. Run complete model comparison experiments (Experiment 1 + Experiment 2)
3. Custom parameter batch run for all alloy types
4. Resume from interruption with progress management

使用方法 / Usage:
    # 列出所有可用配置
    python -m src.pipelines.run_batch --list
    
    # 运行指定配置
    python -m src.pipelines.run_batch --config experiment1_all_ml_models
    
    # 运行多个配置
    python -m src.pipelines.run_batch --config experiment1_all_ml_models experiment2a_all_nn_scibert
    
    # 运行所有实验（实验1 + 实验2）
    python -m src.pipelines.run_batch --all
    
    # 仅运行实验1（传统ML模型）
    python -m src.pipelines.run_batch --experiment1
    
    # 仅运行实验2（神经网络 + BERT嵌入）
    python -m src.pipelines.run_batch --experiment2
    
    # 预览命令（不实际执行）
    python -m src.pipelines.run_batch --all --dry_run
    
    # 自定义参数运行所有合金
    python -m src.pipelines.run_batch --custom \
        --embedding_type steelbert \
        --use_composition_feature \
        --use_element_embedding \
        --models xgboost lightgbm
    
    # 断点续传：从上次中断处继续
    python -m src.pipelines.run_batch --all --resume
    
    # 查看任务进度
    python -m src.pipelines.run_batch --show_progress
    
    # 清除所有进度记录
    python -m src.pipelines.run_batch --clear_progress
    
    # 清除指定配置的进度记录
    python -m src.pipelines.run_batch --clear_progress experiment1_all_ml_models
"""

import argparse
import logging
import subprocess
import sys
import threading
import queue
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置标准输出和标准错误为UTF-8编码
# Set stdout and stderr to UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
from datetime import datetime

# 导入配置
from src.pipelines.batch_configs import (
    ALLOY_CONFIGS,
    BATCH_CONFIGS,
    get_alloy_config,
    list_available_alloys
)

# 配置日志
# 创建日志格式化器
log_format = logging.Formatter(
    fmt='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 创建控制台处理器（实时输出）
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)
console_handler.flush = lambda: sys.stdout.flush()  # 强制刷新

# 创建文件处理器（保存日志）
log_filename = f'batch_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
file_handler.setFormatter(log_format)

# 配置根日志记录器
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_filename}")
logger.info(f"日志级别: INFO (控制台), DEBUG (文件)")


def enqueue_output(stream, queue, stream_name):
    """
    在独立线程中读取流输出并放入队列
    Read stream output in a separate thread and put into queue
    
    Args:
        stream: 输出流 (stdout/stderr)
        queue: 队列对象
        stream_name: 流名称 (用于调试)
    """
    try:
        for line in iter(stream.readline, ''):
            if line:
                queue.put(line.rstrip())
    except Exception as e:
        logger.debug(f"Error reading {stream_name}: {e}")
    finally:
        stream.close()


class SimpleNamespace:
    """简单的命名空间类，用于模拟argparse.Namespace"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ProgressManager:
    """
    任务进度管理器
    Task progress manager for resumable batch runs
    """
    
    def __init__(self, progress_file: str = ".batch_progress.json"):
        """
        初始化进度管理器
        
        Args:
            progress_file: 进度文件路径
        """
        self.progress_file = Path(progress_file)
        self.progress_data = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """加载进度文件"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"无法加载进度文件: {e}，将创建新的进度记录")
                return {}
        return {}
    
    def _save_progress(self):
        """保存进度到文件"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")
    
    def get_config_progress(self, config_name: str) -> Dict[str, str]:
        """
        获取指定配置的进度
        
        Args:
            config_name: 配置名称
            
        Returns:
            Dict[str, str]: 任务进度字典 {task_key: status}
                           task_key 格式: "alloy_type" 或 "alloy_type_model_name"
        """
        return self.progress_data.get(config_name, {})
    
    def is_task_completed(self, config_name: str, task_key: str) -> bool:
        """
        检查任务是否已完成
        
        Args:
            config_name: 配置名称
            task_key: 任务键（合金类型 或 合金类型_模型名）
            
        Returns:
            bool: 是否已完成
        """
        config_progress = self.get_config_progress(config_name)
        return config_progress.get(task_key) == "success"
    
    def update_task_status(self, config_name: str, task_key: str, status: str):
        """
        更新任务状态
        
        Args:
            config_name: 配置名称
            task_key: 任务键（合金类型 或 合金类型_模型名）
            status: 状态 (success/failed/running)
        """
        if config_name not in self.progress_data:
            self.progress_data[config_name] = {}
        
        self.progress_data[config_name][task_key] = status
        self._save_progress()
    
    def get_pending_tasks(self, config_name: str, all_tasks: List[str]) -> List[str]:
        """
        获取待执行的任务列表（排除已成功完成的）
        
        Args:
            config_name: 配置名称
            all_tasks: 所有任务键列表
            
        Returns:
            List[str]: 待执行的任务列表
        """
        config_progress = self.get_config_progress(config_name)
        pending = [
            task for task in all_tasks 
            if config_progress.get(task) != "success"
        ]
        return pending
    
    def get_statistics(self, config_name: str) -> Dict[str, int]:
        """
        获取配置的统计信息
        
        Args:
            config_name: 配置名称
            
        Returns:
            Dict[str, int]: 统计信息 {success: n, failed: n, total: n}
        """
        config_progress = self.get_config_progress(config_name)
        stats = {
            'success': sum(1 for s in config_progress.values() if s == 'success'),
            'failed': sum(1 for s in config_progress.values() if s == 'failed'),
            'total': len(config_progress)
        }
        return stats
    
    def clear_progress(self, config_name: Optional[str] = None):
        """
        清除进度记录
        
        Args:
            config_name: 配置名称，如果为 None 则清除所有进度
        """
        if config_name is None:
            self.progress_data = {}
            logger.info("已清除所有进度记录")
        elif config_name in self.progress_data:
            del self.progress_data[config_name]
            logger.info(f"已清除配置 [{config_name}] 的进度记录")
        
        self._save_progress()
    
    def show_progress(self, config_name: Optional[str] = None):
        """
        显示进度信息
        
        Args:
            config_name: 配置名称，如果为 None 则显示所有配置
        """
        if config_name:
            configs_to_show = {config_name: self.progress_data.get(config_name, {})}
        else:
            configs_to_show = self.progress_data
        
        if not configs_to_show or all(not v for v in configs_to_show.values()):
            logger.info("没有进度记录")
            return
        
        logger.info("\n" + "=" * 100)
        logger.info("任务进度 / Task Progress")
        logger.info("=" * 100)
        
        for cfg_name, tasks in configs_to_show.items():
            if not tasks:
                continue
                
            stats = self.get_statistics(cfg_name)
            logger.info(f"\n配置: {cfg_name}")
            logger.info(f"  总任务数: {stats['total']}")
            logger.info(f"  成功: {stats['success']} ✓")
            logger.info(f"  失败: {stats['failed']} ✗")
            logger.info(f"  完成率: {stats['success']/stats['total']*100:.1f}%" if stats['total'] > 0 else "  完成率: 0%")
            
            logger.info(f"\n  任务详情:")
            for alloy, status in sorted(tasks.items()):
                icon = "✓" if status == "success" else "✗" if status == "failed" else "⋯"
                logger.info(f"    {icon} {alloy}: {status}")
        
        logger.info("\n" + "=" * 100)


def make_task_key(alloy_type: str, model_name: Optional[str] = None) -> str:
    """
    生成任务键
    
    Args:
        alloy_type: 合金类型
        model_name: 模型名称（可选）
        
    Returns:
        str: 任务键，格式为 "alloy_type" 或 "alloy_type_model_name"
    """
    if model_name:
        return f"{alloy_type}_{model_name}"
    return alloy_type


def build_command(alloy_type: str, config: Dict[str, Any], args, model_name: Optional[str] = None) -> List[str]:
    """
    构建单个合金类型的命令
    Build command for a single alloy type
    
    Args:
        alloy_type: 合金类型
        config: 合金配置字典
        args: 参数对象
        
    Returns:
        List[str]: 命令列表
    """
    # 推断数据集名称
    # 移除常见的后缀：_processed, _cleaned, _withID, _with_ID
    data_file = Path(config['raw_data'])
    dataset_name = data_file.stem
    # 移除组合后缀（如 _Processed_cleaned）
    dataset_name = dataset_name.replace('_Processed_cleaned', '')
    # 按顺序移除单个后缀
    for suffix in ['_with_ID', '_withID', '_cleaned', '_processed', '_Processed']:
        dataset_name = dataset_name.replace(suffix, '')

    # 构建结果目录
    result_dir = f"output/new_results_withuncertainty/{alloy_type}/{dataset_name}/{args.embedding_type}"

    # 基础命令
    cmd = [
        sys.executable, "-m", "src.pipelines.end_to_end_pipeline",
        "--data_file", config['raw_data'],
        "--result_dir", result_dir,
        "--target_columns", *config['targets'],
        "--embedding_type", args.embedding_type,
        "--alloy_type", alloy_type,
        "--dataset_name", dataset_name,
    ]
    
    # 添加工艺参数列
    # 优先使用 args 中的配置（来自batch config），如果没有则使用 alloy config
    # 对于神经网络模型，如果配置中有 nn_additional_features，则优先使用它
    processing_cols_to_use = None
    
    # 神经网络模型优先检查 nn_additional_features
    if args.use_nn and 'nn_additional_features' in config:
        processing_cols_to_use = config['nn_additional_features']
        logger.info(f"神经网络模型使用额外数值特征: {processing_cols_to_use}")
    elif hasattr(args, 'processing_cols') and args.processing_cols is not None and len(args.processing_cols) > 0:
        # 使用 batch config 中的设置（非空列表）
        processing_cols_to_use = args.processing_cols
    elif 'processing_cols' in config:
        # 使用 alloy config 中的默认设置
        processing_cols_to_use = config['processing_cols']
    
    if processing_cols_to_use:  # 只有非空时才添加
        cmd.extend(["--processing_cols", *processing_cols_to_use])
    
    # 添加工艺描述列
    if config.get('processing_text_column'):
        cmd.extend(["--processing_text_column", config['processing_text_column']])
    
    # 特征配置
    if args.use_composition_feature:
        cmd.extend(["--use_composition_feature", "True"])
    if args.use_element_embedding:
        cmd.extend(["--use_element_embedding", "True"])
    # 只有当数据集有工艺文本列时才启用工艺嵌入
    # Only enable process embedding when the dataset has a processing text column
    if args.use_process_embedding and config.get('processing_text_column'):
        cmd.extend(["--use_process_embedding", "True"])
    elif args.use_process_embedding and not config.get('processing_text_column'):
        logger.warning(f"数据集 {alloy_type} 没有工艺文本列，跳过工艺嵌入 / Dataset {alloy_type} has no processing text column, skipping process embedding")
    if args.use_temperature:
        cmd.extend(["--use_temperature", "True"])
    
    # 模型配置
    if args.use_nn:
        cmd.append("--use_nn")
    elif model_name:
        # 如果指定了单个模型，只运行该模型
        cmd.extend(["--models", model_name])
    elif args.models:
        # 否则运行所有指定的模型
        cmd.extend(["--models", *args.models])
    
    # 训练配置
    if args.cross_validate:
        cmd.append("--cross_validate")
        cmd.extend(["--num_folds", str(args.num_folds)])
    else:
        cmd.extend(["--test_size", str(args.test_size)])
    
    cmd.extend(["--random_state", str(args.random_state)])
    
    # 神经网络参数
    if args.use_nn:
        cmd.extend(["--epochs", str(args.epochs)])
        cmd.extend(["--patience", str(args.patience)])
        cmd.extend(["--batch_size", str(args.batch_size)])
    
    # 优化配置
    if args.use_optuna:
        cmd.append("--use_optuna")
        cmd.extend(["--n_trials", str(args.n_trials)])
    
    # 评估配置
    if args.evaluate_after_train:
        cmd.append("--evaluate_after_train")
    if args.run_shap_analysis:
        cmd.append("--run_shap_analysis")
        
    # 重复实验配置 - 根据用户要求移除 --n_repeats 参数，使用 num_folds 计算不确定度
    # if hasattr(args, 'n_repeats') and args.n_repeats > 1:
    #     cmd.extend(["--n_repeats", str(args.n_repeats)])
    
    return cmd


def format_command_for_display(cmd: List[str]) -> str:
    """
    格式化命令用于显示，为包含特殊字符的参数添加引号
    Format command for display, add quotes for arguments with special characters

    Args:
        cmd: 命令列表

    Returns:
        str: 格式化后的命令字符串
    """
    formatted_parts = []
    for part in cmd:
        # 如果参数包含空格、括号等特殊字符，添加引号
        if any(char in part for char in [' ', '(', ')', '%', '/', '\\', '℃']):
            formatted_parts.append(f'"{part}"')
        else:
            formatted_parts.append(part)
    return ' '.join(formatted_parts)


def run_alloy_pipeline(alloy_type: str, config: Dict[str, Any], args, model_name: Optional[str] = None) -> bool:
    """
    运行单个合金类型的流水线
    Run pipeline for a single alloy type

    Args:
        alloy_type: 合金类型
        config: 合金配置字典
        args: 参数对象
        model_name: 模型名称（可选，如果指定则只运行该模型）

    Returns:
        bool: 是否成功
    """
    start_time = datetime.now()
    task_desc = f"{alloy_type}"
    if model_name:
        task_desc += f" - {model_name}"
    task_desc += f" - {config['description']}"
    
    logger.info("=" * 100)
    logger.info(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 开始运行: {task_desc}")
    logger.info("=" * 100)

    try:
        cmd = build_command(alloy_type, config, args, model_name)
        logger.info(f"执行命令: {format_command_for_display(cmd)}")
        logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("-" * 100)
        logger.info(f">>> 开始执行子进程，实时输出如下:")
        logger.info("-" * 100)

        # 使用 Popen 实现实时输出，不合并 stderr，分别处理
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # 分别处理 stderr
            text=True,
            encoding='utf-8',
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )

        # 创建队列用于异步读取输出
        output_queue = queue.Queue()
        
        # 启动独立线程读取 stdout 和 stderr
        stdout_thread = threading.Thread(
            target=enqueue_output,
            args=(process.stdout, output_queue, 'stdout'),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=enqueue_output,
            args=(process.stderr, output_queue, 'stderr'),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()

        # 实时读取并输出日志
        output_lines = []
        while True:
            # 检查进程是否结束
            return_code = process.poll()
            
            # 从队列中获取输出
            try:
                line = output_queue.get(timeout=0.1)
                if line:  # 只输出非空行
                    # 添加前缀以区分子进程输出
                    logger.info(f"  [{alloy_type}] {line}")
                    # 强制刷新输出，确保实时显示
                    sys.stdout.flush()
                    output_lines.append(line)
            except queue.Empty:
                # 队列为空，检查进程是否结束
                if return_code is not None:
                    # 进程已结束，读取剩余输出
                    while not output_queue.empty():
                        try:
                            line = output_queue.get_nowait()
                            if line:
                                logger.info(f"  [{alloy_type}] {line}")
                                sys.stdout.flush()
                                output_lines.append(line)
                        except queue.Empty:
                            break
                    break

        # 等待进程结束
        return_code = process.wait()

        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("-" * 100)
        logger.info(f"<<< 子进程执行完成")
        logger.info("-" * 100)
        logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"执行耗时: {duration}")
        logger.info(f"返回代码: {return_code}")

        if return_code == 0:
            logger.info(f"[OK] {alloy_type} 运行成功 ✓")
            logger.info("=" * 100)
            return True
        else:
            logger.error(f"[FAIL] {alloy_type} 运行失败，返回代码: {return_code} ✗")
            logger.error("=" * 100)
            return False

    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.error(f"[FAIL] {alloy_type} 运行失败 ✗")
        logger.error(f"执行耗时: {duration}")
        logger.error(f"错误信息:\n{e.stderr if e.stderr else '无错误信息'}")
        logger.error("=" * 100)
        return False
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.error(f"[FAIL] {alloy_type} 运行出错: {str(e)} ✗")
        logger.error(f"执行耗时: {duration}")
        logger.error("=" * 100)
        return False


def config_to_args(config: Dict[str, Any]) -> SimpleNamespace:
    """
    将配置字典转换为参数对象
    Convert config dict to args object

    Args:
        config: 配置字典

    Returns:
        SimpleNamespace: 参数对象
    """
    args_dict = {
        'alloy_types': config.get('alloy_types'),
        'exclude_alloys': config.get('exclude_alloys', []),
        'embedding_type': config['embedding_type'],
        'use_composition_feature': config.get('use_composition_feature', False),
        'use_element_embedding': config.get('use_element_embedding', False),
        'use_process_embedding': config.get('use_process_embedding', False),
        'use_temperature': config.get('use_temperature', False),
        'models': config.get('models'),
        'use_nn': config.get('use_nn', False),
        'cross_validate': config.get('cross_validate', False),
        'num_folds': config.get('num_folds', 9),
        'test_size': config.get('test_size', 0.2),
        'random_state': config.get('random_state', 42),
        'epochs': config.get('epochs', 200),
        'patience': config.get('patience', 30),
        'batch_size': config.get('batch_size', 256),
        'use_optuna': config.get('use_optuna', False),
        'n_trials': config.get('n_trials', 30),
        'mlp_max_iter': config.get('mlp_max_iter', 300),
        'evaluate_after_train': config.get('evaluate_after_train', True),
        'run_shap_analysis': config.get('run_shap_analysis', True),
        'processing_text_column': None,
        'processing_cols': config.get('processing_cols'),  # 从batch config获取processing_cols设置
        'nn_additional_features': config.get('nn_additional_features'),  # 神经网络额外特征
        'n_repeats': config.get('n_repeats', 1),
    }

    return SimpleNamespace(**args_dict)


def run_batch_config(
    config_name: str, 
    config: Dict[str, Any], 
    dry_run: bool = False, 
    max_workers: int = 4,
    progress_manager: Optional[ProgressManager] = None,
    resume: bool = False
) -> Dict[str, str]:
    """
    运行单个批量配置
    Run a single batch configuration

    Args:
        config_name: 配置名称
        config: 配置字典
        dry_run: 是否仅预览
        max_workers: 最大并发任务数 / Maximum concurrent workers
        progress_manager: 进度管理器（可选）
        resume: 是否从上次中断处继续（断点续传）

    Returns:
        Dict[str, str]: 每个合金类型的运行结果
    """
    batch_start_time = datetime.now()
    logger.info("\n" + "=" * 100)
    logger.info(f"运行配置: {config_name}")
    logger.info(f"描述: {config['description']}")
    logger.info(f"开始时间: {batch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 100)

    # 转换配置为参数对象
    args = config_to_args(config)

    # 确定要运行的合金类型
    if args.alloy_types:
        alloy_types = args.alloy_types
    else:
        alloy_types = list_available_alloys()

    # 排除指定的合金类型
    alloy_types = [a for a in alloy_types if a not in args.exclude_alloys]
    
    # 生成任务列表：(alloy_type, model_name, task_key)
    tasks = []
    if args.use_nn:
        # 神经网络模式：每个合金类型一个任务
        for alloy_type in alloy_types:
            task_key = make_task_key(alloy_type, "nn")
            tasks.append((alloy_type, "nn", task_key))
    elif args.models:
        # ML模型模式：每个合金类型 × 每个模型 = 多个任务
        for alloy_type in alloy_types:
            for model in args.models:
                task_key = make_task_key(alloy_type, model)
                tasks.append((alloy_type, model, task_key))
    else:
        # 没有指定模型（不应该发生）
        for alloy_type in alloy_types:
            task_key = make_task_key(alloy_type)
            tasks.append((alloy_type, None, task_key))
    
    # 断点续传：过滤掉已完成的任务
    original_count = len(tasks)
    if resume and progress_manager:
        task_keys = [task[2] for task in tasks]
        pending_keys = set(progress_manager.get_pending_tasks(config_name, task_keys))
        tasks = [task for task in tasks if task[2] in pending_keys]
        skipped_count = original_count - len(tasks)
        if skipped_count > 0:
            logger.info(f"[断点续传] 跳过已完成的 {skipped_count} 个任务")
            logger.info(f"[断点续传] 剩余 {len(tasks)} 个任务待执行")

    # 显示任务信息
    unique_alloys = list(set(task[0] for task in tasks))
    unique_models = list(set(task[1] for task in tasks if task[1]))
    
    logger.info(f"合金类型: {', '.join(sorted(unique_alloys))}")
    logger.info(f"嵌入类型: {args.embedding_type}")
    if args.use_nn:
        logger.info(f"模型: 神经网络")
    elif unique_models:
        logger.info(f"模型: {', '.join(sorted(unique_models))}")
    logger.info(f"总任务数: {len(tasks)} ({len(unique_alloys)} 合金 × {len(unique_models) if unique_models else 1} 模型)")
    logger.info(f"并发数: {max_workers}")
    logger.info("=" * 100)


    # 运行统计
    results = {}
    task_times = []  # 记录每个任务的耗时

    if dry_run:
        # DRY RUN 模式：不实际运行，只显示命令
        logger.info("\n[DRY RUN 模式] 仅显示命令，不实际执行")
        logger.info("-" * 100)
        for i, (alloy_type, model_name, task_key) in enumerate(tasks, 1):
            alloy_config = ALLOY_CONFIGS[alloy_type]
            cmd = build_command(alloy_type, alloy_config, args, model_name if model_name != "nn" else None)
            task_desc = f"{alloy_type} - {model_name}" if model_name else alloy_type
            logger.info(f"\n[DRY RUN] [{i}/{len(tasks)}] {task_desc}:")
            logger.info(f"  {format_command_for_display(cmd)}\n")
            results[task_key] = "skipped"
    else:
        # 实际运行：使用线程池并行执行
        def run_task(alloy_type: str, model_name: Optional[str], task_key: str, index: int) -> tuple:
            """运行单个任务"""
            task_start = datetime.now()
            alloy_config = ALLOY_CONFIGS[alloy_type]
            
            task_desc = f"{alloy_type} - {model_name}" if model_name else alloy_type
            
            logger.info(f"\n{'='*100}")
            logger.info(f"[进度: {index}/{len(tasks)}] 开始处理: {task_desc}")
            logger.info(f"任务开始时间: {task_start.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*100}")
            
            # 更新进度：任务开始
            if progress_manager:
                progress_manager.update_task_status(config_name, task_key, "running")
            
            # 运行流水线（如果是神经网络，model_name传None）
            model_to_run = None if model_name == "nn" else model_name
            success = run_alloy_pipeline(alloy_type, alloy_config, args, model_to_run)
            
            task_end = datetime.now()
            task_duration = task_end - task_start
            
            # 更新进度：任务完成
            status = "success" if success else "failed"
            if progress_manager:
                progress_manager.update_task_status(config_name, task_key, status)
            
            logger.info(f"\n{'='*100}")
            logger.info(f"[进度: {index}/{len(tasks)}] 完成处理: {task_desc}")
            logger.info(f"任务结束时间: {task_end.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"任务耗时: {task_duration}")
            logger.info(f"状态: {'成功 ✓' if success else '失败 ✗'}")
            logger.info(f"{'='*100}\n")
            
            return (task_key, status, task_duration)

        # 使用线程池执行所有任务
        logger.info(f"\n开始并行执行 {len(tasks)} 个任务 (并发数: {max_workers})...")
        logger.info("=" * 100)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(run_task, alloy_type, model_name, task_key, i): task_key
                for i, (alloy_type, model_name, task_key) in enumerate(tasks, 1)
            }

            # 处理完成的任务
            completed_count = 0
            for future in as_completed(future_to_task):
                completed_count += 1
                task_key, status, duration = future.result()
                results[task_key] = status
                task_times.append(duration.total_seconds())
                
                # 计算平均时间和预估剩余时间
                avg_time = sum(task_times) / len(task_times)
                remaining_tasks = len(tasks) - completed_count
                estimated_remaining = avg_time * remaining_tasks / max_workers  # 考虑并发
                
                current_time = datetime.now()
                elapsed_time = current_time - batch_start_time
                
                logger.info(f"\n{'#'*100}")
                logger.info(f"[总体进度] 已完成: {completed_count}/{len(tasks)} ({completed_count/len(tasks)*100:.1f}%)")
                logger.info(f"[时间统计] 已用时间: {elapsed_time}")
                logger.info(f"[时间统计] 平均任务耗时: {avg_time:.1f}秒")
                logger.info(f"[时间估算] 预计剩余时间: {estimated_remaining:.1f}秒 ({estimated_remaining/60:.1f}分钟)")
                logger.info(f"[当前状态] 成功: {sum(1 for s in results.values() if s == 'success')}, "
                          f"失败: {sum(1 for s in results.values() if s == 'failed')}")
                logger.info(f"{'#'*100}\n")
                
                if status == "failed":
                    logger.warning(f"[WARN] {task_key} 失败，但继续运行其他任务")

    batch_end_time = datetime.now()
    batch_duration = batch_end_time - batch_start_time
    
    logger.info("\n" + "=" * 100)
    logger.info(f"配置 [{config_name}] 执行完成")
    logger.info(f"结束时间: {batch_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {batch_duration}")
    logger.info("=" * 100)

    return results


def list_configs():
    """列出所有可用配置"""
    logger.info("\n可用的批量运行配置 / Available Batch Configurations:\n")
    for name, config in BATCH_CONFIGS.items():
        logger.info(f"  * {name}")
        logger.info(f"    {config['description']}")
        logger.info("")


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='批量运行流水线 / Batch run pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式 / Run Modes:
  1. 使用预定义配置 / Use predefined configurations:
     python -m src.pipelines.run_batch --config <配置名称>

  2. 运行完整实验 / Run complete experiments:
     python -m src.pipelines.run_batch --all          # 所有实验
     python -m src.pipelines.run_batch --experiment1  # 仅实验1
     python -m src.pipelines.run_batch --experiment2  # 仅实验2

  3. 自定义参数运行 / Custom parameter run:
     python -m src.pipelines.run_batch --custom \
         --embedding_type steelbert \
         --models xgboost lightgbm

实验说明 / Experiment Description:
  实验1: 6个合金 × 5个传统ML模型 = 30个任务
  实验2: 6个合金 × 3种BERT嵌入的神经网络 = 18个任务
  总计: 48个训练任务
        """
    )

    # 运行模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--list', action='store_true',
                           help='列出所有可用配置 / List all available configurations')
    mode_group.add_argument('--config', type=str, nargs='*',
                           help='运行指定配置 / Run specified configurations')
    mode_group.add_argument('--all', action='store_true',
                           help='运行所有实验（实验1 + 实验2）/ Run all experiments')
    mode_group.add_argument('--experiment1', action='store_true',
                           help='仅运行实验1（传统ML模型）/ Run experiment 1 only')
    mode_group.add_argument('--experiment2', action='store_true',
                           help='仅运行实验2（神经网络 + BERT嵌入）/ Run experiment 2 only')
    mode_group.add_argument('--custom', action='store_true',
                           help='使用自定义参数运行 / Run with custom parameters')

    # 通用选项
    parser.add_argument('--dry_run', action='store_true',
                       help='仅显示命令不执行 / Show commands without execution')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='最大并发任务数，默认为4 / Maximum concurrent workers (default: 4)')
    
    # 进度管理选项
    progress_group = parser.add_argument_group('进度管理 / Progress Management')
    progress_group.add_argument('--resume', action='store_true',
                               help='断点续传，跳过已完成的任务 / Resume from last run, skip completed tasks')
    progress_group.add_argument('--show_progress', action='store_true',
                               help='显示任务进度 / Show task progress')
    progress_group.add_argument('--clear_progress', type=str, nargs='?', const='__all__', metavar='CONFIG',
                               help='清除进度记录（可选指定配置名称）/ Clear progress (optionally specify config name)')


    # 自定义参数（仅在 --custom 模式下使用）
    custom_group = parser.add_argument_group('自定义参数 / Custom Parameters (for --custom mode)')
    custom_group.add_argument('--alloy_types', type=str, nargs='*', default=None,
                             help='要运行的合金类型 / Alloy types to run')
    custom_group.add_argument('--exclude_alloys', type=str, nargs='*', default=[],
                             help='要排除的合金类型 / Alloy types to exclude')
    custom_group.add_argument('--embedding_type', type=str,
                             choices=['tradition', 'scibert', 'steelbert', 'matscibert'],
                             help='嵌入类型 / Embedding type')
    custom_group.add_argument('--use_composition_feature', action='store_true',
                             help='使用组分特征 / Use composition features')
    custom_group.add_argument('--use_element_embedding', action='store_true',
                             help='使用元素嵌入 / Use element embeddings')
    custom_group.add_argument('--use_process_embedding', action='store_true',
                             help='使用工艺嵌入 / Use process embeddings')
    custom_group.add_argument('--use_temperature', action='store_true',
                             help='使用温度特征 / Use temperature features')
    custom_group.add_argument('--models', type=str, nargs='*',
                             choices=['xgboost', 'sklearn_rf', 'mlp', 'lightgbm', 'catboost'],
                             help='传统ML模型 / Traditional ML models')
    custom_group.add_argument('--use_nn', action='store_true',
                             help='使用神经网络 / Use neural network')
    custom_group.add_argument('--cross_validate', action='store_true',
                             help='使用交叉验证 / Use cross-validation')
    custom_group.add_argument('--num_folds', type=int, default=9,
                             help='交叉验证折数 / Number of CV folds')
    custom_group.add_argument('--test_size', type=float, default=0.2,
                             help='测试集比例 / Test set ratio')
    custom_group.add_argument('--random_state', type=int, default=42,
                             help='随机种子 / Random state')
    custom_group.add_argument('--epochs', type=int, default=200,
                             help='训练轮数 / Training epochs')
    custom_group.add_argument('--patience', type=int, default=30,
                             help='早停耐心值 / Early stopping patience')
    custom_group.add_argument('--batch_size', type=int, default=256,
                             help='批次大小 / Batch size')
    custom_group.add_argument('--use_optuna', action='store_true',
                             help='使用Optuna优化 / Use Optuna optimization')
    custom_group.add_argument('--n_trials', type=int, default=30,
                             help='Optuna试验次数 / Number of Optuna trials')
    custom_group.add_argument('--mlp_max_iter', type=int, default=300,
                             help='MLP最大迭代次数 / MLP maximum iterations')
    custom_group.add_argument('--evaluate_after_train', action='store_true',
                             help='训练后评估 / Evaluate after training')
    custom_group.add_argument('--run_shap_analysis', action='store_true',
                             help='运行SHAP分析 / Run SHAP analysis')
    custom_group.add_argument('--n_repeats', type=int, default=1,
                             help='重复实验次数 / Number of experiment repeats')

    return parser


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 初始化进度管理器
    progress_manager = ProgressManager()
    
    # 处理进度管理命令
    if args.show_progress:
        progress_manager.show_progress()
        return
    
    if args.clear_progress is not None:
        if args.clear_progress == '__all__':
            progress_manager.clear_progress()
        else:
            progress_manager.clear_progress(args.clear_progress)
        return

    # 列出配置
    if args.list:
        list_configs()
        return

    # 确定要运行的配置
    run_configs = []

    if args.config:
        # 使用指定配置
        run_configs = args.config
        # 验证配置名称
        invalid_configs = [c for c in run_configs if c not in BATCH_CONFIGS]
        if invalid_configs:
            logger.error(f"[ERROR] 无效的配置名称: {', '.join(invalid_configs)}")
            logger.info("使用 --list 查看所有可用配置")
            sys.exit(1)

    elif args.all:
        # 运行所有实验
        run_configs = [
            "experiment1_all_ml_models",
            "experiment2a_all_nn_scibert",
            "experiment2b_all_nn_steelbert",
            "experiment2c_all_nn_matscibert"
        ]

    elif args.experiment1:
        # 仅运行实验1
        run_configs = ["experiment1_all_ml_models"]

    elif args.experiment2:
        # 仅运行实验2
        run_configs = [
            "experiment2a_all_nn_scibert",
            "experiment2b_all_nn_steelbert",
            "experiment2c_all_nn_matscibert"
        ]

    elif args.custom:
        # 自定义参数运行
        if not args.embedding_type:
            logger.error("[ERROR] 自定义模式需要指定 --embedding_type")
            sys.exit(1)

        if not args.use_nn and not args.models:
            logger.error("[ERROR] 必须指定 --use_nn 或 --models")
            sys.exit(1)

        # 创建自定义配置
        custom_config = {
            "description": "自定义配置 / Custom configuration",
            "alloy_types": args.alloy_types,
            "exclude_alloys": args.exclude_alloys,
            "embedding_type": args.embedding_type,
            "use_composition_feature": args.use_composition_feature,
            "use_element_embedding": args.use_element_embedding,
            "use_process_embedding": args.use_process_embedding,
            "use_temperature": args.use_temperature,
            "models": args.models,
            "use_nn": args.use_nn,
            "cross_validate": args.cross_validate,
            "num_folds": args.num_folds,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "use_optuna": args.use_optuna,
            "n_trials": args.n_trials,
            "evaluate_after_train": args.evaluate_after_train,
            "evaluate_after_train": args.evaluate_after_train,
            "run_shap_analysis": args.run_shap_analysis,
            "n_repeats": args.n_repeats,
        }

        # 运行自定义配置
        logger.info("=" * 100)
        logger.info("运行自定义配置 / Running Custom Configuration")
        logger.info("=" * 100)

        results = run_batch_config("custom", custom_config, args.dry_run, args.max_workers, progress_manager, args.resume)

        # 打印总结
        logger.info("\n" + "=" * 100)
        logger.info("运行总结 / Summary")
        logger.info("=" * 100)

        for alloy_type, status in results.items():
            status_icon = "[OK]" if status == "success" else "[FAIL]" if status == "failed" else "[SKIP]"
            logger.info(f"{status_icon} {alloy_type}: {status}")

        success_count = sum(1 for s in results.values() if s == "success")
        total_count = len(results)
        logger.info(f"\n成功率: {success_count}/{total_count}")

        if success_count < total_count and not args.dry_run:
            sys.exit(1)

        return

    else:
        logger.error("[ERROR] 请指定运行模式: --list, --config, --all, --experiment1, --experiment2, 或 --custom")
        parser.print_help()
        sys.exit(1)

    # 运行所有指定的配置
    logger.info("=" * 100)
    logger.info("批量运行流水线 / Batch Run Pipeline")
    logger.info("=" * 100)
    logger.info(f"将运行 {len(run_configs)} 个配置")
    logger.info("=" * 100)

    all_results = {}
    start_time = datetime.now()

    for config_name in run_configs:
        config = BATCH_CONFIGS[config_name]
        results = run_batch_config(config_name, config, args.dry_run, args.max_workers, progress_manager, args.resume)
        all_results[config_name] = results

    end_time = datetime.now()
    duration = end_time - start_time

    # 打印总结
    logger.info("\n" + "=" * 100)
    logger.info("总体运行总结 / Overall Summary")
    logger.info("=" * 100)

    for config_name, results in all_results.items():
        config_desc = BATCH_CONFIGS[config_name]['description']
        logger.info(f"\n{config_desc}")
        logger.info(f"配置: {config_name}")

        for alloy_type, status in results.items():
            status_icon = "[OK]" if status == "success" else "[FAIL]" if status == "failed" else "[SKIP]"
            logger.info(f"  {status_icon} {alloy_type}: {status}")

    # 统计
    total_runs = sum(len(r) for r in all_results.values())
    total_success = sum(sum(1 for s in r.values() if s == "success") for r in all_results.values())

    logger.info("\n" + "=" * 100)
    logger.info(f"总体成功率: {total_success}/{total_runs}")
    logger.info(f"总耗时: {duration}")
    logger.info("=" * 100)

    if total_success < total_runs and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()



