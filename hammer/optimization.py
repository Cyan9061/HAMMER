
import concurrent.futures
import json
import random
import typing as T
import logging
import time

import optuna
import pandas as pd
from optuna.storages import BaseStorage

from hammer.configuration import cfg
from hammer.logger import logger
from hammer.optuna_helper import (
    get_completed_trials,
    get_knee_flow,
    get_sampler,
    recreate_with_completed_trials,
    seed_study,
    trial_exists,
    without_non_search_space_params,
)
from hammer.studies import StudyConfig

def run_trial(
    study: optuna.Study,
    objective: T.Callable,
    study_config: StudyConfig,
    components: T.List[str],
    block_name: str | None = None,
) -> T.Dict[str, float] | None:
    """顺序执行单个试验 - 支持单目标和双目标"""
    trial = study.ask()
    if block_name:
        trial.set_user_attr("block_name", block_name)
    try:
        # 🔑 检查是否为单目标优化
        if not study_config.optimization.objective_2_name:
            # 单目标优化
            obj_value = objective(trial, study_config, components)
            study.tell(trial, obj_value)
            return {
                study_config.optimization.objective_1_name: obj_value,
            }
        else:
            # 双目标优化
            obj_1, obj_2 = objective(trial, study_config, components)
            study.tell(trial, [obj_1, obj_2])
            return {
                study_config.optimization.objective_1_name: obj_1,
                study_config.optimization.objective_2_name: obj_2,
            }
    except optuna.TrialPruned:
        return None
    except Exception as e:
        logger.error(f"Trial failed with error: {str(e)}")
        return None

def user_confirm_delete(study_config: StudyConfig) -> bool:
    """用户确认删除研究"""
    study_name = study_config.name

    if cfg.optuna.noconfirm:
        logger.warning("noconfirm set; going to delete %s if it exists", study_name)
        return True

    try:
        confirm = input(
            f"Are you sure you want to overwrite study {study_name} if it exists? yes/no\n>>> "
        )
        if confirm != "yes":
            logger.warning(f"Cowardly refusing to delete study {study_name}")
            return False
        return True
    except (OSError, EOFError):
        # 后台运行时无法读取用户输入，自动跳过确认
        logger.warning("Running in background mode, automatically confirming study deletion for %s", study_name)
        return True

def save_study_config_to_db(study: optuna.Study, study_config: StudyConfig):
    """保存研究配置到数据库"""
    attrs = study_config.model_dump(mode="json")
    logger.info("Saving study config of %s to the database", study.study_name)
    for attr, value in attrs.items():
        study.set_user_attr(attr, value)

def get_study(study_config: StudyConfig) -> optuna.Study:
    """获取或创建研究"""
    study_name = study_config.name
    storage = cfg.database.get_optuna_storage()
    
    if study_config.reuse_study:
        logger.info(
            "Reusing study '%s' or creating new one", study_name
        )
        if study_config.recreate_study:
            recreate_with_completed_trials(study_config, storage)
    elif user_confirm_delete(study_config):
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            logger.info("Study '%s' deleted", study_name)
        except KeyError:
            logger.info(
                "Study '%s' does not exist, creating new", study_name
            )
    sampler = get_sampler(study_config)
    
    # 🔑 检查是否为单目标优化
    if True:#not study_config.optimization.objective_2_name:
        # 单目标优化：最大化F1得分
        logger.info("创建单目标优化study (最大化F1得分): %s", study_name)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=study_config.reuse_study,
            direction="maximize",
            sampler=sampler,
        )
    else:
        # 双目标优化
        logger.info("创建双目标优化study (最大化准确率，最小化成本): %s", study_name)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=study_config.reuse_study,
            directions=["maximize", "minimize"],
            sampler=sampler,
        )
    save_study_config_to_db(study, study_config)
    return study

def _get_user_attrs(row: pd.Series) -> T.Dict[str, T.Any]:
    """从行数据获取用户属性"""
    user_attrs = {}
    for key, value in row.items():
        if key.startswith("user_attrs_"):
            user_attrs[key.replace("user_attrs_", "")] = value
    return user_attrs

def initialize_from_study(
    src_config: StudyConfig,
    dst_config: StudyConfig,
    storage: str | BaseStorage | None = None,
    success_rate: float | None = None,
    src_df: pd.DataFrame | None = None,
) -> optuna.Study:
    """
    从另一个研究初始化研究
    """
    assert success_rate is None or src_df is None, (
        "Cannot use both success_rate and src_df"
    )

    storage = storage or cfg.database.get_optuna_storage()

    src_name = src_config.name
    dst_name = dst_config.name

    logger.info("Initializing study '%s' from study '%s'", dst_name, src_name)

    study = get_study(dst_config)

    if src_df is None:
        src_df = get_completed_trials(
            src_name, storage=storage, success_rate=success_rate
        )
    
    for _, row in src_df.iterrows():
        if not isinstance(row["user_attrs_flow"], str):
            logger.info(
                "Skipping invalid flow in source study '%s'", src_name
            )
            continue
            
        src_flow = json.loads(row["user_attrs_flow"])
        src_flow = without_non_search_space_params(src_flow, src_config)

        if dst_config.optimization.skip_existing and trial_exists(
            dst_name, src_flow, storage
        ):
            logger.warning(
                "Flow already exists in destination study '%s'", dst_name
            )
            continue

        distributions = dst_config.search_space.build_distributions(params=src_flow)
        obj1 = row["values_0"]
        
        # 🔑 检查是否为单目标优化
        if not dst_config.optimization.objective_2_name:
            # 单目标优化
            trial = optuna.create_trial(
                values=[obj1],
                params=src_flow,
                distributions=distributions,
                user_attrs=_get_user_attrs(row),
            )
        else:
            # 双目标优化
            obj2 = row["values_1"]
            trial = optuna.create_trial(
                values=[obj1, obj2],
                params=src_flow,
                distributions=distributions,
                user_attrs=_get_user_attrs(row),
            )

        study.add_trial(trial)
        logger.debug("Added trial with params: %s", trial.params)

    return study

class StudyRunner:
    """
    研究运行器 - 按块顺序执行优化
    """

    def __init__(
        self, 
        study_config: StudyConfig, 
        objective: T.Callable, 
        seeder: T.Callable
    ):
        self.study_config = study_config
        self.name = study_config.name
        self.objective = objective
        self.seeder = seeder
        self.seeder_timeout = study_config.optimization.seeder_timeout
        self.method = study_config.optimization.method
        self.blocks = study_config.optimization.blocks.copy()
        self.shuffle_blocks = study_config.optimization.shuffle_blocks
        self.num_trials = study_config.optimization.num_trials
        self.max_concurrent_trials = self.study_config.optimization.max_concurrent_trials
        self.raise_on_failed_trial = self.study_config.optimization.raise_on_failed_trial
        self.success_rate = self.study_config.optimization.pareto_eval_success_rate

    def run(self) -> optuna.Study:
        """
        按配置顺序执行优化过程
        """
        study_config = self.study_config
        study = get_study(study_config)

        # 使用线程池执行种子任务
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(seed_study, self.seeder, study_config)
            
            if self.seeder_timeout:
                logger.info("Seeding timeout: %i seconds", self.seeder_timeout)
            try:
                future.result(timeout=self.seeder_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning("Seeding timed out")

        logger.info("Starting sequential optimization")

        components = []
        trials_completed = 0
        
        while trials_completed < self.num_trials:
            if self.shuffle_blocks:
                logger.info("Shuffling optimization blocks")
                random.shuffle(self.blocks)
                
            for block in self.blocks:
                logger.info("Starting optimization block: %s", block.name)
                
                if self.method == "expanding":
                    logger.info("Expanding searchable components with block: %s", block.name)
                    components.extend(block.components)
                else:
                    logger.info("Using searchable components from block: %s", block.name)
                    components = block.components
                    
                # 计算当前块要运行的试验次数
                num_trials = min(block.num_trials, self.num_trials - trials_completed)
                trials_completed += num_trials
                
                # 顺序运行试验
                for i in range(num_trials):
                    logger.info("Running trial %d/%d in block %s", 
                               i+1, num_trials, block.name)
                    
                    try:
                
                        run_trial(
                            study=study,
                            objective=self.objective,
                            study_config=study_config,
                            components=components,
                            block_name=block.name
                        )

                    except Exception as e:
                        if self.raise_on_failed_trial:
                            raise
                        logger.error("Trial failed: %s", str(e))
                    
                    # 添加小延迟避免资源冲突
                    time.sleep(0.1)
                
                # 如果是knee方法，更新默认值
                if self.method == "knee":
                    flow = get_knee_flow(
                        study_config=study_config,
                        success_rate=self.success_rate,
                    )
                    defaults = {
                        key: value
                        for key, value in flow.items()
                        if key in components
                    }
                    logger.info(
                        "Knee flow defaults from block '%s': %s",
                        block.name,
                        defaults,
                    )
                    study_config.search_space.update_defaults(defaults)

        logger.info("Finished optimizing '%s'", self.name)
        return study

from hammer.flows import Flows
from hammer.optuna_helper import set_metrics
def get_flow_name(rag_mode: str):
    match rag_mode:
        # case "no_rag":
        #     return Flows.GENERATOR_FLOW.value.__name__
        case "rag":
            return Flows.RAG_FLOW.value.__name__
        # case "sub_question_rag":
        #     return Flows.LLAMA_INDEX_SUB_QUESTION_FLOW.value.__name__
        # case "react_rag_agent":
        #     return Flows.LLAMA_INDEX_REACT_AGENT_FLOW.value.__name__
        # case "critique_rag_agent":
        #     return Flows.LLAMA_INDEX_CRITIQUE_AGENT_FLOW.value.__name__
        # case "lats_rag_agent":
        #     return Flows.LLAMA_INDEX_LATS_RAG_AGENT.value.__name__
        case _:
            raise RuntimeError("Cannot identify flow")
        
def set_trial(
    trial: optuna.trial.FrozenTrial | optuna.trial.Trial,
    study_config: StudyConfig | None = None,
    params: dict[str, str | bool | int | float] | None = None,
    is_seeding: bool | None = None,
    metrics: T.Dict[str, float] | None = None,
    flow_json: str | None = None,
):
    if params:
        flow_name = get_flow_name(str(params["rag_mode"]))
        trial.set_user_attr("flow_name", flow_name)
    if study_config:
        trial.set_user_attr("dataset", study_config.dataset.name)
    if is_seeding is not None:
        trial.set_user_attr("is_seeding", is_seeding)
    if flow_json:
        trial.set_user_attr("flow", flow_json)
    if metrics:
        set_metrics(trial, metrics)