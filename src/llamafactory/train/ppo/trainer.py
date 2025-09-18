# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# 本文件实现了基于 TRL 的 PPO 训练器封装，并与 HuggingFace Trainer/Accelerate
# 框架进行集成。核心思想：
# - 复用基础 CausalLM 基座 + ValueHead；
# - 在 LoRA/OFT 模式下通过切换适配器实现“策略/奖励/默认”三种头部的共享；
# - 以最少显存开销完成“生成→打分→PPO更新”的完整训练闭环。
#
# 重要概念说明：
# - 策略模型（policy）：参与生成与更新的主模型（含 value head）。
# - 参考模型（ref）：用于 KL 正则的参考分布，通常冻结。
# - 奖励模型（reward）：给定 query/response 计算 reward（标量），可为 LoRA。
# - 值函数（value head）：在策略模型上方追加，用以估计每个 token 的 value。
# - 缓冲区（ppo_buffer_size）：单轮生成的样本组数，影响 batch 规模与吞吐。
#
# 性能/显存要点：
# - 生成阶段消耗显存较多，使用 autocast(bf16/fp16) + 关闭 KV cache 减少占用；
# - LoRA/OFT 奖励模型通过切换适配器避免加载“另一份完整基座权重”；
# - Trainer/Accelerate 负责并行与设备放置，DeepSpeed/FSDP 时逻辑做了相应适配。
#
# 若需调优吞吐：优先调小 max_new_tokens / per_device_train_batch_size；
# 若需调优稳定性：增大 gradient_accumulation_steps / ppo_epochs，适当降学习率。
#
# 参考实现来源：
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer.

    扩展自 TRL 的 PPOTrainer 与 HF 的 Trainer：
    - 负责根据 YAML 中的参数构造 PPOConfig/优化器/调度器；
    - 处理 DeepSpeed/FSDP 等分布式策略下的设备放置与日志；
    - 将 LLaMA-Factory 的 generate 配置与 Tokenizer/Processor 融合；
    - 提供自定义的训练循环 ppo_train（与 HF Trainer 的 inner loop 类似）。
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        # PPO 当前不支持在 Trainer 级别传入 eval_dataset，避免误用
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        # 构造 PPOConfig：其中 batch_size= backward_batch_size * ppo_buffer_size * world_size（见下文计算）
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,  # 打开后可减少频繁的显存碎片
            target=finetuning_args.ppo_target,  # KL 目标或优势裁剪目标等
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # DeepSpeed 集成：配置 DDP handler，并在启用时禁用外部 logger（兼容限制）
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # 训练步数：优先使用 max_steps；否则基于数据集长度与 world_size 计算
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        # 初始化 TRL 的 PPOTrainer 基类
        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        # 保存重要引用，供训练循环使用
        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        # 生成配置：由 YAML 中 generate 段落转为 HF GenerationConfig
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        # HF Trainer 运行时状态与回调初始化
        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            # full 模式下，reward_model 为独立模型，需要做设备放置/加速准备
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        # 固定 ValueHead 的一些保存/还原行为（避免被意外覆盖）
        self.add_callback(FixValueHeadModelCallback)

        # 若存在 Processor（多模态等），保存到权重目录便于推理复现
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # BAdam 优化器（可选）：如启用则替换梯度裁剪行为
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.

        训练主循环说明：
        - 以 dataloader 为基础，每个 step 进行一次“生成→奖励→PPO 更新”；
        - total_train_batch_size = per_device * grad_accum * ppo_buffer_size * world_size；
        - max_steps > 0 时以步数为准；否则按 epoch × dataloader 长度展开。
        - 训练中每隔 logging_steps 输出 loss/reward 等指标，并定期保存 checkpoint。
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                # 一个 epoch 结束后重新迭代数据
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # 1) 生成：eval 模式 + 右侧 padding，减少生成端 pad 干扰
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []

            # 由于 PPO 使用缓冲区，按 mini_batch_size 切分当前 batch 进行多次生成以填满缓冲
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch = {
                    "input_ids": batch["input_ids"][idx : idx + self.config.mini_batch_size],
                    "attention_mask": batch["attention_mask"][idx : idx + self.config.mini_batch_size],
                }
                mini_batch_queries, mini_batch_responses = self.get_inputs(mini_batch)
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # 2) PPO 更新：切回 train 模式，计算 PPO loss 并反传/更新
            self.model.train()
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            # 3) 日志记录：尝试保存 query/response 文本，可能因为分布式而失败，故 try/except
            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            # 4) 步进与定期保存
            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        # 允许外部注入自定义优化器；否则基于权重衰减划分参数组并构建优化器
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        # 允许外部注入自定义调度器；默认使用 HF 提供的 scheduler 工厂
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: dict[str, "torch.Tensor"]) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries.

        生成阶段说明（PPO 开销主要来源）：
        - 为每个 prompt 生成 response，用于后续奖励打分与 PPO 更新。
        - 这里在单样本（且梯度累积>1）的场景下，会先移除左侧 pad，避免将 pad token 作为上下文参与生成。
        - 生成时切换到 autocast（bf16/fp16）与 unwrap 的模型，以减少显存并提升推理速度。
        - 生成结果再与原 query 对齐切分，构造 queries/responses 返回，后续用于计算 logprobs/masks 等。
        """
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            # 计算首个非 pad 位置索引，裁掉左侧 pad，避免 pad 干扰生成与长度统计
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        # 使用 unwrap 后的模型进行 generate，保证在分布式/DeepSpeed 下的推理正确
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            # 基于 self.generation_config（由 YAML 的 generate 段落填充）进行生成
            generate_output: torch.Tensor = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        # 将生成的 response 与原 query 对齐切分，并移除填充位
        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            # 找到 query 的实际起始位置（去掉左侧 pad）
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            # 统计 response 的实际有效长度（考虑到 eos/pad 的不同设定）
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            # 去掉 query 左 pad；去掉 response 右 pad
            queries.append(query[i, query_start_index:])
            responses.append(response[i, :response_length])

        return queries, responses

    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        奖励计算说明：
        - 当 `reward_model_type` 为 "lora"/"oft" 时，通过 `replace_model` 将当前策略模型切换到对应适配器头（"reward"），
          以复用同一基座权重，仅注入 RM 的 LoRA 适配器与 ValueHead，从而节省显存。
        - 当类型为 "full" 时，直接使用独立的 `self.reward_model` 进行前向。
        - 最后从最后一个有效 token（mask 的尾部）处提取 value 作为标量奖励。
        - 返回的奖励为 fp32 张量，并在 CPU 上进行后处理，避免不必要的 GPU 占用。
        """
        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type in ["lora", "oft"]:
            # 将当前模型切换到 RM 适配器（共享基座，仅切换 LoRA 权重与 value head）
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        # 关闭缓存、保持 autocast，以减少显存并兼容 bf16
        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type in ["lora", "oft"]:
            # 计算完奖励后恢复回默认策略适配器
            replace_model(unwrapped_model, target="default")

        # 取每条样本最后一个有效 token 的 value 作为奖励标量
        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        前向计算说明：
        - 按 mini_batch_size 切分前向，降低峰值显存；
        - 关闭 use_cache，避免 KV cache 堆积；
        - 根据 query/response 边界构造 masks，仅在响应区域计算 logprobs/values；
        - 可选返回 logits（通常会关闭以减少显存）。
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            # 取当前子批次的输入张量与对应的 query/response 片段
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            # 计算下一个 token 的条件对数概率；根据 attention_mask 构造响应区域的有效掩码
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                # start 为 response 开始前的 query 末尾位置；end 为 response 末尾位置
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        # 聚合各子批次结果
        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        保存权重说明：
        - 在 DeepSpeed/FSDP 下优先通过 accelerator.get_state_dict() 获取权重；
        - 如 stage3 设定不允许 16bit 聚合保存，则落盘完整 checkpoint 并提示使用 zero_to_fp32 恢复；
        - 单卡/非并行场景下直接 unwrap 后保存 state_dict。
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
