# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from trainer import PrefixTrainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


# 1. 定义了一个名为 Seq2SeqTrainer 的类,继承自 PrefixTrainer 类。
    #    Seq2SeqTrainer 是一个专门用于序列到序列(Seq2Seq)任务的训练器,如机器翻译、摘要生成等。
    #    通过继承 PrefixTrainer 类,它可以复用其中的前缀编码技术,同时针对 Seq2Seq 任务进行了一些扩展和优化。
class Seq2SeqTrainer(PrefixTrainer):

    # evaluate 方法是重写了 PrefixTrainer 类对应的方法,以支持序列到序列任务的评估和预测。
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        #    1) eval_dataset: 可选的评估数据集,如果未提供则使用默认数据集。
        #    2) ignore_keys: 可选的要忽略的键列表,用于过滤不需要计算的指标。
        #    3) metric_key_prefix: 指标键前缀,默认为 "eval",用于区分训练和评估指标。
        #    4) **gen_kwargs: 用于生成的其他关键字参数,如 max_length、num_beams 等。
        #    该方法的返回值是一个字典,包含评估过程中计算的各种指标及其对应的值。

        gen_kwargs = gen_kwargs.copy()  # 3. 创建 gen_kwargs 的副本,以避免修改原始参数。

        # 4. 如果 max_length 和 max_new_tokens 都没有提供,则使用默认的 generation_max_length 作为 max_length。
        #    max_length 参数用于限制生成序列的最大长度,可以防止生成过长的序列,从而节省计算资源。
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        # 5. 如果未提供 num_beams,则使用默认的 generation_num_beams 作为 Beam Search 的束宽。
        #    num_beams 参数控制了 Beam Search 算法中保留的候选序列数量,较大的值可以提高生成质量,但也会增加计算量。
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs  # 6. 将准备好的生成参数存储在 self._gen_kwargs 中,以便后续使用。
        # 7. 调用父类 PrefixTrainer 的 evaluate 方法,传递相应的参数。
        #    这个方法会在评估数据集上计算模型的各种指标,如准确率、困惑度等。
        #    通过继承和扩展 PrefixTrainer 类,Seq2SeqTrainer 可以复用其中的评估逻辑,同时添加了一些 Seq2Seq 任务特有的处理。
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    # predict 方法是重写了 PrefixTrainer 类对应的方法,以支持序列到序列任务的评估和预测。
    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        #    1) test_dataset: 测试数据集,用于生成模型预测结果。
        #    2) ignore_keys: 可选的要忽略的键列表,用于过滤不需要计算的指标。
        #    3) metric_key_prefix: 指标键前缀,默认为 "test",用于区分训练、评估和测试指标。
        #    4) **gen_kwargs: 用于生成的其他关键字参数,如 max_length、num_beams 等。
        #    该方法的返回值是一个 PredictionOutput 对象,包含损失、生成的令牌序列和标签等信息。

        gen_kwargs = gen_kwargs.copy()  # 9. 创建 gen_kwargs 的副本,以避免修改原始参数。
        # 10. 如果 max_length 和 max_new_tokens 都没有提供,则使用默认的 generation_max_length 作为 max_length。
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        # 11. 如果未提供 num_beams,则使用默认的 generation_num_beams 作为 Beam Search 的束宽。
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs  # 12. 将准备好的生成参数存储在 self._gen_kwargs 中,以便后续使用。

        # 13. 调用父类 PrefixTrainer 的 predict 方法,传递相应的参数。
        #     这个方法会在测试数据集上进行预测,生成模型的输出序列。
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    # prediction_step 方法是重写了 PrefixTrainer 类对应的方法,以支持序列到序列任务的评估和预测。
    # 14. 定义了 prediction_step 方法,用于执行单步预测。
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        #     1) model: 用于预测的模型对象。
        #     2) inputs: 包含输入数据的字典,如输入序列、注意力掩码等。
        #     3) prediction_loss_only: 一个布尔值,指示是否仅计算预测损失。
        #     4) ignore_keys: 可选的要忽略的键列表,用于过滤不需要计算的指标。
        #     该方法的返回值是一个包含损失、生成令牌和标签的元组。

         # 15. 如果未启用 predict_with_generate 或仅需要预测损失,则调用父类的 prediction_step 方法进行普通预测。
        #     predict_with_generate 是一个标志,用于指示是否使用生成式预测,如机器翻译、文本生成等。
        #     如果未启用该标志或仅需要计算损失,则使用普通的预测步骤,通常用于分类、回归等任务。
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs  # 16. 检查输入数据是否包含标签,将结果存储在 has_labels 变量中。
        inputs = self._prepare_inputs(inputs)   # 17. 调用 _prepare_inputs 方法对输入数据进行预处理,如填充、编码等,以确保输入数据的格式符合模型的要求。

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()   # 18. 创建 self._gen_kwargs 的副本,以避免修改原始参数。
        # 19. 如果 max_length 和 max_new_tokens 都没有提供,则使用模型配置中的 max_length 作为默认值。
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        # 20. 如果未提供 num_beams,则使用模型配置中的 num_beams 作为 Beam Search 的束宽。
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        # 21. 根据是否启用了 DeepSpeed ZeRO-3 优化,设置 default_synced_gpus 的值。
        #     DeepSpeed 是一种用于加速大型模型训练的库,ZeRO-3 是其中的一种优化策略,可以通过模型并行化和优化状态分片来减少内存占用。
        #     synced_gpus 参数用于控制是否在多个 GPU 之间同步张量,以减少内存占用并支持更大的模型。
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (         # 22. 设置 synced_gpus 参数,如果未提供则使用默认值 default_synced_gpus。
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # 23. 如果输入数据中包含注意力掩码,则将其添加到生成参数中。
        #     注意力掩码用于指示哪些位置应该被忽略或关注,可以提高模型的效率和性能。
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        # 24. 如果输入数据中包含位置 ID,则将其添加到生成参数中。
        #     位置 ID 用于表示输入序列中每个令牌的位置信息,对于一些位置编码模型很重要。
        if "position_ids" in inputs:
            gen_kwargs["position_ids"] = inputs.get("position_ids", None)
        # 25. 如果输入数据中包含全局注意力掩码,则将其添加到生成参数中。
        #     全局注意力掩码用于控制自注意力机制中的计算范围,可以提高计算效率。
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        # 26. 根据模型的输入名称,准备生成输入。
        #     对于编码器-解码器模型,编码器和解码器可能使用不同的输入名称,因此需要根据具体情况选择合适的输入名称。
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        gen_kwargs["input_ids"] = generation_inputs  # 27. 将准备好的生成输入添加到生成参数中。
        # 28. 使用准备好的生成参数,调用模型的 generate 方法生成令牌序列。
        generated_tokens = self.model.generate(**gen_kwargs)
        # 29. 从生成的令牌序列中去除输入部分,保留真正生成的部分。
        #     这一步是为了确保生成的结果不包含原始输入,只包含模型生成的新令牌序列。
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

        # in case the batch is shorter than max length, the output should be padded
        # 30. 如果生成的序列长度小于指定的最大长度,则使用 _pad_tensors_to_max_len 方法将序列填充到最大长度。
        #     这一步是为了确保生成的序列长度一致,便于后续的批处理和计算。
        #     填充操作会在序列末尾添加特殊的填充令牌,以达到指定的最大长度。
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        loss = None  # 31. 初始化损失值为 None。

        if self.args.prediction_loss_only:
            # 32. 如果仅需要预测损失,则直接返回一个包含损失、生成令牌和标签的元组,其中生成令牌和标签为 None。
            #     在某些情况下,用户可能只关心模型的预测损失,而不需要生成实际的令牌序列,这可以节省计算资源。
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None
        
        return (loss, generated_tokens, labels)

    # 定义的私有辅助函数, 没有重写父类
    def _pad_tensors_to_max_len(self, tensor, max_length):
        # 35. 定义一个名为 _pad_tensors_to_max_len 的私有方法,用于将张量填充到指定的最大长度。
        #     该方法接受两个参数:
        #     - tensor: 需要填充的张量
        #     - max_length: 目标最大长度

        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            # 37. 如果分词器不存在,则从模型配置中获取 pad_token_id。
        #     如果配置中也没有设置 pad_token_id,则引发 ValueError 异常。
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        # 38. 创建一个新的张量,其形状为 (batch_size, max_length),初始值为填充令牌 ID。
        #     首先使用 torch.ones 创建一个全为 1 的张量,然后将其乘以填充令牌 ID,得到一个填充值为指定令牌 ID 的张量。
        #     为了确保新张量与原始张量具有相同的数据类型和设备,使用了 tensor.dtype 和 tensor.device。
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        # 39. 将原始张量复制到新张量的前部。
        #     使用切片操作将原始张量的值复制到新张量的前部,从而实现了填充操作。
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor  # 该方法的返回值是一个新的张量,长度为指定的最大长度,前部是原始张量的值,后部是填充值。
