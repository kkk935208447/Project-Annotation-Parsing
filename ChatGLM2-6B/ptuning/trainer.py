# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
这个 Trainer 类的用途,可以轻松地训练或微调一个 Hugging Face Transformers 模型。
"""
import os
from typing import Optional
from transformers import Trainer

import torch
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"

# 1. 定义一个名为 PrefixTrainer 的子类,继承自 transformers.Trainer 类。
#    这个类用于训练和微调具有前缀编码的模型。
class PrefixTrainer(Trainer):
    def __init__(self, *args, 
                 save_changed=False,  #    save_changed 参数用于控制是否仅保存模型的前缀编码部分。
                 **kwargs):           #    *args 和 **kwargs 用于传递给父类 Trainer 的其他初始化参数。
        self.save_changed = save_changed
        super().__init__(*args, **kwargs)

    # 重写父类 Trainer 中的 _save 方法,以实现自定义的模型保存策略。这种方法重写机制允许子类针对特定需求,对父类方法进行细节级别的定制和扩展。
    def _save(self, output_dir: Optional[str] = None, 
              state_dict=None):
        
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                # 使用 unwrap_model 提取出底层的 PreTrainedModel 实例
                # unwrap_model 函数是 transformers.modeling_utils 模块中定义的一个工具函数,它的作用是从给定的模型中提取出底层的 PreTrainedModel 实例。
                # 在 transformers 框架中,模型可能被包裹在其他容器类中,比如 torch.nn.DataParallel 或 torch.nn.parallel.DistributedDataParallel。这些容器类的目的是支持数据并行训练,通过将模型复制到多个 GPU 上并行运行来加速训练过程。
                # 但是,当我们需要保存模型时,我们通常只需要保存底层的 PreTrainedModel 实例,而不需要保存这些容器类。unwrap_model 函数就是用来从这些容器类中提取出底层的 PreTrainedModel 实例的。
                # 在 PrefixTrainer 类的 _save 方法中,我们使用了 unwrap_model 函数来处理这种情况
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                # 这里保存时参数
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # 6. 如果模型是 PreTrainedModel 的实例:
        #    如果 self.save_changed 为 True,则仅保存前缀编码部分;
        #    否则,保存整个模型。
        #    这种灵活的保存策略可以帮助用户根据实际需求进行权衡,在存储空间和模型性能之间进行权衡。
        else:
            if self.save_changed:
                print("Saving PrefixEncoder")
                state_dict = self.model.state_dict()
                filtered_state_dict = {}
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        filtered_state_dict[k] = state_dict[k]
                self.model.save_pretrained(output_dir, state_dict=filtered_state_dict)
            else:
                print("Saving the whole model")
                self.model.save_pretrained(output_dir, state_dict=state_dict)
         # 7. 如果存在分词器,则保存分词器。
        #    这确保了模型与分词器的一致性,便于后续使用。
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
