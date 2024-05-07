from typing import Any, Dict, List
import torch
# TODO ++ 新增代码
import numpy as np
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq


class Collator(object):
    def __init__(self, tokenizer, max_seq_length, ignore_index):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch) -> Dict[str, Any]:
        raise ImportError


class PretrainCollator(Collator):
    def __call__(self, batch: List[str]) -> Dict[str, Any]:
        inputs = self.tokenizer(
            batch, return_tensors='pt', max_length=self.max_seq_length,
            truncation=True, padding='max_length'
            # padding=True,
            # add_special_tokens=False
        )
        input_ids = inputs.input_ids
        # 将pad_token_id替换为-100
        labels = torch.where(input_ids != self.tokenizer.pad_token_id, input_ids, self.ignore_index)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': labels
        }
        return inputs

# # NOTE 弃用
# class EvalCollator(Collator):
#     def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#         input_ids = [x['input_ids'] for x in batch]
#         labels = [x['labels'] for x in batch]

#         input_ids = torch.tensor(input_ids, dtype=torch.long)
#         labels = torch.tensor(labels, dtype=torch.long)
#         inputs = {
#             'input_ids': input_ids,
#             'labels': labels
#         }
#         return inputs

# # NOTE 弃用
# class SFTCollator(Collator):

#     def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#         input_ids = []
#         attention_mask = []
#         labels = []

#         for x in batch:
#             input_ids.append(x['input_ids'])
#             attention_mask.append(x['attention_mask'])
#             labels.append(x['labels'])

#         # 将list转换为tensor，得到最终的的模型输入
#         input_ids = torch.tensor(input_ids, dtype=torch.long)
#         attention_mask = torch.tensor(attention_mask, dtype=torch.long)
#         labels = torch.tensor(labels, dtype=torch.long)
#         inputs = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'labels': labels
#         }
#         return inputs


# TODO ++ 新增代码
# 输入未pad的input_ids labels 等, 输出是pad后的input_ids (attention_mask 也可能有 position_ids) 和 labels
class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        output_ids = (  # 从 features 中提取 output_ids,如果不存在则设置为 None
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)  # 计算 output_ids 的最大长度
            # 如果指定了 pad_to_multiple_of,则将最大长度向上取整到该值的整数倍
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
             # 遍历 features,为每个 output_ids 填充至最大长度, 右填充, 这是为了 eval/ predict
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        # 调用父类 _DataCollatorForSeq2Seq 的 __call__ 方法,返回处理后的特征
        return super().__call__(features, return_tensors)
