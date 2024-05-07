import os
import torch
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
# 从transformers.utils中导入logging和PaddingStrategy,用于日志记录和填充策略
from transformers.utils import logging, PaddingStrategy
# 从transformers.tokenization_utils_base中导入EncodedInput和BatchEncoding类, 这些类用于表示编码后的输入和批量编码
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding


class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path) # 3. 使用SentencePieceProcessor加载指定路径的SentencePiece模型

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()  # 4. 获取词表大小(vocabulary size)
        # 5. 获取BOS(开始)、EOS(结束)和PAD(填充)标记的ID
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()  # 6. 确保词表大小与SentencePiece模型的片段(piece)大小一致

        # 7. 定义一些特殊标记,如MASK、SOP(Sequence Start)和EOP(Sequence End)
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        # 8. 为每个特殊标记分配一个ID,并存储在字典中
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    # 9. tokenize方法用于将字符串分词为子词(subword)列表
    def tokenize(self, s: str):
        return self.sp_model.EncodeAsPieces(s)

    # 10. encode方法用于将字符串编码为ID序列
    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    # 11. decode方法用于将ID序列解码为字符串
    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    # 12. decode_tokens方法用于将子词列表解码为字符串
    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    # 13. convert_token_to_id方法用于将子词转换为ID
    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    # 14. convert_id_to_token方法用于将ID转换为子词
    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens or index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)


# 这是一个定制的Tokenizer类,用于处理ChatGLM-6B模型的输入
class ChatGLMTokenizer(PreTrainedTokenizer):
    # 1. vocab_files_names 是一个字典,指定了词表文件的名称
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    # 2. model_input_names 指定了模型输入所需的字段名称
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    # 3. __init__ 是构造函数,用于初始化Tokenizer
    def __init__(self, 
                 vocab_file, 
                 # 注意: 训练阶段与测试阶段的填充不同
                    # 训练阶段是tokenizer是一个个样本进行的, 总的来说是后pad
                    # 测试阶段是tokenizer是batch进行的, 总的来说是前pad和后pad一起
                 padding_side="left", 
                 clean_up_tokenization_spaces=False, 
                 **kwargs):
        self.name = "GLMTokenizer"    # 4. 设置Tokenizer的名称为"GLMTokenizer"

        self.vocab_file = vocab_file              # 5. 保存词表文件路径
        self.tokenizer = SPTokenizer(vocab_file)   # 6. 初始化底层的SentencePieceTokenizer
        self.special_tokens = {                    # 7. 定义一些特殊token及其对应的id
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }
        super().__init__(padding_side=padding_side, 
                         clean_up_tokenization_spaces=clean_up_tokenization_spaces, 
                         **kwargs)

    # 9. get_command 方法用于获取特殊token对应的id
    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def pad_token(self) -> str:    # 11. pad_token 属性返回填充token
        return "<unk>"

    @property
    def pad_token_id(self):         # 12. pad_token_id 属性返回填充token的id
        return self.get_command("<pad>")

    @property
    def eos_token(self) -> str:
        return "</s>"

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    # 16. get_vocab 方法返回词表(token到id的映射)
    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    # 17. _tokenize 方法使用底层的SentencePieceTokenizer对文本进行tokenize
    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    # 20. convert_tokens_to_string 方法将一个token列表转换为原始的字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    # 21. save_vocabulary 方法将词表保存到指定的目录
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    # 22. get_prefix_tokens 方法返回一个列表,包含一些特殊的前缀token
    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    # 23. build_prompt 方法根据给定的query和历史对话构建一个完整的prompt
    def build_prompt(self, query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    # 24. build_inputs_with_special_tokens 方法将输入的token ids序列与特殊token(如[CLS]、[SEP])拼接
    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(    # 25. _pad 方法用于对编码后的输入序列进行填充
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults, 
        # 注意: 训练阶段与测试阶段的填充不同
            # 训练阶段是tokenizer是一个个样本进行的, 总的来说是后pad
            # 测试阶段是tokenizer是batch进行的, 总的来说是前pad和后pad一起
        assert self.padding_side == "left"
        # 27. 获取输入序列的长度
        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)
        # 28. 如果padding_strategy为LONGEST,则将max_length设置为batch中最长序列的长度
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)
        # 29. 如果max_length不是pad_to_multiple_of的整数倍,则将其上调到最近的整数倍
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        # 30. 判断序列是否需要填充
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
        # 31. 如果attention_mask不存在,则初始化为全1
        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length
        # 32. 如果position_ids不存在,则初始化为序列位置
        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))
        # 33. 如果需要填充
        if needs_to_be_padded:
            difference = max_length - len(required_input)   # 计算需要填充的长度
            # 填充attention_mask
            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            # 填充position_ids
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            # 填充输入序列
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
