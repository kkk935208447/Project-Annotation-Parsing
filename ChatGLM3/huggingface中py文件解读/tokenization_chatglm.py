import json
import os
import re
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding


logger = logging.get_logger(__name__)

# SPTokenizer 类用于对文本进行分词(tokenization)和 token 与 ID 之间的转换
# 它利用 SentencePiece 技术对文本进行子词切分(subword segmentation)
# 这种方法能够有效地处理未登录词,并缩小词汇表的大小,在处理大型语料时十分有用
class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        # 加载 SentencePiece 模型,该模型用于执行分词和 token-ID 转换, SentencePiece 是一种用于文本编码的无损压缩技术,适用于处理大规模语料
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size() # 词汇表大小
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id() # 填充标记 ID,使用未知标记的 ID 作为填充
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size() # 断言词汇表大小与 get_piece_size() 方法返回的结果一致,这是一个检查,确保词汇表大小的计算方式正确
        # 定义一些特殊标记,如用于标识系统、用户和助手的角色标记, 这些特殊标记在对话模型中用于区分不同的角色
        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        # 添加其他特殊标记,如掩码标记和分句标记, 这些特殊标记在特定任务中会被使用,如掩码语言模型等
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        # 初始化两个字典,用于存储特殊标记及其对应的 ID
        self.special_tokens = {}
        self.index_special_tokens = {}
        # 为每个特殊标记分配一个 ID
        # 这些 ID 从词汇表大小开始递增分配,确保不会与已有词汇重复
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
        # 构建一个正则表达式,用于匹配所有特殊标记,这个表达式在分词时用于识别和处理特殊标记
        self.role_special_token_expression = "|".join([re.escape(token) for token in special_tokens]) # for apply_chat_template

    def tokenize(self, s: str, encode_special_tokens=False):
        # 对输入字符串进行分词(tokenization),如果 encode_special_tokens 为 True,则会对特殊标记进行特殊处理
        if encode_special_tokens:
            last_index = 0
            t = []
            # 使用正则表达式查找所有特殊标记
            for match in re.finditer(self.role_special_token_expression, s):
                # 对特殊标记之前的文本进行普通分词
                if last_index < match.start():
                    t.extend(self.sp_model.EncodeAsPieces(s[last_index:match.start()]))
                # 将特殊标记直接添加到结果列表中
                t.append(s[match.start():match.end()])
                last_index = match.end()
            # 对特殊标记之后的文本进行普通分词
            if last_index < len(s):
                t.extend(self.sp_model.EncodeAsPieces(s[last_index:]))
            return t
        else:
            # 如果不需要特殊处理特殊标记,则直接进行普通分词
            return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        # 将输入字符串编码为一个 ID 列表
        # 可以选择是否在开头和结尾添加 BOS 和 EOS 标记的 ID
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        # 将一个 ID 列表解码为原始字符串
        # 会根据 ID 是否对应特殊标记进行不同处理
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:  # 如果是特殊标记的 ID,则直接添加对应的字符串
                    text += self.sp_model.decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else: # 如果是普通 token 的 ID,则先缓存起来
                buffer.append(token)
        if buffer: # 将最后剩余的 buffer 中的 token 解码并添加到结果中
            text += self.sp_model.decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str: # 将一个 token 列表直接解码为原始字符串
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. 
            # 将一个 token 转换为对应的 ID
            # 如果是特殊标记,则从特殊标记字典中查找
            # 否则使用 SentencePiece 模型进行转换
        """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab.
            # 将一个 ID 转换为对应的 token
            # 如果是特殊标记的 ID,则从特殊标记字典中查找
            # 否则使用 SentencePiece 模型进行转换
        """
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0 or index > self.sp_model.vocab_size():
            return ""
        return self.sp_model.IdToPiece(index)

# ChatGLMTokenizer是一个用于对文本进行分词(将文本切分为token序列)和编码(将token转换为对应的数字ID)的类
# 它继承自huggingface的PreTrainedTokenizer基类,针对ChatGLM这种基于Transformer的大型语言模型进行了专门的设计和优化
class ChatGLMTokenizer(PreTrainedTokenizer):
    # vocab_files_names是一个字典,指定了用于存储词汇表的文件名,在这里,词汇表文件名为"tokenizer.model"
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    # model_input_names是一个列表,指定了语言模型在运行时需要的输入张量名称,在Transformer等序列模型中,通常需要输入token的数字ID(input_ids)、注意力掩码(attention_mask)和位置编码(position_ids)
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
        self,
        vocab_file,     # 词汇表文件的路径
        padding_side="left",  # 指定填充(pad)的方向,left表示左侧填充
        clean_up_tokenization_spaces=False,  # 是否清除分词后产生的空格
        encode_special_tokens=False,  # 是否对特殊标记(如<pad>、<bos>等)进行编码
        **kwargs
    ):
        self.name = "GLMTokenizer"      # 设置分词器的名称为"GLMTokenizer"
        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file) # 从词汇表文件中加载SPTokenizer对象
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<unk>": self.tokenizer.pad_id,   # pad 与 unk 的id相同
            "<pad>": self.tokenizer.pad_id
        }
        self.encode_special_tokens = encode_special_tokens

        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

    # get_command方法用于获取指定token对应的ID, 如果token是已定义的特殊标记,直接返回其对应的ID, 否则检查token是否为SPTokenizer中定义的其他特殊标记,如果是则返回其ID
    def get_command(self, token): 
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    # 以下是一些属性方法,用于获取特殊标记的字符串形式及其对应ID
    # 属性方法使用@property装饰器,可以像访问属性一样使用,无需调用普通方法
    @property
    def unk_token(self) -> str:
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<unk>"))

    @property
    def pad_token(self) -> str:
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<pad>"))

    @property
    def eos_token(self) -> str:
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<eos>"))

    @property
    def unk_token_id(self) -> int:
        return self.get_command("<unk>")

    @property
    def pad_token_id(self) -> int:
        return self.get_command("<pad>")

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    # 以下setter方法用于设置特殊标记,但实际上并不支持修改
    # 会打印警告信息,提示使用默认的特殊标记
    @unk_token.setter
    def unk_token(self, value):
        logger.warning("Setting unk_token is not supported, use the default one.")

    @pad_token.setter
    def pad_token(self, value):
        logger.warning("Setting pad_token is not supported, use the default one.")

    @eos_token.setter
    def eos_token(self, value):
        logger.warning("Setting eos_token is not supported, use the default one.")

    @property
    def vocab_size(self):   # vocab_size属性返回词汇表的大小, @property 装饰器可以让该方法成为一个属性
        return self.tokenizer.n_words

    def get_vocab(self):
        """ Returns vocab as a dict 
            # get_vocab方法返回整个词汇表,以字典的形式
            # 键为token的字符串形式,值为对应的数字ID   
        """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs): # 将文本切分为一个个token, encode_special_tokens指定是否对特殊标记进行编码
        return self.tokenizer.tokenize(text, encode_special_tokens=self.encode_special_tokens)

    def _convert_token_to_id(self, token):  # 将给定的一个token(字符串形式)转换为对应的数字ID 
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):  # 将给定的一个ID转换为对应的token(字符串形式)
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:  # 将一系列token转换为原始的字符串文本
        return self.tokenizer.decode_tokens(tokens)

    # save_vocabulary方法用于将词汇表保存到指定的目录路径
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
        # 读取原始的词汇表文件内容
        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()
        # 将内容写入到新的词汇表文件中
        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):  # 自定义的方法, get_prefix_tokens返回用于构建输入序列的前缀标记
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    # 自定义的方法, build_single_message方法构建单条消息的token序列
    # 根据给定的角色(role)、元数据(metadata)和消息内容(message)构建对应的token序列
    def build_single_message(self, role, metadata, message): 
        assert role in ["system", "user", "assistant", "observation"], role
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message)
        tokens = role_tokens + message_tokens
        return tokens

    # build_chat_input方法构建对话输入
    # 包括查询(query)和历史对话记录(history),以及指定的角色(role)
    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(self.build_single_message(role, "", query))
        input_ids.extend([self.get_command("<|assistant|>")])
        # 最后对构建好的输入进行编码,并将其转换为张量的形式,方便输入模型
        return self.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)

    # build_inputs_with_special_tokens方法用于构建模型输入
    # 将给定的token ID序列(token_ids_0)和可选的第二个序列(token_ids_1,用于序列对任务)
    # 添加特殊标记,如[CLS]、[SEP]等
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    # _pad方法对编码后的输入进行填充操作, 将输入序列填充至指定的最大长度,以满足模型输入的要求
    def _pad(
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
        # Load from model defaults
        assert self.padding_side == "left"  # 确保填充方向为左侧填充

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)
        # 如果max_length不是指定长度的整数倍,则向上取整
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present. 初始化注意力掩码和位置编码,如果不存在则创建
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]

            # 使用pad_token_id对输入序列进行填充 
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
        return encoded_inputs
