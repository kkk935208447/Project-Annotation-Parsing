# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# 1. 导入一些标准Python库,用于处理JSON、文件路径、时间等操作。
# 2. 从 typing 库导入一些类型注解,用于增强代码的可读性和类型安全性。
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
# 3. 导入 PyTorch 库,用于构建和训练神经网络模型。
# 4. 从 torch.nn.functional 导入激活函数等,这是PyTorch中提供的函数接口。
import torch
import torch.nn.functional as F
# 5. 从 fairscale 库导入一些函数,用于初始化模型并行。
# 6. 模型并行是一种分布式训练技术,可以将大型模型分割到多个GPU上,从而减少单个GPU的内存压力。
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
# 7. 从 llama 库导入模型和分词器相关的类。
# 8. llama 是一个基于 Transformer 的大型语言模型,通常用于自然语言处理任务。
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
# 9. 定义一个字面量类型 Role,用于表示对话中的角色,包括系统、用户和助手。
Role = Literal["system", "user", "assistant"]

# 10. 定义一个类型化字典 Message,用于表示对话中的单个消息,包括角色和内容。
# 11. TypedDict 是 Python 3.8 引入的一种新的数据类型,它继承自 dict,但具有额外的类型约束。
class Message(TypedDict):
    role: Role
    content: str

# 12. 定义一个类型化字典 CompletionPrediction,用于表示语言模型的完形填空预测结果。
# 13. total=False 表示这个类型化字典中的字段不是必需的,可以有缺失值。
class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

# 14. 定义一个类型化字典 ChatPrediction,用于表示语言模型的对话预测结果。
# 15. 它包含一个 Message 类型的 generation 字段,以及可选的 tokens 和 logprobs 字段。
class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

# 16. 定义一个类型别名 Dialog,表示由多个 Message 组成的对话列表。
Dialog = List[Message]
# 17. 定义一些特殊标签,用于在输入数据中标记指令、系统消息等。
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# 18. 将特殊标签存储在一个列表中,并定义一个错误信息,用于处理包含特殊标签的不安全提示。
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        # 1. 这是一个静态方法,用于构建 Llama 对象,加载预训练模型和分词器。
        # 2. 它接受一些参数,如检查点文件目录、分词器文件路径、最大序列长度、最大批量大小和模型并行大小。
        # 3. 返回值是一个初始化好的 Llama 对象,包含加载的模型和分词器。
        # 4. 如果指定的目录中没有检查点文件,或者模型并行大小与检查点文件数量不匹配,会引发 AssertionError。

        
        # 5. 如果分布式进程组还没有初始化,则使用 NCCL 后端初始化它。
        # 6. 分布式训练可以在多台机器或多个 GPU 上并行训练模型,提高训练效率。
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        # 7. 如果模型并行还没有初始化,则根据环境变量或提供的值初始化模型并行。
        # 8. 模型并行是一种将大型模型分割到多个 GPU 上训练的技术,可以减少单个 GPU 的内存压力。
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        # 9. 获取本地进程的等级,并将 CUDA 设备设置为该等级。
        # 10. 在分布式训练中,每个进程都会被分配一个本地等级,用于确定它在当前节点上的位置。
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        # 11. 在所有进程中设置相同的随机种子,以确保可复现性。
        torch.manual_seed(seed)

         # 12. 如果本地等级大于 0,则将标准输出重定向到 /dev/null,以避免多个进程的输出混淆。
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")
        # 13. 获取检查点文件路径,并加载相应的检查点文件。
        # 14. 检查点文件包含了预训练模型的权重等信息,用于恢复模型状态。
        # 15. 根据模型并行等级加载对应的检查点文件。
        # 16. 确保检查点文件的数量与模型并行大小匹配,否则会引发异常。
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(checkpoints), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # 17. 从 params.json 文件中加载模型参数。
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        # 18. 创建 ModelArgs 对象,包含了最大序列长度、最大批量大小等参数。
        # 19. 创建 Tokenizer 对象,用于文本编码和解码。
        # 20. 设置词汇表大小为分词器的词汇量。
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        # 21. 将 PyTorch 默认张量类型设置为 FP16 (半精度浮点数),以节省内存。
        # 22. 创建 Transformer 对象,即语言模型的主要组件。
        # 23. 从检查点文件中加载模型参数,恢复模型状态。
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        # 24. 打印加载模型所用的时间。
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        # 25. 返回一个初始化好的 Llama 对象,包含加载的模型和分词器。
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        # 26. 这是 Llama 类的构造函数,接受预先初始化的模型和分词器对象。
        # 27. 将模型和分词器对象保存为类的属性,以便后续使用。
        self.model = model
        self.tokenizer = tokenizer


    # 26-53. 这个方法用于根据给定的提示生成文本序列。
    # 它首先对输入进行预处理,包括检查批量大小、计算提示长度、创建存储张量等。
    # 然后,它使用一个循环,逐步生成序列:
    #   1. 对当前输入进行前向传播,获取 logits 输出。
    #   2. 根据温度值和 top-p 值,对 logits 进行采样或贪婪搜索,获取下一个令牌。
    #   3. 将下一个令牌添加到序列中,但保留提示部分不变。
    #   4. 如果需要计算对数概率,则根据 logits 输出计算交叉熵损失作为对数概率。
    #   5. 更新是否已经生成结束符的标记,如果所有序列都已生成,则退出循环。
    # 最后,它根据设置对生成的序列进行后处理,包括截断到最大长度、移除结束符后的部分等。
    # 此方法使用了核采样(nucleus sampling)技术,通过控制温度值和 top-p 值来引入可控的随机性,生成更自然、更多样化的文本。
    # 如果设置了 logprobs 标志,它会计算每个生成令牌的对数概率,用于评估模型的置信度。
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        # 28. 这是一个实例方法,用于根据给定的提示生成文本序列。
        # 29. 它接受一个列表,每个元素是一个整数列表,表示一个被分词后的提示。
        # 30. 还接受其他参数,如最大生成长度、温度值、top-p 值、是否计算对数概率以及是否包含提示令牌。
        # 31. 返回一个元组,包含生成的令牌序列和对应的对数概率(如果计算的话)。
        # 32. 这个方法使用了核采样(nucleus sampling)技术,通过控制温度值和 top-p 值来引入可控的随机性。
        # 33. 如果设置了 logprobs 标志,它会计算每个生成令牌的对数概率。

        # 34. 获取模型参数对象,并检查批量大小是否超过了最大批量大小。
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # 35. 计算提示的最小长度和最大长度,并检查最大长度是否超过了模型允许的最大序列长度。
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        # 36. 计算总长度,即生成序列的最大长度加上提示的最大长度,但不能超过模型允许的最大序列长度。
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        # 37. 获取填充令牌的 ID,并创建一个填充了填充令牌的张量,用于存储生成的序列。
        # 38. 将提示令牌复制到张量的前面部分。
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        # 39. 如果需要计算对数概率,则创建一个与序列张量形状相同的零张量,用于存储对数概率。
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        # 40. 初始化一些辅助变量,用于记录当前位置、是否已经生成结束符以及输入文本的掩码。
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        # 41. 如果提示长度等于总长度,则直接对提示进行前向传播,并计算交叉熵损失作为对数概率的初始值。
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            # 42. 对输入序列进行前向传播,获取当前位置的 logits 输出。
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # 43. 根据温度值和 top-p 值,对当前位置的 logits 进行采样或贪婪搜索,获取下一个令牌。
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            # 44. 将下一个令牌填入序列张量中,但如果当前位置是提示部分,则保留原始令牌不变。
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # 45. 如果需要计算对数概率,则对当前位置的 logits 输出计算交叉熵损失,作为对数概率的值。
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            # 46. 更新是否已经生成结束符的标记,如果所有序列都已经生成结束符,则退出循环
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                # 47. 如果需要计算对数概率,则从对数概率张量中提取与当前生成序列相应的部分。
                # 48. 这里将生成序列的起始位置设置为 0 (如果要回显提示)或提示长度 (如果不要回显提示)。
                # 49. 同时将生成序列的长度限制为提示长度加上最大生成长度。
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            # 50. 如果生成序列中包含结束符令牌,则截断序列,只保留结束符之前的部分。
            # 51. 如果计算了对数概率,也将对数概率列表截断到结束符之前。
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            # 52. 将处理后的令牌序列和对数概率(如果计算了)添加到输出列表中。
            out_tokens.append(toks)
            out_logprobs.append(probs)
        # 53. 返回一个元组,包含生成的令牌序列和对数概率(如果计算了)。
        return (out_tokens, out_logprobs if logprobs else None)


    # 54-62. 这个方法用于文本完形填空任务。
    # 它首先对输入的提示进行分词编码,得到整数列表的列表。
    # 然后,它调用 generate 方法,根据编码后的提示生成文本序列,并获取生成的令牌序列和对数概率(如果计算的话)。
    # 最后,它根据是否计算了对数概率,返回不同格式的结果列表。
    # 如果计算了对数概率,每个结果字典中包含生成序列文本、每个令牌的文本以及对应的对数概率。
    # 如果没有计算对数概率,每个结果字典中只包含生成序列文本。
    # 这个方法使用了 generate 方法中的核采样技术,通过控制温度值和 top-p 值来引入可控的随机性,生成更自然、更多样化的文本完形填空结果。
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        # 54. 这是一个用于文本完形填空的方法。
        # 55. 它接受一个字符串列表作为提示,以及其他参数,如温度值、top-p 值、最大生成长度、是否计算对数概率和是否回显提示。
        # 56. 返回一个 CompletionPrediction 对象列表,每个对象包含生成的完形填空结果。

        # 57. 如果没有提供最大生成长度,则将其设置为模型允许的最大序列长度减 1。
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        # 58. 对提示进行分词编码,得到一个整数列表的列表,每个内部列表表示一个提示。
        # 59. 在编码时,添加开始符但不添加结束符。
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        # 60. 调用 generate 方法,根据编码后的提示生成文本序列,并获取生成的令牌序列和对数概率(如果计算的话)。
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        # 61. 如果计算了对数概率,则返回一个列表,每个元素是一个字典,包含:
        #     1. 解码后的生成序列文本
        #     2. 生成序列中每个令牌的解码文本
        #     3. 对应的令牌对数概率
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        # 62. 如果没有计算对数概率,则只返回一个列表,每个元素是一个字典,包含解码后的生成序列文本。
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]


    # 63-79. 这个方法用于对话生成任务。
    # 它首先对输入的对话进行预处理,包括检查是否包含特殊标签、处理系统消息、检查角色顺序等。
    # 然后,它对对话进行分词编码,将用户消息和助手回复编码为一个整数列表,并插入特殊标签以区分不同消息。
    # 接着,它调用 generate 方法,根据编码后的对话生成文本序列,并获取生成的令牌序列和对数概率(如果计算的话)。
    # 最后,它根据是否计算了对数概率,返回不同格式的结果列表。
    # 如果计算了对数概率,每个结果字典中包含生成的助手回复、每个令牌的文本以及对应的对数概率。如果请求是不安全的,则返回错误信息。
    # 如果没有计算对数概率,每个结果字典中只包含生成的助手回复。如果请求是不安全的,则返回错误信息。
    # 这个方法使用了 generate 方法中的核采样技术,通过控制温度值和 top-p 值来引入可控的随机性,生成更自然、更多样化的对话回复。
    # 它还做了一些特殊处理,如处理系统消息、检查角色顺序等,以确保对话的有效性和正确性。
    # 如果请求中包含特殊标签,则会被标记为不安全请求,并返回错误信息。
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        # 63. 这是一个用于对话生成的方法。
        # 64. 它接受一个对话列表作为输入,每个对话是一个消息列表。
        # 65. 它还接受其他参数,如温度值、top-p 值、最大生成长度和是否计算对数概率。
        # 66. 返回一个 ChatPrediction 对象列表,每个对象包含生成的助手回复。
        # 67. 如果最后一条消息不是用户发送的,或者对话角色顺序不正确,它会引发 AssertionError 异常。


        # 68. 如果没有提供最大生成长度,则将其设置为模型允许的最大序列长度减 1。
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        
        # 69. 检查每个对话中是否包含特殊标签,如果包含,则标记为不安全请求。
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            # 70. 如果第一条消息是系统消息,则将其与第二条消息合并,并用特殊标签包裹系统消息。
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            # 71. 检查对话中的角色顺序是否正确,即用户和助手消息交替出现,并且可以有一个可选的系统消息在开头。
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

            # 72. 对对话进行分词编码,将用户消息和助手回复编码为一个整数列表。
            # 73. 在每个用户消息和助手回复之间插入特殊标签,用于区分不同的消息。
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            # 74. 检查最后一条消息是否来自用户,并将其编码为整数列表,但不添加结束符。
            # 75. 这是因为我们需要为最后一条消息生成助手回复。
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            # 76. 将编码后的对话令牌添加到提示令牌列表中。
            prompt_tokens.append(dialog_tokens)
        # 77. 调用 generate 方法,根据编码后的提示(对话)生成文本序列,并获取生成的令牌序列和对数概率(如果计算的话)。
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        # 78. 如果计算了对数概率,则返回一个列表,每个元素是一个字典,包含:
        #     1. 生成的助手回复,如果请求是不安全的,则返回错误信息
        #     2. 生成序列中每个令牌的解码文本
        #     3. 对应的令牌对数概率
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]

        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


# 这个函数实现了 top-p (nucleus) 采样,它是一种常用的文本生成技术。
# 与传统的 top-k 采样相比,top-p 采样可以更好地控制生成文本的多样性和质量。
# 它通过选择累积概率质量超过阈值 p 的最小标记集合,并对剩余的概率分布进行归一化,从而实现了更加多样和自然的文本生成。
# 这种采样技术在自然语言处理领域中得到了广泛应用,尤其在生成式任务中,如机器翻译、对话系统和文本摘要等。
def sample_top_p(probs, p):
    # 这是一个函数,用于对概率分布执行 top-p (nucleus) 采样。
    # Top-p 采样是一种常用的文本生成技术,可以控制生成文本的多样性和质量。
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    # 1. 首先,将概率分布 probs 按降序排序,得到排序后的概率值 probs_sort 和对应的索引 probs_idx。
    # 2. 这是为了方便后续的 top-p 采样操作。
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 3. 计算排序概率值的累积和 probs_sum,用于确定哪些标记的累积概率质量超过了阈值 p。
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 4. 创建一个掩码 mask,用于标记哪些标记的累积概率质量超过了阈值 p。
    mask = probs_sum - probs_sort > p
    # 5. 将掩码对应的概率值置为 0,这样就可以剔除掉累积概率质量超过阈值 p 的标记。
    probs_sort[mask] = 0.0
    # 6. 对剩余的概率值进行归一化,使它们的总和为 1。
    # 7. 这是因为在剔除掉一部分标记后,概率分布的总和可能不再为 1,需要进行归一化。
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 8. 从归一化后的概率分布中,使用 torch.multinomial 函数进行采样,得到下一个标记的索引 next_token。
    # 9. num_samples=1 表示只采样一个标记。
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # 10. 使用 torch.gather 函数,从原始的索引 probs_idx 中收集采样得到的标记索引 next_token。
    # 11. 这样可以确保返回的标记索引与原始概率分布一致。
    next_token = torch.gather(probs_idx, -1, next_token)
    # 12. 返回采样得到的标记索引 next_token。
    return next_token
