from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
import os


class ModelUtils(object):

    @classmethod
    def load_model(cls, 
                   model_name_or_path,   # model_name_or_path: 基础模型的路径或名称
                   config=None,   # config: 基础模型的配置对象,可选
                   load_in_4bit=False,  # load_in_4bit: 是否使用4位量化进行推理,默认为 False
                   adapter_name_or_path=None  # adapter_name_or_path: 适配器模型的路径或名称,可选
                ):
        # 是否使用4bit量化进行推理
        if load_in_4bit:
            # 如果使用4位量化,则设置相应的量化配置
            # 使用4位量化可以减少模型大小和内存占用,加速推理速度
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,   # 在4位量化时,计算使用 float16 数据类型
                bnb_4bit_use_double_quant=True,  # 使用双量化技术
                bnb_4bit_quant_type="nf4",   # 使用 NF4 量化类型
                llm_int8_threshold=6.0,   # 在量化时,大于该阈值的权重使用 int8 表示
                llm_int8_has_fp16_weight=False,  # 指示权重不使用 float16 表示
            )
        else:
            quantization_config = None

        # 加载base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,   
            config=config,
            # load_in_4bit=load_in_4bit,   # load_in_4bit 指定是否使用4位量化
            trust_remote_code=True,
            low_cpu_mem_usage=True,     # low_cpu_mem_usage 使用低 CPU 内存模式
            torch_dtype=torch.float16,   # torch_dtype 指定模型权重的数据类型为 float16,可以节省内存
            device_map='auto',   # device_map 自动将模型分布到不同设备上,加速推理
            quantization_config=quantization_config
        )

        # 加载adapter
        # 如果提供了适配器模型路径,则加载适配器模型
        # 适配器模型可以在基础模型的基础上进行微调和扩展
        if adapter_name_or_path is not None:
            trainable_params_file = os.path.join(adapter_name_or_path, "trainable_params.bin")
            if os.path.isfile(trainable_params_file):
                # 加载其他的可训练参数
                model.load_state_dict(
                                torch.load(trainable_params_file, map_location=model.device),
                                strict=False)
            model = PeftModel.from_pretrained(model, adapter_name_or_path)

        return model
