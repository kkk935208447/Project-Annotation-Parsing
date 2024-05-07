from transformers import PretrainedConfig

# 定义 ChatGLMConfig 预训练模型的参数, 部分参数会被  config 文件覆盖
class ChatGLMConfig(PretrainedConfig):   
    model_type = "chatglm"     # 1. 设置模型类型为 "chatglm"
    def __init__(
        self,
        num_layers=28,              # 3. num_layers 参数指定 Transformer 模型的层数,默认为 28 层
        padded_vocab_size=65024,        # 4. padded_vocab_size 参数指定填充后的词表大小,默认为 65024
        hidden_size=4096,                   # 5. hidden_size 参数指定模型隐藏层的大小,默认为 4096
        ffn_hidden_size=13696,              # 6. ffn_hidden_size 参数指定 Feed-Forward 层的隐藏层大小,默认为 13696
        kv_channels=128,                    # q k v dim
        num_attention_heads=32,
        seq_length=2048,                    # 模型允许的最大长度, 后面会被config覆盖, chatglm2默认32k
        hidden_dropout=0.0,                 # 10. hidden_dropout 参数指定隐藏层的 Dropout 比例,默认为 0.0
        classifier_dropout=None,            # 11. classifier_dropout 参数指定分类器的 Dropout 比例,默认为 None
        attention_dropout=0.0,              # 12. attention_dropout 参数指定注意力层的 Dropout 比例,默认为 0.0
        layernorm_epsilon=1e-5,
        rmsnorm=True,                                         # 14. rmsnorm 参数指示是否使用 RMSNorm 归一化,默认为 True
        apply_residual_connection_post_layernorm=False,      # 15. 是否在层归一化后应用残差连接,默认为 False
        post_layer_norm=True,           # 16. post_layer_norm 参数指示是否在最后层用于归一化,增加模型的泛化能力
        add_bias_linear=False,
        add_qkv_bias=False,             # 18. add_qkv_bias 参数指示是否在查询、键和值投影中添加偏置项,默认为 False
        bias_dropout_fusion=True,       # 17. 是否将dropout与bias操作融合,可以提高计算效率
        multi_query_attention=False,        # 20. multi_query_attention 参数指示是否使用多查询注意力机制,默认为 False
        multi_query_group_num=1,            # 21. multi_query_group_num 参数指定多查询注意力组的数量,默认为 1
        apply_query_key_layer_scaling=True,  # 22. apply_query_key_layer_scaling 参数指示是否对查询和键进行层缩放,默认为 True
        attention_softmax_in_fp32=True,     # 23. attention_softmax_in_fp32 参数指示是否在 FP32 精度下计算注意力的 Softmax,默认为 True
        fp32_residual_connection=False,     # 24. fp32_residual_connection 参数指示是否在 FP32 精度下计算残差连接,默认为 False
        quantization_bit=0,                 # 25. quantization_bit 参数指定量化位数,默认为 0(表示不进行量化)
        pre_seq_len=None,                   # 26. pre_seq_len 参数指定前缀序列的长度,默认为 None
        prefix_projection=False,            # 27. prefix_projection 参数指示是否使用前缀投影,默认为 False
        **kwargs        # 28. **kwargs 用于接收其他额外的关键字参数
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection

        # 30. 调用父类 PretrainedConfig 的 __init__ 方法,传递额外的关键字参数
        super().__init__(**kwargs)