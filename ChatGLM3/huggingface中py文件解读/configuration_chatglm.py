from transformers import PretrainedConfig

# ChatGLMConfig是一个预训练配置类,后续会被config.json覆盖部分参数
class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    def __init__(
        self,
        num_layers=28,               # 1. 模型总共有 num_layers 层 transformer block
        padded_vocab_size=65024,     # 2. 模型的词表大小(包括padding后的尺寸),一般是2的幂次方
        hidden_size=4096,            # 3. 模型的隐藏状态大小,决定了模型的参数量和推理性能
        ffn_hidden_size=13696,       # 4. 前馈网络的隐藏层大小,通常是4倍的hidden_size
        kv_channels=128,             # 5. q k v dim
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,          # 8. 隐藏层dropout比例,用于防止过拟合
        classifier_dropout=None,
        attention_dropout=0.0,       # 10. 注意力层dropout比例,用于防止过拟合       
        layernorm_epsilon=1e-5,      # 11. Layer Norm的Epsilon值,用于提高数值稳定性
        rmsnorm=True,                # 12. 是否使用RMSNorm替代LayerNorm,RMSNorm可以提高训练稳定性
        apply_residual_connection_post_layernorm=False,      # 13. 是否在残差连接之后应用LayerNorm,影响模型的收敛性
        post_layer_norm=True,                                # 14. 是否在最后一层使用LayerNorm,影响最终输出的分布
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,                           # 17. 是否将dropout与bias操作融合,可以提高计算效率
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,                 # 20. 是否在注意力计算中应用query-key缩放,可以提高模型稳定性
        attention_softmax_in_fp32=True,                     # 21. 是否在注意力softmax计算中使用fp32,提高数值稳定性
        fp32_residual_connection=False,                     # 22. 是否在残差连接中使用fp32计算,提高数值稳定性
        quantization_bit=0,
        pre_seq_len=None,                                   # 24. 如果使用prefix tuning,指定prefix的长度
        prefix_projection=False,                            # 25. 是否使用prefix projection,可以提高prefix tuning的效果
        **kwargs
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
        super().__init__(**kwargs)