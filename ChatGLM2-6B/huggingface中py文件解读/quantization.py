from torch.nn import Linear
from torch.nn.parameter import Parameter

import bz2
import torch
import base64
import ctypes
from transformers.utils import logging

from typing import List
from functools import partial

logger = logging.get_logger(__name__)

try:
    # 1. 导入cpm_kernels库中的关键类和函数, cpm_kernels是一个用于量化和优化深度学习模型的库, 它提供了一些底层的优化功能,如权重压缩、量化等
        # 1) LazyKernelCModule: 用于延迟加载内核代码,提高性能和内存利用率
        # 2) KernelFunction: 用于调用内核函数,执行特定的优化操作
        # 3) round_up: 一个辅助函数,用于将数值向上舍入到某个整数倍
    from cpm_kernels.kernels.base import LazyKernelCModule, KernelFunction, round_up

    # 2. 定义Kernel类,用于管理和调用自定义CUDA内核
    class Kernel:
        def __init__(self, code: bytes, function_names: List[str]):
            self.code = code       # 4. code是编译后的CUDA代码,以字节序列的形式提供
            self._function_names = function_names  # 5. function_names是一个列表,包含了要加载的CUDA内核函数的名称
            self._cmodule = LazyKernelCModule(self.code)  # 6. 使用LazyKernelCModule类加载CUDA代码,创建一个可调用模块
            for name in self._function_names:  # 7. 遍历function_names列表中的每个函数名称
                # 8. 使用KernelFunction类创建一个可调用的CUDA内核函数
                # 将其作为Kernel类的属性添加,以便后续调用
                setattr(self, name, KernelFunction(self._cmodule, name))

    # 3. 定义一个压缩的 base64 字符串,表示 CUDA 内核代码, 这段代码被编码成 base64 格式,用于减小文件大小和传输成本, 在实际使用时,这段 base64 字符串需要被解码,以获取原始的 CUDA 内核二进制代码
    quantization_code = "$QlpoOTFBWSZTWU9yuJUAQHN//////////f/n/8/n///n//bt4dTidcVx8X3V9FV/92/v4B7/AD5FBQFAAAChSgKpFCFAFVSigUAAAEKhSgUUqgFBKigqVREQAABQBQIANDTTIGI00BkZBkNGE0A0BkBkGQGRkaNAaAGQNBoGgDIAAYIGTI0DQAQAaGmmQMRpoDIyDIaMJoBoDIDIMgMjI0aA0AMgaDQNAGQAAwQMmRoGgAgA0NNMgYjTQGRkGQ0YTQDQGQGQZAZGRo0BoAZA0GgaAMgABggZMjQNABABoaaZAxGmgMjIMhowmgGgMgMgyAyMjRoDQAyBoNA0AZAADBAyZGgaAAmqU1NEgJqnptU/Sn4jRR6J6epk2pqb1Q/SgAPUGgyNNGjQ2SBpoAZAAGg0NB6mgDIAAAAA2oaApSREBNAARhGiYEaEwU8pvImlP0k2aam1GaGqbFNM1MHpTwmkepmyU9R6nqPKekHqNNPUxNGhp6n6p6QaZ6o9TG1GMqcoV9ly6nRanHlq6zPNbnGZNi6HSug+2nPiZ13XcnFYZW+45W11CumhzYhchOJ2GLLV1OBjBjGf4TptOddTSOcVxhqYZMYwZXZZY00zI1paX5X9J+b+f4e+x43RXSxXPOdquiGpduatGyXneN696M9t4HU2eR5XX/kPhP261NTx3JO1Ow7LyuDmeo9a7d351T1ZxnvnrvYnrXv/hXxPCeuYx2XsNmO003eg9J3Z6U7b23meJ4ri01OdzTk9BNO96brz+qT5nuvvH3ds/G+m/JcG/F2XYuhXlvO+jP7U3XgrzPN/lr8Sf1n6j4j7jZs+s/T0tNaNNYzTs12rxjwztHlnire3Nzc3N1wuBwOBwXBvZfoHpD7rFmR99V5vj3aXza3xdBbXMalubTg/jIv5dfAi54Pdc75j4z412n3Npj3Ld/ENm7a3b/Cod6h/ret1/5vn/C+l+gdslMvgPSLJ8d8q+U66fevYn/tW1chleEtNTGlcHCbLRlq0tHzF5tsbbZZfHjjLgZu42XCuC3NrdjTasZGNzgxPIrGqp7r3p7L2p5XjnpPSmTd5XtzqnB6U87zzg1Ol0zd0zsLszxR6lkxp35u6/teL0L0W922cR7Lu1lpL9CsHirzuM2T+BgsyViT6LHcm0/Vr6U/7LGGyJeqTEjt0PHWhF5mCT7R9mtlDwriYv0Tyr/OxYt6qp5r0mPVT0608TqnqMZaarU2nFwrTzzlrs1ed7z1ux60wyr4ydCaTi3enW8x68x0zU7tXSlcmPSW1mGpWJMg4zmPC2lK96tp0OE80y4MfEvnZj8zGluR6b22ki1Ou9V2nCd9xovcPvcYMZYy0lvN60ScZ45vN6yeCeeXFb1lVjnnCar5fwXwE2bzJ4HI1XVPXfXZMm44GUsMpYsmLB65TuVdm0cl0b+i/wGNN66XjeV7zuPpHcnK/juhhjdfId5jMdE5nN0dGmmm2zZs2cexD5n9p/dY352XsvXHaZNWWsmmS1atjR452nYudzvqv2HMRyvNNnlMcDl3R2+yx2uVrBubTW9icHDVtbNXlZm7jma1rM4VurZZd2y6nUau7ZXZ7bVU+mnoOVxZGMrVmvX60605JwmzGZhhhjTWtaaaMaaGTGmNMZasY0iX8VMUl8eepaIrzGSpemWOQyZORk2bNpjUybMmxqYmknCGCFynutfksaZpjTNMaaatM0xsxcGR0sociNqxNSmhhR1ZJPbsn8qyF0t2qH6iYBclclalbtTTcHTDsPaX6rlnElph2Jyumumtynv2Kk8GI7rsvXbIcJgHJOSaSXnnGaI3m87RtVXJOZ/YtgdTE6Wpha6ZlE8ayXkef1fh602r2WwvfMXtMdLlkfnLFdYYwYso+bWqm7yJqHXZGw2nrS5ZanSYnWlxBxMF1V940K2wdrI7R6OYf7DGGamMmTSbRhlS45xmVOumF1EyPCmHrrN8wwZOOrdNtLeMtzFzDlWnfTBxMk2NaXIZHBYxYLD4w8yju0ao65Vz1OIXoS9dLanwCe1PWrYuWMqf1if1z2k2yYfKJ741PDgno1ZQ8DRqvUny3mNoWTzGO6m1DkrJI8JiR5cSd+vZdGOO8nrMoc5+NDUFsMSXaZJeNlMmGLtJsovOsUp7I9S5VojKxF6bTVEelXqlfJobQr3LozSh2Jk7VcrVMfhXqszGWMzNqGhqZY0OadxkyyMssKugZR0KNFXBHlqwmJgTE/BNVMk6ItJXZMR0H47GpXv/DMOvNkmVuaV1PRfEdxuqc7Hcd+ZV/zTLaRxWk0nl9CdCeM6mn5rstHIBcpiuwmUZXeq81DacHI2rmrZ5SuE5mOZd6LQrZg9mx32TprA8BMo5jKN6yLTCi3WzQaZSuhzTtM1fUTGVpG8Tw+KXI0tjEpiWxtLYynOlktSbVlaI5kxP8TDH8kx50xoxi5KcA4pcja8KWLRlO/Ks6q06ergnvm1ca3Tq8Uw7LTUsmWyctXPWmpitl/uvGcWTGXGuAXDfhqazGmjkxcJW5hMMMMpYsXl2TZYtVOddG3XCarUt6Ptq9CZXSNzyuRzqRZOjsxdBbFVz6OA5HI43r1jityVlVpVkxmOsyaYWE1NTGq1sOVh36mHMcxtSvcy70edG0ZGR3I1Go1GRlV7mWWo1G0ZGRqlvH40l7o4m5xMWLLLYyNjnqc8556mdPqLJ31n/1nWOncxzG1tizrHs/Z+d2vP/B/l8wdJ6rHUn2nbbDq4p6htFtYzMMMTaZis1K5GKzGNmxhmUx2DDlZ/qNnIx41xnaMfCZWYaZWtNLTNW8ND4Fw1MyZOCdM428suKG1ehW8TesOydg7J+YYcD4cYR+8dFK6M4E3HM9ZfRNNL+Sn6rsl4DsrDl2HpPCnfxjGXtbZtYys1ttlyJ4T+BvexjGWRjMszK4Jpc77D3GyuVD7q0+G8m9G+2+rGm7cOR2y7FdtY2XUYx/oNlfRYxhMYyYZkyyg55enna9Kt/FFi6GMMwYwdwxWgxGMLKYmUyGExTKMZkMFhkymKuh0NOBNnBu+23LdwDoZYYzGGMxtORaTU1pjTGWTTGGtMrNWUsyyTTLLG1qy2ZjbK2DBllWqxMtBMaYZQmcE7zvvRcTkclUwdkxTaSdyySt/7fpL+T1v516Ji97fwr5JbLu305zMn5+GMTTZ9F+y7ExwmGVfG44yxn3dLv6l5i+Wth1jCrDq21nW9LqvvDzz3Vf3LLH/O/32TJ/erx3bXftO4eF+G956D952K/An4NfvOpjFjExjevP/UmE0fIoZXx6/w6lX/no3D0bLt+ixjieBM6ksRd0yB4Lt2SwYNE+gd1detlZWUnpiZfGfFaK+4PyCa/v18V8X75pe9fLXzp7l3VjF76vWZmHwGz1IZNWT7b8yddJ4q5kyrVdfru6atWc7bVYztL9Jf4GXvT+Y8m9/YsXP6H018a8D4XVOqvfzqeR+6yZOD8dPv0+U7/q5Pl+2dNb0MjzGVH5p6MNQ7cOWvw62U9aHE8DprDek+McLyvDz+te+9Zhq5+YTruufMcWMabqysTmZVWjKPfnK0wyVcrsuhjZRdLkHNvD72b9abriOSGIxiLixMOoalNPXzy+wT/tf+U6HHONfsz+xe8ufHBdQWWGWLA9if0rsnmrxK5LvRZQeWsTCsrmOYy8VteVfuRfcVTtDLItLIsMYxZLdU/DbtSemxF6Z6Zo5WBXE4tFdCyVMMXMTEMZXVlS6Xec2T4e0tHsRcEuWshcJ2YsNF5rUx1E8ifCq6Z+ZP7qdCeu/aTwFd53l16/o0NOw6O3dLavP4Hbi4RdmuDk6DoYaninC0+o4uZjbJ7Rxeu0/FbuFg+q7DVS6fQe0rZ6NDGUNNU6DEqOaLTicKnYZMnBWruljQxoaS3dZhocDge0bSTyOvdAbG5hxe2xji7E/L55xX13wWNDi6HCekcFxfCPGxY0MXC+s7afWaMdDyjyr+o8Rudm/NabOZvdl274zH4f5XK9z6On1Pe/K5TdPAslg77BjuO6Y3eO7GqvOPG/stknp1leyvLL0Z7bl9I4noMvLkzytLhWYzrOZzLXCORe028rORzOg4N/L0HlMOQ3Pgmnbb6KczlabORpu980q37TBqRu0/p3PO6234Bl03Ynuz+9W7gnsEcmvYaYY3aMYY0wx3pYd+ujsXauWdaY5Xkbtl23fPzFHiDB/QMo0yFjBllYxTQYYyxkrwn7JufwJ/PfgJ+C83X69ni6zvXcnyXabv0ncbLwsceS+RNlyN2mnneJtX0ngYO0+e+0+UnA+Wch3ji8hj5an4h+i6XBySU4n+R0roVcbw5yvHrmr4Yw8Y7x6c+9POPYHI5HI5HI5HI5HGXGww4nE4nrVyOR8XeqPEO7PLOiukYa3Novk5hV4cdtYZLI93e+uxff2jRo0aNGjRo0aNG1bVtW1dy3m83m8+tQ5ZzHw3nObwOu8La9Rc1dtkdS8A3eTk823tnktXWlxN6Oixe06zrN70Isd9jiOgZFq9yfkPqP/SLhN2Myl8jDM43bl1nbcb4cO57jlh8Jow6pzXZdL4dyODTuuhu77FyO27DdwdRxmvO+O+3N2+BdqyTwLHVczDVY4UPE4O66/ZO2cx1LFzVdSXtF7G4HMbrauOHRw6c8FdZ5m9fHZHYZXfTlZquyynSyTTKke6vcffSD9pzPA/G7n7jxPmuhc1DHMynPMrGL6AdewYmwu5ko+UUyTwrMv27rPH1v1nGqd87+p6N6LU8k3NEng53xXyHS97+44OSg/sy/hn+Se6yfYNjW0/uTgP+PvWYzLMmjhcLB/gGpri6H83/84eUXWT6T9Hsv7785z/7z4icpW+zfXypuR7rx/gMdZb1/wC678pcs8/2a3mDitGHxl9mfPlll5MafWWqxk/eYuTDgcNMzDGWLWvsuglNxs53GtN6uWpktlW1tZZYcuinMMWmnNnJydze3b2Y1McBxrBkXw799izLMZZYyy0TkbsGM4p03S2uVu5s/XXUdSdec6smVxZYYGpVmT8A+8ajuEyV5FatkvVru2x6uxGXXbH4A+jvgP4GMYy3iPLXzq/6z65+E005ey+cwMZD3fZcqc6xpjTFjQ0P3U+e++cPYmTIwj0nrK5NPTfl3WvpfLtXDcb2HQMudYOxFXQBor4L4T6vrOauFctYXJQ++NUWmJe5bmx1jDiZS1dTqWxo4GR8jm3fttpmPHppk9PEyv4/y8/sO07XacOmcqc0x2Vi9BvNJvN5oW8x4mOsydpidRxMYJPx06m1bqPzq9KtK8sxXNXFodD/+MYYaJTLwOhc9brCsV18oOR1i4tXChyTkq4lf4y1Ke+9axjDHqs1mfBbMXuP4Hzi+X7t8vzv7bHerrUPgPCxhjre4fXdfLNtNM+Jd+Zdh8xd8wP87uNPoPgv4W7/5P2BuxfsMabNnMnza+54Pdi5U671GPZY8CehX8Voeoo7FHpkeEc6715FwHZrIrUrHaviPUbPZHND+IhczrP6FcYvhOZ0Di/ETt0OI+YwNWR9r7tpf6WDeZKZDB1+z2IthOl1mPyb5FluvEx9h9d0NnM0Y1XPFkWIsk1WotJ0PBMmkvjvQTd0e71tfeV+8r8lQ/tpzpsmxJ+InrI/dj2UajUajVTUajatRqNRtGo1Go1Go4wjeMpZFMVV9CHbofPraLsJ3JpWV2XOoanCuFky4y3PPNxucK2uKC1Lbdb1eo+m5XomN6HfeZsabHLHRX/K+offtNGGmHWctcVcG44MdSqsOLY9VzX+Zxfxn2HPdWTpzWvkrtJ8M5zorrKcquRytJ5N5DZmcaW02l76nWO+BqPXm1A2Ry/0q71dH/mqrqeFjkYxjEXtsX8qubTk67rGycyqsdm4tZx5D6D5hhi0waaWmiaMP81Yjii5qxPlPuU/GfTL1Y5E6Jyfiq63qTa39A4J0sOGDgO9WF9bOXl0XfPRbsY2bPNKPy1YrFYrFYmRhhlTIyMjJWJYZHXuCXI8OoXsvfljGLFicNifpp2XunoPiG1wtx3p1Tah+/DD66OnVtVXP9rKbVxOnL0tR/rHtqB5UDErUVcl11D4qqvjpOcxX7armUNJB3LpW6bxVvD08e8h3odKKvyCFZBdSh2FVcST9xV3n3T8t1j7Kr9qgrqXg+13Pt5U7JCvFXVIV1YG5lRhkVYZJYYDDD4KOIMoHCp26WS8GB7uBh2zIdgq/PKyInjV2STShuoapUdCpX1yTwqq/z1VvET7Kh5nVPkO8YyxjLt2MaaMmWTLQvx3qnzltnXW0p2jxgbEtSny/Osv8Y9pLMXYoHVPAhkVdWVeODhR6q9/Sxe2liwwZWMVvFXfRkeIDxAePUPIrdJ4ey6yquzH+PD/bUOWAu05qVHtFd8rrKHSoeNIOUqrYr3FXyToqfYJgwmJdKpXXOwYYegNNGMzfZPp/t3t/DVs4zjNTN61rRqaWaa4NYbRjTa0tWwy2Y2tGN8ZO8ofNKq4j9SL7I+cSm4/6ovLV5HNXLI0jJidwrtk6ynCaP6Z++GjRlWS3tLeW129Mi9evxU9mtz6s5J3Z7M2ngTgnKvmpomxpaLCzPfmx0JWE+m3NLDDGOX47RctdYYNK5jakdqLkRlI39n590T5zctGSwwZZDJj6kW8XSi6ot2MmWWJ0DUT3nuvebBudScjZ79g8cWJ8av0k+/bE5WKd5MdbFpbDVMxu1DVMmtNZGJvq1mtRbn6M+g/kP0FwDwr7quZs7xosNGpbscyxhhd9TyJyFwbLcxlTasg75vW7TsV5K7ji44XPMMrdoj+Y3rT0Hie62nlYV/pwczzOmdLqLhYkzGMzCZWGMQzGMSsZYY6Di1t4nlJ+Em63mJxrVLxPbYxNEdgc1dU2iOKyoYYWjNrEeHTYybVk0atSa7ehuwsWMWTqn1TrnS6hYsi71d1+s+k+ic70e20fzE/VaTdxT9ZtU4GIXdeNx3X77guYYfpHeTQjaMX6brOu4OY4K7Y2d9mbHarI5ox3p4GpJ2Vd/Tst60f7j999pppjR+Q/Qf8J/VaORs3cji7FfFuN61+ui9s8hix1OCh5KGVV23BPXvZfz3CLyHpix+exi8z/KnCnosY2eunor+cxyPO/xJ0vKey9OvE9VjqaYu0x3Z3jd6o2b1T12D+F8l232lwaaacD5LE8LBxu7WTlbWraWpew8Xexjel3E+wWD4APITdNqR8F3R3T0lunCQ4GaE9R37DxeCYfcHi4xci5ovKfxVs55y2hf+65E/Xdp6jR5nrebTmi5incpkyOjs50JvrZwstbbW6kfuuQw+2mykf/EXNFzxfKTrxew929TR6bWnGL//F3JFOFCQT3K4lQ"

    # 4. 创建一个 Kernel 实例, 首先,使用 bz2.decompress 和 base64.b64decode 对压缩的 base64 字符串进行解码,获取原始的 CUDA 内核代码,然后,将解码后的 CUDA 内核代码和指定的函数名称列表传递给 Kernel 类的构造函数,创建一个 Kernel 实例
    kernels = Kernel(
        bz2.decompress(base64.b64decode(quantization_code)),
        [
            "int4WeightCompression",     # 该函数将一个 (n, 2m) 形状的 4 位权重张量压缩为一个 (n, m) 形状的 8 位权重张量
            "int4WeightExtractionFloat", # 该函数将一个量化的 4 位权重张量转换为 float 类型的权重张量
            "int4WeightExtractionHalf",  # 该函数将一个量化的 4 位权重张量转换为 half (float16) 类型的权重张量   
            "int8WeightExtractionFloat", # 该函数将一个量化的 8 位权重张量转换为 float 类型的权重张量
            "int8WeightExtractionHalf",  # 该函数将一个量化的 8 位权重张量转换为 half (float16) 类型的权重张量
        ],
    )
except Exception as exception:
    # 3. 如果加载cpm_kernels库失败,设置kernels为None并打印警告信息
    # 这是为了防止因为外部依赖库未安装或版本不匹配而导致程序无法运行
    kernels = None
    logger.warning("Failed to load cpm_kernels:" + str(exception))


# 6. 继承自 torch.autograd.Function, 这是一个自定义的函数,用于实现量化线性层的前向和反向传播
class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                inp: torch.Tensor,              #   1) inp: 输入张量
                quant_w: torch.Tensor,          #   2) quant_w: 量化的权重张量
                scale_w: torch.Tensor,          #   3) scale_w: 用于量化权重的缩放因子
                weight_bit_width                #   4) weight_bit_width: 量化权重所使用的位宽
                ):
        #   1) 保存输入张量的形状 (inp.size()) 和权重位宽 (weight_bit_width) 到上下文对象 (ctx) 中,以备后续使用
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        #   2) 获取输出特征数量 (out_features),即量化权重张量的第一个维度 (quant_w.size(0))
        out_features = quant_w.size(0)
        #   3) 将输入张量 (inp) 转换为连续内存布局,并展平为二维张量 (view(-1, inp.size(-1)))
        inp = inp.contiguous().view(-1, inp.size(-1))
        #   将量化的权重张量 (quant_w) 和缩放因子 (scale_w) 转换为 float16 或 bfloat16 类型的权重
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        #   5) 保存转换后的权重张量形状 (weight.size()) 到上下文对象 (ctx) 中
        ctx.weight_shape = weight.size()
        #   6) 使用矩阵乘法 (inp.mm(weight.t())) 计算线性层的输出
        output = inp.mm(weight.t())
        #   7) 将输入张量 (inp)、量化权重张量 (quant_w) 和缩放因子 (scale_w) 保存到上下文对象 (ctx) 中,以备后续在反向传播时使用
        ctx.save_for_backward(inp, quant_w, scale_w)
         #   8) 将输出张量的形状还原为期望的形状 (ctx.inp_shape[:-1] + (out_features,)),并返回结果
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod # 8. backward 方法实现了量化线性层的反向传播
    def backward(ctx, 
                 grad_output: torch.Tensor      #   1) grad_output: 输出梯度张量
                 ):
        #   1) 从上下文对象 (ctx) 中获取保存的输入张量 (inp)、量化权重张量 (quant_w) 和缩放因子 (scale_w)
        inp, quant_w, scale_w = ctx.saved_tensors
        #   将量化的权重张量 (quant_w) 和缩放因子 (scale_w) 转换为 float16 或 bfloat16 类型的权重
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        #   3) 将输出梯度张量 (grad_output) 转换为连续内存布局,并展平为二维张量 (grad_output.contiguous().view(-1, weight.size(0)))
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        #   4) 根据链式法则,计算输入梯度 (grad_input = grad_output.mm(weight))
        grad_input = grad_output.mm(weight)
        #   5) 根据链式法则,计算权重梯度 (grad_weight = grad_output.t().mm(inp))
        grad_weight = grad_output.t().mm(inp)
        #   6) 将输入梯度 (grad_input) 的形状还原为原始输入张量的形状 (ctx.inp_shape)
        #   7) 将权重梯度 (grad_weight) 的形状还原为原始权重张量的形状 (ctx.weight_shape)
        #   8) 返回输入梯度 (grad_input)、权重梯度 (grad_weight) 和两个 None (对应缩放因子和权重位宽无需计算梯度)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None

# 10. 将一个 (n, 2m) 形状的 4 位权重张量压缩为一个 (n, m) 形状的 8 位权重张量
def compress_int4_weight(
                        weight: torch.Tensor   # 输入参数是一个 (n, 2m) 形状的 4 位权重张量
                    ):  # (n, m)
    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        #   5) 创建一个新的张量 out,形状为 (n, m),数据类型为 torch.int8,放置在 CUDA 设备上,用于存储压缩后的 8 位权重
        out = torch.empty(n, m, dtype=torch.int8, device="cuda")
        #   6) 获取当前 CUDA 流
        stream = torch.cuda.current_stream()

        # 7) 计算 CUDA 内核的网格维度 gridDim 和块维度 blockDim
        #    网格维度设置为 (n, 1, 1),表示有 n 个线程块沿着第一个维度
        #    块维度设置为 (min(round_up(m, 32), 1024), 1, 1),表示每个线程块最多有 1024 个线程,线程数量由 m 决定,并向上取整到 32 的倍数
        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)
        kernels.int4WeightCompression(    # 8) 调用 kernels.int4WeightCompression CUDA 内核函数,将 4 位权重压缩为 8 位权重
            gridDim,
            blockDim,
            0,
            stream,
            [ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(n), ctypes.c_int32(m)],
        )
        return out     # 9) 返回压缩后的 8 位权重张量 out

# 11. 用于从量化的权重张量中提取 float16 或 bfloat16 类型的权重
def extract_weight_to_half(
                            weight: torch.Tensor,    #   1) weight: 量化的权重张量,数据类型为 torch.int8
                            scale_list: torch.Tensor,  #   2) scale_list: 用于量化权重的缩放因子,数据类型为 torch.half 或 torch.bfloat16
                            source_bit_width: int      #   3) source_bit_width: 量化权重所使用的位宽,可以是 4 或 8
                        ):
    #   1) 断言 scale_list 的数据类型是 torch.half 或 torch.bfloat16
    assert scale_list.dtype in [torch.half, torch.bfloat16]
    #   2) 断言 weight 的数据类型是 torch.int8
    assert weight.dtype in [torch.int8]
    #   3) 如果 source_bit_width 是 8,直接将权重转换为 scale_list 的数据类型,并与缩放因子相乘
    if source_bit_width == 8:
        return weight.to(scale_list.dtype) * scale_list[:, None]
    #   4) 如果 source_bit_width 是 4,根据 scale_list 的数据类型选择相应的 CUDA 内核函数
    #      如果 scale_list 是 torch.half,选择 kernels.int4WeightExtractionHalf 函数
    #      如果 scale_list 是 torch.bfloat16,选择 kernels.int4WeightExtractionBFloat16 函数
    elif source_bit_width == 4:
        func = (
            kernels.int4WeightExtractionHalf if scale_list.dtype == torch.half else kernels.int4WeightExtractionBFloat16
        )
    #   5) 如果 source_bit_width 不是 4 或 8,抛出异常,表示不支持的位宽
    else:
        assert False, "Unsupported bit-width"
    #   6) 使用 with torch.cuda.device(weight.device) 上下文管理器,确保在正确的 CUDA 设备上执行操作
    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)    #   7) 获取权重张量的形状 (n, m)
        #   8) 创建一个新的张量 out,形状为 (n, m * (8 // source_bit_width)),数据类型为 scale_list.dtype,放置在 CUDA 设备上
        out = torch.empty(n, m * (8 // source_bit_width), dtype=scale_list.dtype, device="cuda")
         #   9) 获取当前 CUDA 流
        stream = torch.cuda.current_stream()
        #   10) 计算 CUDA 内核的网格维度 gridDim 和块维度 blockDim,与 compress_int4_weight 函数中的计算方式相同
        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)
         #   11) 调用相应的 CUDA 内核函数,将量化的权重提取为 float16 或 bfloat16 类型
        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out  #   12) 返回提取后的权重张量 out (float16 或 bfloat16 类型的权重)

# QuantizedLinear 类的构造函数,量化线性层
class QuantizedLinear(torch.nn.Module):
    def __init__(self, 
                weight_bit_width: int,  #   1) weight_bit_width: 量化权重所使用的位宽,可以是 4 或 8
                weight,                  #   2) weight: 原始的浮点权重张量,如果为 None 或 empty_init 为 True,则随机初始化权重
                bias=None,               #   3) bias: 原始的浮点偏置向量,如果为 None,则不使用偏置
                device="cpu",           #   4) device: 指定权重和偏置所在的设备,可以是 "cpu" 或 "cuda"
                dtype=None,             #   5) dtype: 指定缩放因子的数据类型,通常为 torch.half 或 torch.bfloat16
                empty_init=False,       #   6) empty_init: 一个布尔值,指示是否随机初始化权重
                *args,
                **kwargs):
        super().__init__()
        self.weight_bit_width = weight_bit_width

        shape = weight.shape
        #   2) 如果原始权重为 None 或 empty_init 为 True,随机初始化量化权重和缩放因子
        if weight is None or empty_init:
            self.weight = torch.empty(shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=device)
            self.weight_scale = torch.empty(shape[0], dtype=dtype, device=device)
        else:  
            #   3) 否则,根据原始权重计算量化权重和缩放因子
            self.weight_scale = weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)
            #   4) 对权重进行量化, 如果 weight_bit_width 为 4,使用 compress_int4_weight 函数进行压缩
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        # 将量化权重和缩放因子包装为 PyTorch 的 Parameter 对象,并指定 requires_grad=False
        self.weight = Parameter(self.weight.to(device), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(device), requires_grad=False)
        self.bias = Parameter(bias.to(device), requires_grad=False) if bias is not None else None

    def forward(self, input):
        # 15. forward 方法实现了量化线性层的前向传播
        #   1) 调用 W8A16Linear.apply 函数,传入输入数据、量化权重、缩放因子和权重位宽
        output = W8A16Linear.apply(input, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output = output + self.bias
        return output


def quantize(
                model, 
                weight_bit_width, 
                empty_init=False, 
                device=None
            ):
    """Replace fp16 linear with quantized linear"""
    for layer in model.layers:
        layer.self_attention.query_key_value = QuantizedLinear(
            # weight_bit_width: 量化权重的位宽
            weight_bit_width=weight_bit_width,  
            # weight: 原始的浮点权重张量,首先将其移动到 CUDA 设备上     
            weight=layer.self_attention.query_key_value.weight.to(torch.cuda.current_device()),
            # bias: 原始的偏置向量
            bias=layer.self_attention.query_key_value.bias,
            # dtype: 原始权重的数据类型,用于初始化缩放因子
            dtype=layer.self_attention.query_key_value.weight.dtype,
            # device: 指定量化权重和缩放因子所在的设备,如果为 None,则使用原始权重所在的设备
            device=layer.self_attention.query_key_value.weight.device if device is None else device,
            # empty_init: 指示是否随机初始化权重
            empty_init=empty_init
        )
        layer.self_attention.dense = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.self_attention.dense.weight.to(torch.cuda.current_device()),
            bias=layer.self_attention.dense.bias,
            dtype=layer.self_attention.dense.weight.dtype,
            device=layer.self_attention.dense.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.mlp.dense_h_to_4h = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_h_to_4h.weight.to(torch.cuda.current_device()),
            bias=layer.mlp.dense_h_to_4h.bias,
            dtype=layer.mlp.dense_h_to_4h.weight.dtype,
            device=layer.mlp.dense_h_to_4h.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.mlp.dense_4h_to_h = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_4h_to_h.weight.to(torch.cuda.current_device()),
            bias=layer.mlp.dense_4h_to_h.bias,
            dtype=layer.mlp.dense_4h_to_h.weight.dtype,
            device=layer.mlp.dense_4h_to_h.weight.device if device is None else device,
            empty_init=empty_init
        )

    return model
