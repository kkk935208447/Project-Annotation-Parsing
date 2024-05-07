"""
Helpers to support streaming generate output.
Borrowed from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/callbacks.py
"""
# 这是 Python 的内置垃圾回收(Garbage Collection)模块。
# 在 Iteratorize 类中,可能会产生一些临时对象和内存泄漏,使用 gc 模块可以帮助自动回收这些内存,以优化内存使用。
import gc
# 这是 Python 的内置异常处理模块。
# 在 Iteratorize 类的 gentask 函数中,如果执行原始函数时发生异常,会使用 traceback 模块打印出异常的堆栈信息,便于问题的诊断和调试。
import traceback
# 这是 Python 的内置队列模块。
# 在 Iteratorize 类中,使用 Queue 作为临时存储生成结果的容器,可以实现异步生成和动态获取的功能。
from queue import Queue
# 这是 Python 的内置多线程模块。
# 在 Iteratorize 类中,使用 Thread 创建了一个新线程来执行原始函数,从而避免了主线程的阻塞。
from threading import Thread

import torch
import transformers

"""
这段代码主要使用了以下技术:
继承和抽象基类: Stream 类继承自 transformers.StoppingCriteria 抽象基类,实现了 __call__ 方法来定义文本生成的停止条件。
生成器模式: Iteratorize 类使用了生成器模式,将一个接受回调函数的函数转换为一个惰性迭代器(generator)。这样可以在生成文本的过程中动态获取中间结果,而不必等待整个生成过程完成。
多线程: Iteratorize 类创建了一个新线程来执行原始函数,并将结果放入 Queue 中,从而避免了主线程的阻塞。
异常处理: Iteratorize 类中的 gentask 函数使用了 try-except 块来捕获可能发生的异常,并打印 traceback 信息。

这段代码主要解决了以下问题:
灵活的文本生成: 通过 Stream 类,可以在文本生成过程中动态监控和控制生成过程,例如在每生成一个 token 时调用回调函数。这对于需要实时监控或交互式生成的场景很有帮助。
异步文本生成: Iteratorize 类将文本生成过程与结果获取过程分离,使得生成过程可以在后台异步执行,而主线程可以动态地获取中间结果,而不必等待整个生成过程完成。这可以提高用户体验,特别是在生成时间较长的情况下。
异常处理: Iteratorize 类中的异常处理机制可以更好地应对生成过程中可能出现的各种异常情况,并提供相应的错误信息,有助于问题的诊断和解决。

总的来说,这段代码展示了如何使用 Python 中的面向对象编程、多线程和异常处理等技术,来构建一个灵活、异步和健壮的文本生成系统。这种设计模式可以广泛应用于需要动态获取中间结果或处理异常情况的各种 AI 应用场景中。
"""

# 1. 使用 transformers 库中的 StoppingCriteria 类实现了 Stream 类。
# StoppingCriteria 是一个抽象基类,用于定义文本生成时的停止条件。
# Stream 类继承自 StoppingCriteria,并实现了 __call__ 方法,用于在每生成一个 token 时检查是否需要停止生成。
class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        # 1.1. 保存用户传入的回调函数,在生成每个 token 时会调用该函数。
        self.callback_func = callback_func

    # 1.2. 实现 __call__ 方法,该方法会在生成每个 token 时被调用。
    # 如果存在回调函数 callback_func,则调用它并传入当前生成的 token。
    # 最后返回 False,表示应该停止生成。
    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

# 2. 使用生成器(generator)模式实现了 Iteratorize 类。
# 生成器模式可以将一个接受回调函数的函数转换为一个惰性迭代器,
# 这样在生成文本的过程中就可以动态获取中间结果,而不必等待整个生成过程完成。
class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        # 2.1. 保存原始函数 func、用户传入的回调函数 callback,以及关键字参数 kwargs。
        self.mfunc = func
        self.c_callback = callback
         # 2.2. 创建一个 Queue 用于存放生成的结果,并定义一个结束标记 sentinel。
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False
        # 2.3. 定义一个内部 _callback 函数,用于将生成的 token 放入 Queue 中。
        # 如果遇到停止标记,则抛出 ValueError 异常。
        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        # 2.4. 定义 gentask 函数,在新线程中执行原始函数 func,并将结果放入 Queue 中。
        # 如果遇到异常,会打印 traceback 信息。
        # 最后向 Queue 中放入结束标记,并调用用户传入的回调函数(如果有)。
        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)
        # 2.5. 启动新线程执行 gentask 函数。
        self.thread = Thread(target=gentask)
        self.thread.start()
    # 2.6. 实现 __iter__ 和 __next__ 方法,使得 Iteratorize 对象可以作为迭代器使用。
    # 每次调用 __next__ 方法,都会从 Queue 中获取下一个结果,直到遇到结束标记。
    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj
    # 2.7. 实现 __enter__ 和 __exit__ 方法,使得 Iteratorize 对象可以作为上下文管理器使用。
    # 在退出时,设置 stop_now 标记以停止迭代。
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
