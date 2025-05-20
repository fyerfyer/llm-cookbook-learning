from helper.helper import get_completion
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional


class TongyiQianwenLLM(LLM):
    """
    通义千问自定义LLM实现
    """
    model_name: str = "qwen-turbo"
    temperature: float = 0.0
    
    @property
    def _llm_type(self) -> str:
        return "tongyi_qianwen"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 使用helper中的函数调用通义千问
        return get_completion(prompt, model=self.model_name, temperature=self.temperature)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}
    
    def get_num_tokens(self, text: str) -> int:
        """简单估算token数量，用于摘要记忆功能"""
        return len(text) // 2  # 简单地将字符数除以2作为估计值


class MemoryDemo:
    """记忆模块演示基础类"""
    
    def __init__(self, temperature=0.0):
        self.llm = TongyiQianwenLLM(temperature=temperature)
        self.memory = None
        self.conversation = None

    def show_memory_content(self):
        """显示内存中存储的内容"""
        if not self.memory:
            print("记忆对象尚未初始化")
            return
        
        try:
            memory_vars = self.memory.load_memory_variables({})
            print("当前记忆内容:", memory_vars)
            return memory_vars
        except Exception as e:
            print(f"获取记忆内容出错: {e}")


class ConversationBufferMemoryDemo(MemoryDemo):
    """对话缓存储存演示类"""
    
    def __init__(self, temperature=0.0, verbose=False):
        super().__init__(temperature)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            verbose=verbose
        )
    
    def add_to_memory(self, input_text, output_text):
        """直接添加内容到储存缓存"""
        self.memory.save_context({"input": input_text}, {"output": output_text})
        
    def chat(self, user_input):
        """进行对话并返回结果"""
        return self.conversation.predict(input=user_input)


class ConversationBufferWindowMemoryDemo(MemoryDemo):
    """对话缓存窗口储存演示类"""
    
    def __init__(self, window_size=1, temperature=0.0, verbose=False):
        super().__init__(temperature)
        self.memory = ConversationBufferWindowMemory(k=window_size)
        self.conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            verbose=verbose
        )
    
    def add_to_memory(self, input_text, output_text):
        """直接添加内容到储存缓存"""
        self.memory.save_context({"input": input_text}, {"output": output_text})
        
    def chat(self, user_input):
        """进行对话并返回结果"""
        return self.conversation.predict(input=user_input)


class ConversationTokenBufferMemoryDemo(MemoryDemo):
    """对话令牌缓存储存演示类"""
    
    def __init__(self, max_token_limit=100, temperature=0.0, verbose=False):
        super().__init__(temperature)
        self.memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=max_token_limit)
        self.conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            verbose=verbose
        )
    
    def add_to_memory(self, input_text, output_text):
        """直接添加内容到储存缓存"""
        self.memory.save_context({"input": input_text}, {"output": output_text})
        
    def chat(self, user_input):
        """进行对话并返回结果"""
        return self.conversation.predict(input=user_input)


class ConversationSummaryBufferMemoryDemo(MemoryDemo):
    """对话摘要缓存储存演示类"""
    
    def __init__(self, max_token_limit=100, temperature=0.0, verbose=False):
        super().__init__(temperature)
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=max_token_limit)
        self.conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            verbose=verbose
        )
    
    def add_to_memory(self, input_text, output_text):
        """直接添加内容到储存缓存"""
        self.memory.save_context({"input": input_text}, {"output": output_text})
        
    def chat(self, user_input):
        """进行对话并返回结果"""
        return self.conversation.predict(input=user_input)


def demo_buffer_memory():
    """演示基本的对话缓存储存功能"""
    print("===== 对话缓存储存演示 =====")
    
    # 初始化对话缓存记忆演示
    buffer_demo = ConversationBufferMemoryDemo(verbose=True)
    
    # 第一轮对话
    print("\n第一轮对话：")
    response1 = buffer_demo.chat("你好, 我叫皮皮鲁")
    print(response1)
    
    # 第二轮对话
    print("\n第二轮对话：")
    response2 = buffer_demo.chat("1+1等于多少？")
    print(response2)
    
    # 第三轮对话 - 测试记忆能力
    print("\n第三轮对话：")
    response3 = buffer_demo.chat("我叫什么名字？")
    print(response3)
    
    # 查看记忆内容
    print("\n查看记忆内容：")
    buffer_demo.show_memory_content()
    
    # 直接添加内容到记忆
    print("\n手动添加记忆测试：")
    new_buffer_demo = ConversationBufferMemoryDemo()
    new_buffer_demo.add_to_memory("你好，我叫皮皮鲁", "你好啊，我叫鲁西西")
    new_buffer_demo.add_to_memory("很高兴和你成为朋友！", "是的，让我们一起去冒险吧！")
    new_buffer_demo.show_memory_content()


def demo_window_memory():
    """演示对话缓存窗口储存功能"""
    print("\n\n===== 对话缓存窗口储存演示 =====")
    
    # 初始化一个记忆窗口大小为1的演示对象
    window_demo = ConversationBufferWindowMemoryDemo(window_size=1)
    
    # 手动添加两轮对话
    print("\n添加两轮对话到窗口记忆：")
    window_demo.add_to_memory("你好，我叫皮皮鲁", "你好啊，我叫鲁西西")
    window_demo.add_to_memory("很高兴和你成为朋友！", "是的，让我们一起去冒险吧！")
    window_demo.show_memory_content()
    
    # 使用对话链进行测试
    print("\n使用带窗口记忆的对话链：")
    
    # 新建一个窗口记忆对话链用于演示
    window_chain_demo = ConversationBufferWindowMemoryDemo(window_size=1)
    
    print("第一轮对话：")
    response1 = window_chain_demo.chat("你好, 我叫皮皮鲁")
    print(response1)
    
    print("\n第二轮对话：")
    response2 = window_chain_demo.chat("1+1等于多少？")
    print(response2)
    
    print("\n第三轮对话：")
    response3 = window_chain_demo.chat("我叫什么名字？")
    print(response3)
    
    # 查看记忆窗口内容
    print("\n查看记忆窗口内容：")
    window_chain_demo.show_memory_content()


def demo_token_buffer_memory():
    """演示对话令牌缓存储存功能"""
    print("\n\n===== 对话令牌缓存储存演示 =====")
    
    # 初始化一个令牌限制为30的演示对象
    token_demo = ConversationTokenBufferMemoryDemo(max_token_limit=30)
    
    # 添加对话到令牌缓存
    print("\n添加诗句到令牌缓存：")
    token_demo.add_to_memory("朝辞白帝彩云间，", "千里江陵一日还。")
    token_demo.add_to_memory("两岸猿声啼不住，", "轻舟已过万重山。")
    
    # 查看令牌缓存内容
    print("令牌缓存内容：")
    token_demo.show_memory_content()


def demo_summary_buffer_memory():
    """演示对话摘要缓存储存功能"""
    print("\n\n===== 对话摘要缓存储存演示 =====")
    
    # 创建一个长字符串作为日程安排
    schedule = "在八点你和你的产品团队有一个会议。 \
    你需要做一个PPT。 \
    上午9点到12点你需要忙于LangChain。\
    Langchain是一个有用的工具，因此你的项目进展的非常快。\
    中午，在意大利餐厅与一位开车来的顾客共进午餐 \
    走了一个多小时的路程与你见面，只为了解最新的 AI。 \
    确保你带了笔记本电脑可以展示最新的 LLM 样例."
    
    # 初始化一个摘要缓存演示对象，减小token限制
    summary_demo = ConversationSummaryBufferMemoryDemo(max_token_limit=50, verbose=True)
    
    try:
        # 添加对话到摘要缓存
        print("\n添加对话到摘要缓存：")
        summary_demo.add_to_memory("你好，我叫皮皮鲁", "你好啊，我叫鲁西西")
        summary_demo.add_to_memory("很高兴和你成为朋友！", "是的，让我们一起去冒险吧！")
        summary_demo.add_to_memory("今天的日程安排是什么？", schedule)
        
        # 查看摘要缓存内容
        print("\n摘要缓存内容：")
        summary_demo.show_memory_content()
        
        # 进行新的对话，测试摘要更新
        print("\n进行新的对话，测试摘要更新：")
        response = summary_demo.chat("展示什么样的样例最好呢？")
        print(f"回复: {response}")
        
        # 再次查看摘要缓存内容
        print("\n更新后的摘要缓存内容：")
        summary_demo.show_memory_content()
    
    except Exception as e:
        print(f"摘要记忆演示出错: {e}")
        print("跳过摘要记忆演示...")


def main():
    # 演示基本缓存记忆功能
    demo_buffer_memory()
    
    # 演示窗口缓存记忆功能
    demo_window_memory()
    
    # 演示令牌缓存记忆功能
    demo_token_buffer_memory()
    
    # 演示摘要缓存记忆功能
    try:
        demo_summary_buffer_memory()
    except ImportError as e:
        print(f"\n\n===== 对话摘要缓存储存演示 =====")
        print(f"无法导入必要的依赖: {e}")
        print("请安装transformers库或使用自定义token计算方法。")


if __name__ == "__main__":
    main()