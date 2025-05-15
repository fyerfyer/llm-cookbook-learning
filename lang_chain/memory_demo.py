from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from dotenv import load_dotenv, find_dotenv

# 从helper模块导入工具函数
from helper.helper import create_client

class LLMBuilder:
    """创建自定义语言模型接口，避开langchain兼容性问题"""

    def __init__(self, temperature=0.0):
        self.client = create_client()
        self.temperature = temperature

    def create_llm(self):
        """创建自定义语言模型接口"""

        def invoke(messages):
            # 转换LangChain消息格式到通义千问支持的格式
            api_messages = []

            # 处理ChatPromptValue对象 - 这是从ChatPromptTemplate传来的
            from langchain_core.prompt_values import ChatPromptValue
            if isinstance(messages, ChatPromptValue):
                # 从ChatPromptValue中提取消息
                messages_list = messages.to_messages()

                for msg in messages_list:
                    role = msg.type
                    # 映射角色
                    if role == "human":
                        role = "user"
                    elif role == "ai":
                        role = "assistant"
                    # 确保角色有效
                    if role not in ["system", "assistant", "user", "tool", "function"]:
                        role = "user"
                    api_messages.append({"role": role, "content": msg.content})
            # 处理tuple格式
            elif len(messages) == 1 and isinstance(messages[0], tuple) and messages[0][0] == "messages":
                # 从tuple中获取实际消息列表
                msg_list = messages[0][1]

                # 处理列表中的每个消息
                for msg in msg_list:
                    role = msg.type
                    if role == "human":
                        role = "user"
                    elif role == "ai":
                        role = "assistant"
                    if role not in ["system", "assistant", "user", "tool", "function"]:
                        role = "user"
                    api_messages.append({"role": role, "content": msg.content})
            # 处理常规消息列表
            else:
                for idx, msg in enumerate(messages):

                    if hasattr(msg, 'type'):
                        role = msg.type
                        # 确保角色名称正确映射到API支持的值
                        if role == "human":
                            role = "user"
                        elif role == "ai":
                            role = "assistant"
                        # 添加确保角色名为有效值的检查
                        if role not in ["system", "assistant", "user", "tool", "function"]:
                            role = "user"
                        api_messages.append({"role": role, "content": msg.content})
                    elif isinstance(msg, tuple):
                        role, content = msg
                        # 将LangChain的角色名映射到通义千问支持的角色
                        if role == "human":
                            role = "user"
                        api_messages.append({"role": role, "content": content})

            api_response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=api_messages,
                temperature=self.temperature,
            )
            return api_response.choices[0].message.content

        return invoke


class ConversationBufferDemo:
    """演示基本的对话缓存"""

    def __init__(self, temperature=0.0):
        self.llm_builder = LLMBuilder(temperature=temperature)
        self.llm = self.llm_builder.create_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是一个有帮助的助手，能够清晰记住对话历史。"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.chain = self.prompt | self.llm

    def chat(self, user_message):
        """使用简单的消息传递进行对话"""
        return self.chain.invoke({"messages": user_message})

    def demonstrate_manual_memory(self):
        """演示手动添加和检索对话记忆"""
        print("\n=== 手动对话记忆演示 ===")

        # 创建一些示例消息
        messages = [
            HumanMessage(content="你好，我叫皮皮鲁"),
            AIMessage(content="你好啊，皮皮鲁！很高兴认识你。我是一个AI助手，有什么能帮到你吗？"),
            HumanMessage(content="1+1等于多少？"),
            AIMessage(content="1+1等于2。"),
        ]

        # 添加新消息并获取回复
        response = self.chat(messages + [HumanMessage(content="我叫什么名字？")])
        print("\n用户: 我叫什么名字？")
        print(f"AI: {response}")

        # 添加不相关问题，验证记忆能力
        messages.append(HumanMessage(content="我叫什么名字？"))
        messages.append(AIMessage(content=response))
        response = self.chat(messages + [HumanMessage(content="2+2等于多少？")])
        print("\n用户: 2+2等于多少？")
        print(f"AI: {response}")

        # 再次询问名字，测试记忆保留
        messages.append(HumanMessage(content="2+2等于多少？"))
        messages.append(AIMessage(content=response))
        response = self.chat(messages + [HumanMessage(content="再告诉我我的名字")])
        print("\n用户: 再告诉我我的名字")
        print(f"AI: {response}")


class MemoryManager:
    """管理不同类型的记忆实现"""

    def __init__(self, temperature=0.0):
        self.llm_builder = LLMBuilder(temperature)
        self.llm = self.llm_builder.create_llm()

    def demonstrate_langgraph_memory(self):
        """演示使用LangGraph的内存状态管理"""
        print("\n=== LangGraph记忆管理演示 ===")

        # 创建工作流图
        workflow = StateGraph(state_schema=MessagesState)

        # 定义调用模型的函数
        def call_model(state: MessagesState):
            system_prompt = "你是一个有帮助的助手。尽力回答所有问题。"
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = self.llm(messages)
            return {"messages": response}

        # 定义节点和边
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")

        # 添加简单的内存存储器
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        # 执行对话
        print("\n第一轮对话:")
        result = app.invoke(
            {"messages": [HumanMessage(content="你好，我叫皮皮鲁")]},
            config={"configurable": {"thread_id": "demo1"}}
        )
        print(f"用户: 你好，我叫皮皮鲁")
        print(f"AI: {result['messages'][-1].content}")

        print("\n第二轮对话:")
        result = app.invoke(
            {"messages": [HumanMessage(content="1+1等于多少?")]},
            config={"configurable": {"thread_id": "demo1"}}
        )
        print(f"用户: 1+1等于多少?")
        print(f"AI: {result['messages'][-1].content}")

        print("\n第三轮对话 (测试记忆):")
        result = app.invoke(
            {"messages": [HumanMessage(content="我叫什么名字?")]},
            config={"configurable": {"thread_id": "demo1"}}
        )
        print(f"用户: 我叫什么名字?")
        print(f"AI: {result['messages'][-1].content}")

    def demonstrate_window_memory(self):
        """演示窗口记忆实现"""
        print("\n=== 窗口记忆演示 (k=1) ===")

        # 创建有限窗口大小的工作流
        workflow = StateGraph(state_schema=MessagesState)

        # 定义带窗口限制的调用模型函数
        def call_model_with_window(state: MessagesState):
            # 只保留最新的一条消息
            if len(state["messages"]) > 1:
                recent_messages = state["messages"][-1:]
            else:
                recent_messages = state["messages"]

            system_prompt = "你是一个有帮助的助手。尽力回答所有问题。"
            messages = [SystemMessage(content=system_prompt)] + recent_messages
            response = self.llm(messages)
            return {"messages": response}

        # 定义节点和边
        workflow.add_node("model", call_model_with_window)
        workflow.add_edge(START, "model")

        # 添加简单的内存存储器
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        # 执行对话序列
        print("\n第一轮对话:")
        result = app.invoke(
            {"messages": [HumanMessage(content="你好，我叫皮皮鲁")]},
            config={"configurable": {"thread_id": "demo2"}}
        )
        print(f"用户: 你好，我叫皮皮鲁")
        print(f"AI: {result['messages'][-1].content}")

        print("\n第二轮对话:")
        result = app.invoke(
            {"messages": [HumanMessage(content="1+1等于多少?")]},
            config={"configurable": {"thread_id": "demo2"}}
        )
        print(f"用户: 1+1等于多少?")
        print(f"AI: {result['messages'][-1].content}")

        print("\n第三轮对话 (测试记忆限制):")
        result = app.invoke(
            {"messages": [HumanMessage(content="我叫什么名字?")]},
            config={"configurable": {"thread_id": "demo2"}}
        )
        print(f"用户: 我叫什么名字?")
        print(f"AI: {result['messages'][-1].content}")

    def demonstrate_summary_memory(self):
        """演示摘要记忆实现"""
        print("\n=== 摘要记忆演示 ===")

        # 创建工作流图
        workflow = StateGraph(state_schema=MessagesState)

        # 用于保存摘要的变量
        current_summary = ""

        # 定义调用模型的函数，包括摘要生成
        def call_model_with_summary(state: MessagesState):
            nonlocal current_summary

            system_prompt = "你是一个有帮助的助手。尽力回答所有问题。"
            system_message = SystemMessage(content=system_prompt)

            # 如果消息历史足够长，生成摘要
            if len(state["messages"]) >= 4 and not current_summary:
                # 创建摘要提示
                summary_prompt = "请将以上对话内容总结成一个简短的摘要，保留重要的细节。"
                summary_message = self.llm([SystemMessage(content=system_prompt)] +
                                           state["messages"] +
                                           [HumanMessage(content=summary_prompt)])
                current_summary = summary_message

            # 使用最新消息和摘要
            if current_summary:
                last_human_message = state["messages"][-1]
                messages = [system_message,
                            SystemMessage(content=f"对话摘要: {current_summary}"),
                            last_human_message]
            else:
                messages = [system_message] + state["messages"]

            response = self.llm(messages)
            return {"messages": response}

        # 定义节点和边
        workflow.add_node("model", call_model_with_summary)
        workflow.add_edge(START, "model")

        # 添加简单的内存存储器
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        # 创建一段日程安排的长文本
        schedule = """在八点你和你的产品团队有一个会议。
        你需要做一个PPT。
        上午9点到12点你需要忙于LangChain。
        Langchain是一个有用的工具，因此你的项目进展的非常快。
        中午，在意大利餐厅与一位开车来的顾客共进午餐
        走了一个多小时的路程与你见面，只为了解最新的 AI。
        确保你带了笔记本电脑可以展示最新的 LLM 样例."""

        # 执行对话序列
        print("\n第一轮对话:")
        result = app.invoke(
            {"messages": [HumanMessage(content="你好，我叫皮皮鲁")]},
            config={"configurable": {"thread_id": "demo3"}}
        )
        print(f"用户: 你好，我叫皮皮鲁")
        print(f"AI: {result['messages'][-1].content}")

        print("\n第二轮对话:")
        result = app.invoke(
            {"messages": [HumanMessage(content="很高兴和你成为朋友！")]},
            config={"configurable": {"thread_id": "demo3"}}
        )
        print(f"用户: 很高兴和你成为朋友！")
        print(f"AI: {result['messages'][-1].content}")

        print("\n第三轮对话:")
        result = app.invoke(
            {"messages": [HumanMessage(content="今天的日程安排是什么？")]},
            config={"configurable": {"thread_id": "demo3"}}
        )
        print(f"用户: 今天的日程安排是什么？")
        print(f"AI: {result['messages'][-1].content}")

        # 添加日程安排回复，触发摘要生成
        result = app.invoke(
            {"messages": [AIMessage(content=schedule)]},
            config={"configurable": {"thread_id": "demo3"}}
        )

        # 生成摘要后的新问题
        print("\n第四轮对话 (基于摘要):")
        result = app.invoke(
            {"messages": [HumanMessage(content="展示什么样的样例最好呢？")]},
            config={"configurable": {"thread_id": "demo3"}}
        )
        print(f"用户: 展示什么样的样例最好呢？")
        print(f"AI: {result['messages'][-1].content}")


def main():
    """主函数，运行各种记忆演示"""
    load_dotenv(find_dotenv())  # 加载环境变量

    print("=== LangChain记忆组件演示 ===")

    # 演示基本的消息传递
    convo_demo = ConversationBufferDemo()
    convo_demo.demonstrate_manual_memory()

    # 演示不同类型的记忆机制
    memory_manager = MemoryManager()
    memory_manager.demonstrate_langgraph_memory()
    memory_manager.demonstrate_window_memory()
    memory_manager.demonstrate_summary_memory()


if __name__ == "__main__":
    main()