from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda
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
            # 处理常规消息列表
            else:
                for msg in messages:
                    if hasattr(msg, 'type'):
                        role = msg.type
                        # 确保角色名称正确映射
                        if role == "human":
                            role = "user"
                        elif role == "ai":
                            role = "assistant"
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


class LLMChainDemo:
    """演示基本的大语言模型链"""

    def __init__(self, temperature=0.0):
        """初始化LLM链演示"""
        self.llm_builder = LLMBuilder(temperature)
        self.llm = self.llm_builder.create_llm()

    def run_basic_chain(self, product):
        """运行基本的大语言模型链"""
        # 初始化提示模板
        prompt = ChatPromptTemplate.from_template(
            "描述制造{product}的一个公司的最佳名称是什么?"
        )

        # 构建链
        chain = prompt | self.llm | StrOutputParser()

        # 运行链
        result = chain.invoke({"product": product})
        return result


class SequentialChainDemo:
    """演示顺序链"""

    def __init__(self, temperature=0.9):
        """初始化顺序链演示"""
        self.llm_builder = LLMBuilder(temperature)
        self.llm = self.llm_builder.create_llm()

    def run_simple_sequential_chain(self, product):
        """运行简单顺序链"""
        # 提示模板1：这个提示将接受产品并返回最佳名称来描述该公司
        first_prompt = ChatPromptTemplate.from_template(
            "描述制造{product}的一个公司的最好的名称是什么"
        )

        # 提示模板2：接受公司名称，然后输出该公司的长为20个单词的描述
        second_prompt = ChatPromptTemplate.from_template(
            "写一个20字的描述对于下面这个公司：{company_name}"
        )

        # 构建第一个链
        chain_one = first_prompt | self.llm | StrOutputParser()

        # 构建第二个链
        chain_two = second_prompt | self.llm | StrOutputParser()

        # 使用纯函数式方法创建组合链
        def run_chain(product_input):
            company_name = chain_one.invoke({"product": product_input})
            result = chain_two.invoke({"company_name": company_name})
            return result

        # 运行组合链
        result = run_chain(product)
        print(f"> Entering new SimpleSequentialChain chain...\n{product}\n{result}\n\n> Finished chain.")
        return result

    def run_sequential_chain(self, review):
        """运行多输入多输出的顺序链"""
        # 子链1：翻译成英语
        first_prompt = ChatPromptTemplate.from_template(
            "把下面的评论review翻译成英文:\n\n{Review}"
        )
        chain_one = first_prompt | self.llm | StrOutputParser()

        # 子链2：用一句话总结
        second_prompt = ChatPromptTemplate.from_template(
            "请你用一句话来总结下面的评论review:\n\n{English_Review}"
        )
        chain_two = second_prompt | self.llm | StrOutputParser()

        # 子链3：确定语言
        third_prompt = ChatPromptTemplate.from_template(
            "下面的评论review使用的什么语言:\n\n{Review}"
        )
        chain_three = third_prompt | self.llm | StrOutputParser()

        # 子链4：写后续回复
        fourth_prompt = ChatPromptTemplate.from_template(
            "使用特定的语言对下面的总结写一个后续回复:"
            "\n\n总结: {summary}\n\n语言: {language}"
        )
        chain_four = fourth_prompt | self.llm | StrOutputParser()

        # 构建多步骤链
        def run_chains(input_dict):
            # 翻译评论
            english_review = chain_one.invoke({"Review": input_dict["Review"]})

            # 总结英文评论
            summary = chain_two.invoke({"English_Review": english_review})

            # 确定语言
            language = chain_three.invoke({"Review": input_dict["Review"]})

            # 生成后续回复
            followup = chain_four.invoke({
                "summary": summary,
                "language": language
            })

            return {
                "Review": input_dict["Review"],
                "English_Review": english_review,
                "summary": summary,
                "language": language,
                "followup_message": followup
            }

        # 执行链并返回结果
        print("> Entering new SequentialChain chain...")
        result = RunnableLambda(run_chains).invoke({"Review": review})
        print("> Finished chain.")

        return result


class RouterChainDemo:
    """演示路由链"""

    def __init__(self, temperature=0.0):
        """初始化路由链演示"""
        self.llm_builder = LLMBuilder(temperature)
        self.llm = self.llm_builder.create_llm()

        # 定义提示模板
        self.physics_template = """你是一个非常聪明的物理专家。
你擅长用一种简洁并且易于理解的方式去回答问题。
当你不知道问题的答案时，你承认你不知道。

这是一个问题:
{input}"""

        self.math_template = """你是一个非常优秀的数学家。
你擅长回答数学问题。
你之所以如此优秀，是因为你能够将棘手的问题分解为组成部分，
回答组成部分，然后将它们组合在一起，回答更广泛的问题。

这是一个问题：
{input}"""

        self.history_template = """你是一位非常优秀的历史学家。
你对一系列历史时期的人物、事件和背景有着极好的学识和理解
你有能力思考、反思、辩证、讨论和评估过去。
你尊重历史证据，并有能力利用它来支持你的解释和判断。

这是一个问题:
{input}"""

        self.computerscience_template = """你是一个成功的计算机科学专家。
你有创造力、协作精神、前瞻性思维、自信、解决问题的能力、
对理论和算法的理解以及出色的沟通技巧。
你非常擅长回答编程问题。
你之所以如此优秀，是因为你知道如何通过以机器可以轻松解释的命令式步骤描述解决方案来解决问题，
并且你知道如何选择在时间复杂性和空间复杂性之间取得良好平衡的解决方案。

这是一个问题：
{input}"""

    def run_router_chain(self, question):
        """运行路由链"""
        # 创建提示信息
        prompt_infos = [
            {"name": "物理学", "description": "擅长回答关于物理学的问题", "template": self.physics_template},
            {"name": "数学", "description": "擅长回答数学问题", "template": self.math_template},
            {"name": "历史", "description": "擅长回答历史问题", "template": self.history_template},
            {"name": "计算机科学", "description": "擅长回答计算机科学问题", "template": self.computerscience_template}
        ]

        # 创建路由选择函数
        def choose_route(input_text):
            # 创建路由选择提示
            destinations = "\n".join([f"{p['name']}: {p['description']}" for p in prompt_infos])
            router_template = f"""给语言模型一个原始文本输入，
让其选择最适合输入的模型提示。
系统将为您提供可用提示的名称以及最适合该提示的描述。

<< 候选提示 >>
{destinations}

<< 输入 >>
{input_text}

分析这个问题属于哪个类别，只需要回答类别名称，不需要解释。
"""
            router_prompt = ChatPromptTemplate.from_template(router_template)
            route_chain = router_prompt | self.llm | StrOutputParser()
            route = route_chain.invoke({})

            # 清理路由输出并找到最佳匹配
            route = route.strip().lower()
            for p_info in prompt_infos:
                if p_info["name"].lower() in route:
                    return p_info

            # 默认使用通用模板
            default_template = """你是一个有帮助的AI助手。
请用简洁明了的方式回答以下问题:
{input}"""
            return {"name": "默认", "description": "通用回答", "template": default_template}

        # 执行路由选择并处理问题
        best_route = choose_route(question)
        print(f"> Entering new MultiPromptChain chain...\n{best_route['name']}: {{'input': '{question}'}}")

        # 使用选定的模板创建提示
        selected_prompt = ChatPromptTemplate.from_template(best_route["template"])
        response_chain = selected_prompt | self.llm | StrOutputParser()

        # 处理问题
        response = response_chain.invoke({"input": question})
        print("> Finished chain.")

        return response


def main():
    """主函数，演示各种链"""
    load_dotenv(find_dotenv())  # 加载环境变量

    print("=== LangChain 模型链演示 ===\n")

    # 演示基本的LLM链
    print("=== 1. 基本 LLM 链 ===")
    llm_chain_demo = LLMChainDemo()
    product = "大号床单套装"
    result = llm_chain_demo.run_basic_chain(product)
    print(f"产品: {product}")
    print(f"生成的公司名称: {result}\n")

    # 演示简单顺序链
    print("=== 2. 简单顺序链 ===")
    seq_chain_demo = SequentialChainDemo(temperature=0.9)
    result = seq_chain_demo.run_simple_sequential_chain("大号床单套装")
    print(f"最终结果: {result}\n")

    # 演示多输入多输出的顺序链
    print("=== 3. 多输入多输出顺序链 ===")
    # 示例评论
    review = "Je trouve le goût médiocre. La mousse ne tient pas, c'est bizarre. J'achète les mêmes dans le commerce et le goût est bien meilleur...\nVieux lot ou contrefaçon !?"
    result = seq_chain_demo.run_sequential_chain(review)
    print("\n多输入多输出顺序链结果:")
    print(f"原始评论: {result['Review']}")
    print(f"英文翻译: {result['English_Review']}")
    print(f"总结: {result['summary']}")
    print(f"语言: {result['language']}")
    print(f"后续回复: {result['followup_message']}\n")

    # 演示路由链
    print("=== 4. 路由链 ===")
    router_demo = RouterChainDemo()

    print("\n物理学问题:")
    physics_question = "什么是黑体辐射？"
    physics_result = router_demo.run_router_chain(physics_question)
    print(f"问题: {physics_question}")
    print(f"回答: {physics_result}\n")

    print("\n数学问题:")
    math_question = "2+2等于多少？"
    math_result = router_demo.run_router_chain(math_question)
    print(f"问题: {math_question}")
    print(f"回答: {math_result}\n")

    print("\n生物学问题 (默认路由):")
    bio_question = "为什么我们身体里的每个细胞都包含DNA？"
    bio_result = router_demo.run_router_chain(bio_question)
    print(f"问题: {bio_question}")
    print(f"回答: {bio_result}")


if __name__ == "__main__":
    main()