from helper.helper import get_completion, get_completion_from_messages
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
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


class TongyiQianwenChat:
    """
    通义千问聊天模型的简单封装
    """
    def __init__(self, temperature=0.0, model="qwen-turbo"):
        self.temperature = temperature
        self.model = model
    
    def __call__(self, messages):
        # 转换为helper能理解的格式
        formatted_messages = []
        for msg in messages:
            role = "user"
            if hasattr(msg, "type"):
                if msg.type == "human":
                    role = "user"
                elif msg.type == "ai":
                    role = "assistant"
                elif msg.type == "system":
                    role = "system"
            content = msg.content
            formatted_messages.append({"role": role, "content": content})
        
        # 调用get_completion_from_messages
        content = get_completion_from_messages(
            formatted_messages, 
            model=self.model,
            temperature=self.temperature
        )
        
        # 创建一个简单的响应对象
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(content)


class LLMChainDemo:
    """大语言模型链演示"""

    def __init__(self, temperature=0.0):
        self.llm = TongyiQianwenLLM(temperature=temperature)
    
    def company_name_generator(self, product):
        """生成公司名称的LLM链演示"""
        prompt = ChatPromptTemplate.from_template(
            "描述制造{product}的一个公司的最佳名称是什么?"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(product)


class SimpleSequentialChainDemo:
    """简单顺序链演示"""

    def __init__(self, temperature=0.9, verbose=True):
        self.llm = TongyiQianwenLLM(temperature=temperature)
        self.verbose = verbose
        
    def company_description_generator(self, product):
        """根据产品生成公司名称和描述"""
        # 第一个链：生成公司名称
        first_prompt = ChatPromptTemplate.from_template(
            "描述制造{product}的一个公司的最好的名称是什么"
        )
        chain_one = LLMChain(llm=self.llm, prompt=first_prompt)
        
        # 第二个链：根据公司名称生成描述
        second_prompt = ChatPromptTemplate.from_template(
            "写一个20字的描述对于下面这个公司：{company_name}的"
        )
        chain_two = LLMChain(llm=self.llm, prompt=second_prompt)
        
        # 组合成简单顺序链
        overall_chain = SimpleSequentialChain(
            chains=[chain_one, chain_two],
            verbose=self.verbose
        )
        
        return overall_chain.run(product)


class SequentialChainDemo:
    """多输入输出的顺序链演示"""

    def __init__(self, temperature=0.9, verbose=True):
        self.llm = TongyiQianwenLLM(temperature=temperature)
        self.verbose = verbose
        
    def process_review(self, review):
        """处理评论：翻译、总结、分析语言并生成回复"""
        # 链1：翻译评论
        first_prompt = ChatPromptTemplate.from_template(
            "把下面的评论review翻译成英文:\n\n{Review}"
        )
        chain_one = LLMChain(
            llm=self.llm, 
            prompt=first_prompt, 
            output_key="English_Review"
        )
        
        # 链2：总结评论
        second_prompt = ChatPromptTemplate.from_template(
            "请你用一句话来总结下面的评论review:\n\n{English_Review}"
        )
        chain_two = LLMChain(
            llm=self.llm, 
            prompt=second_prompt, 
            output_key="summary"
        )
        
        # 链3：识别语言
        third_prompt = ChatPromptTemplate.from_template(
            "下面的评论review使用的什么语言:\n\n{Review}"
        )
        chain_three = LLMChain(
            llm=self.llm, 
            prompt=third_prompt, 
            output_key="language"
        )
        
        # 链4：生成回复
        fourth_prompt = ChatPromptTemplate.from_template(
            "使用特定的语言对下面的总结写一个后续回复:"
            "\n\n总结: {summary}\n\n语言: {language}"
        )
        chain_four = LLMChain(
            llm=self.llm, 
            prompt=fourth_prompt, 
            output_key="followup_message"
        )
        
        # 组合成顺序链
        overall_chain = SequentialChain(
            chains=[chain_one, chain_two, chain_three, chain_four],
            input_variables=["Review"],
            output_variables=["English_Review", "summary", "followup_message"],
            verbose=self.verbose
        )
        
        return overall_chain({"Review": review})


class MultiPromptChainDemo:
    """路由链演示"""

    def __init__(self, temperature=0.0, verbose=True):
        self.llm = TongyiQianwenLLM(temperature=temperature)
        self.verbose = verbose
        
    def setup_prompt_infos(self):
        """设置不同领域的提示模板信息"""
        # 物理学提示模板
        physics_template = """你是一个非常聪明的物理专家。 \
        你擅长用一种简洁并且易于理解的方式去回答问题。\
        当你不知道问题的答案时，你承认\
        你不知道.

        这是一个问题:
        {input}"""
        
        # 数学提示模板
        math_template = """你是一个非常优秀的数学家。 \
        你擅长回答数学问题。 \
        你之所以如此优秀， \
        是因为你能够将棘手的问题分解为组成部分，\
        回答组成部分，然后将它们组合在一起，回答更广泛的问题。

        这是一个问题：
        {input}"""
        
        # 历史提示模板
        history_template = """你是以为非常优秀的历史学家。 \
        你对一系列历史时期的人物、事件和背景有着极好的学识和理解\
        你有能力思考、反思、辩证、讨论和评估过去。\
        你尊重历史证据，并有能力利用它来支持你的解释和判断。

        这是一个问题:
        {input}"""
        
        # 计算机科学提示模板
        computerscience_template = """你是一个成功的计算机科学专家。\
        你有创造力、协作精神、\
        前瞻性思维、自信、解决问题的能力、\
        对理论和算法的理解以及出色的沟通技巧。\
        你非常擅长回答编程问题。\
        你之所以如此优秀，是因为你知道\
        如何通过以机器可以轻松解释的命令式步骤描述解决方案来解决问题，\
        并且你知道如何选择在时间复杂性和空间复杂性之间取得良好平衡的解决方案。

        这还是一个输入：
        {input}"""
        
        # 生物学提示模板
        biology_template = """你是一位专业的生物学家。 \
        你对分子生物学、细胞生物学、生态学和进化论有深入的了解。\
        你能够用清晰易懂的方式解释复杂的生物学概念。\
        当有人询问生物学问题时，你会提供最新、最准确的科学信息。

        这是一个问题:
        {input}"""
        
        # 返回提示信息列表
        return [
            {
                "名字": "物理学", 
                "描述": "擅长回答关于物理学的问题", 
                "提示模板": physics_template
            },
            {
                "名字": "数学", 
                "描述": "擅长回答数学问题", 
                "提示模板": math_template
            },
            {
                "名字": "历史", 
                "描述": "擅长回答历史问题", 
                "提示模板": history_template
            },
            {
                "名字": "计算机科学", 
                "描述": "擅长回答计算机科学问题", 
                "提示模板": computerscience_template
            },
            {
                "名字": "生物学", 
                "描述": "擅长回答生物学问题", 
                "提示模板": biology_template
            }
        ]

    def create_chain(self):
        """创建路由链"""
        # 获取提示信息
        prompt_infos = self.setup_prompt_infos()
        
        # 创建目标链
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["名字"]
            prompt_template = p_info["提示模板"]
            prompt = ChatPromptTemplate.from_template(template=prompt_template)
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain  
        
        # 创建默认链
        default_prompt = ChatPromptTemplate.from_template("{input}")
        default_chain = LLMChain(llm=self.llm, prompt=default_prompt)
        
        # 准备目标信息字符串
        destinations = [f"{p['名字']}: {p['描述']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        
        # 设置路由模板
        router_template = """给语言模型一个原始文本输入，\
        让其选择最适合输入的模型提示。\
        系统将为您提供可用提示的名称以及最适合改提示的描述。\
        如果你认为修改原始输入最终会导致语言模型做出更好的响应，\
        你也可以修改原始输入。

        << 格式 >>
        返回一个带有JSON对象的markdown代码片段，该JSON对象的格式如下：
        ```json
        {{{{
            "destination": 字符串 \\ 使用的提示名字或者使用 "DEFAULT"
            "next_inputs": 字符串 \\ 原始输入的改进版本
        }}}}

        记住："destination"必须是下面指定的候选提示名称之一，\
        或者如果输入不太适合任何候选提示，\
        则可以是 "DEFAULT" 。
        记住：如果您认为不需要任何修改，\
        则 "next_inputs" 可以只是原始输入。

        << 候选提示 >>
        {destinations}

        << 输入 >>
        {{input}}

        << 输出 (记得要包含 ```json)>>

        样例:
        << 输入 >>
        "什么是黑体辐射?"
        << 输出 >>
        ```json
        {{{{
            "destination": "物理学",
            "next_inputs": "什么是黑体辐射，请用简单易懂的方式解释?"
        }}}}
        """.format(destinations=destinations_str)
        
        # 创建路由提示
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        
        # 创建路由链
        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)
        
        # 创建多提示链
        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=self.verbose
        )
        
        return chain
        
    def route_query(self, query):
        """根据查询路由到适当的链并获取回答"""
        chain = self.create_chain()
        return chain.run(query)


def demo_llm_chain():
    """演示基本的LLM链功能"""
    print("\n===== 大语言模型链演示 =====")
    
    # 初始化LLM链演示
    llm_chain_demo = LLMChainDemo(temperature=0.0)
    
    # 示例：生成公司名称
    product = "大号床单套装"
    company_name = llm_chain_demo.company_name_generator(product)
    print(f"产品: {product}")
    print(f"生成的公司名称: {company_name}")


def demo_simple_sequential_chain():
    """演示简单顺序链功能"""
    print("\n===== 简单顺序链演示 =====")
    
    # 初始化简单顺序链演示
    simple_seq_chain_demo = SimpleSequentialChainDemo(temperature=0.9, verbose=True)
    
    # 示例：生成公司名称和描述
    product = "大号床单套装"
    result = simple_seq_chain_demo.company_description_generator(product)
    print(f"产品: {product}")
    print(f"最终结果: {result}")


def demo_sequential_chain():
    """演示顺序链功能"""
    print("\n===== 顺序链演示 =====")
    
    # 初始化顺序链演示
    seq_chain_demo = SequentialChainDemo(temperature=0.9, verbose=True)
    
    # 示例：处理评论
    sample_review = "Je trouve le goût médiocre. La mousse ne tient pas, c'est bizarre. J'achète les mêmes dans le commerce et le goût est bien meilleur...\nVieux lot ou contrefaçon !?"
    print(f"原始评论: {sample_review}")
    
    result = seq_chain_demo.process_review(sample_review)
    
    print("\n处理结果:")
    print(f"英文翻译: {result['English_Review']}")
    print(f"评论总结: {result['summary']}")
    print(f"后续回复: {result['followup_message']}")


def demo_multi_prompt_chain():
    """演示路由链功能"""
    print("\n===== 路由链演示 =====")
    
    # 初始化路由链演示
    multi_prompt_demo = MultiPromptChainDemo(temperature=0.0, verbose=True)
    
    # 示例1：物理学问题
    physics_query = "什么是黑体辐射？"
    print(f"\n物理学问题: {physics_query}")
    physics_result = multi_prompt_demo.route_query(physics_query)
    print(f"回答: {physics_result}")
    
    # 示例2：数学问题
    math_query = "2+2等于多少？"
    print(f"\n数学问题: {math_query}")
    math_result = multi_prompt_demo.route_query(math_query)
    print(f"回答: {math_result}")
    
    # 示例3：生物学问题
    biology_query = "为什么我们身体里的每个细胞都包含DNA？"
    print(f"\n生物学问题: {biology_query}")
    biology_result = multi_prompt_demo.route_query(biology_query)
    print(f"回答: {biology_result}")


def main():
    """主函数，运行各种链的演示"""
    # 演示大语言模型链
    demo_llm_chain()
    
    # 演示简单顺序链
    demo_simple_sequential_chain()
    
    # 演示顺序链
    demo_sequential_chain()
    
    # 演示路由链
    demo_multi_prompt_chain()


if __name__ == "__main__":
    main()