from helper.helper import get_completion, get_completion_from_messages
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM

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

# 自定义聊天模型实现
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

class TextStyleTransformer:
    """
    将文本从一种风格转换为另一种风格
    """
    def __init__(self):
        pass
        
    def transform_text_direct(self, text, style):
        """使用直接OpenAI API调用转换文本风格"""
        prompt = f"""把由三个反引号分隔的文本翻译成一种{style}风格。
        文本: ```{text}```
        """
        return get_completion(prompt)
    
    def transform_text_langchain(self, text, style, temperature=0.0):
        """使用LangChain转换文本风格"""
        chat = TongyiQianwenChat(temperature=temperature)
        
        template_string = """把由三个反引号分隔的文本\
        翻译成一种{style}风格。\
        文本: ```{text}```
        """
        
        prompt_template = ChatPromptTemplate.from_template(template_string)
        messages = prompt_template.format_messages(style=style, text=text)
        response = chat(messages)
        return response.content

class ReviewAnalyzer:
    """
    分析客户评论并提取结构化信息
    """
    def __init__(self):
        self.chat = TongyiQianwenChat(temperature=0.0)
    
    def extract_info_simple(self, review_text):
        """不使用输出解析器提取信息"""
        review_template = """\
        对于以下文本，请从中提取以下信息：

        礼物：该商品是作为礼物送给别人的吗？ \
        如果是，则回答 是的；如果否或未知，则回答 不是。

        交货天数：产品需要多少天\
        到达？ 如果没有找到该信息，则输出-1。

        价钱：提取有关价值或价格的任何句子，\
        并将它们输出为逗号分隔的 Python 列表。

        使用以下键将输出格式化为 JSON：
        礼物
        交货天数
        价钱

        文本: {text}
        """
        
        prompt_template = ChatPromptTemplate.from_template(review_template)
        messages = prompt_template.format_messages(text=review_text)
        response = self.chat(messages)
        return response.content
    
    def extract_info_structured(self, review_text):
        """使用输出解析器提取结构化信息"""
        # 定义响应模式
        gift_schema = ResponseSchema(
            name="礼物",
            description="这件物品是作为礼物送给别人的吗？\
            如果是，则回答 是的，\
            如果否或未知，则回答 不是。"
        )
        
        delivery_days_schema = ResponseSchema(
            name="交货天数",
            description="产品需要多少天才能到达？\
            如果没有找到该信息，则输出-1。"
        )
        
        price_value_schema = ResponseSchema(
            name="价钱",
            description="提取有关价值或价格的任何句子，\
            并将它们输出为逗号分隔的 Python 列表"
        )
        
        response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        # 创建提示模板
        review_template = """\
        对于以下文本，请从中提取以下信息：

        礼物：该商品是作为礼物送给别人的吗？
        如果是，则回答 是的；如果否或未知，则回答 不是。

        交货天数：产品到达需要多少天？ 如果没有找到该信息，则输出-1。

        价钱：提取有关价值或价格的任何句子，并将它们输出为逗号分隔的 Python 列表。

        文本: {text}

        {format_instructions}
        """
        
        prompt_template = ChatPromptTemplate.from_template(template=review_template)
        messages = prompt_template.format_messages(
            text=review_text, 
            format_instructions=format_instructions
        )
        
        response = self.chat(messages)
        
        # 添加异常处理，确保输出解析错误时程序不会崩溃
        try:
            return output_parser.parse(response.content)
        except Exception as e:
            print(f"解析输出时出错: {e}")
            print(f"原始响应: {response.content}")
            # 返回一个简单的字典作为回退
            return {
                "礼物": "解析错误",
                "交货天数": "解析错误",
                "价钱": "解析错误",
                "原始响应": response.content
            }

class SimpleCalculator:
    """
    使用LLM进行简单计算
    """
    def __init__(self):
        pass
        
    def calculate(self, expression):
        """计算简单数学表达式"""
        return get_completion(expression)

def main():
    # 示例1：直接使用OpenAI - 简单计算
    print("===== 示例1: 简单计算 =====")
    calculator = SimpleCalculator()
    result = calculator.calculate("1+1是什么？")
    print(f"计算结果: {result}")
    print()
    
    # 示例2：文本风格转换 - 将海盗风格邮件转为正式普通话
    print("===== 示例2: 文本风格转换 - 海盗邮件转正式普通话 =====")
    
    customer_email = """
    嗯呐，我现在可是火冒三丈，我那个搅拌机盖子竟然飞了出去，把我厨房的墙壁都溅上了果汁！
    更糟糕的是，保修条款可不包括清理我厨房的费用。
    伙计，赶紧给我过来！
    """
    
    customer_style = """正式普通话 \
    用一个平静、尊敬、有礼貌的语调
    """
    
    transformer = TextStyleTransformer()
    
    # 使用直接API调用
    print("直接API调用结果:")
    direct_result = transformer.transform_text_direct(customer_email, customer_style)
    print(direct_result)
    print()
    
    # 使用LangChain
    print("LangChain调用结果:")
    langchain_result = transformer.transform_text_langchain(customer_email, customer_style)
    print(langchain_result)
    print()
    
    # 示例3：将客服回复转为海盗风格
    print("===== 示例3: 文本风格转换 - 客服回复转海盗风格 =====")
    
    service_reply = """嘿，顾客， \
    保修不包括厨房的清洁费用， \
    因为您在启动搅拌机之前 \
    忘记盖上盖子而误用搅拌机, \
    这是您的错。 \
    倒霉！ 再见！
    """
    
    service_style_pirate = """\
    一个有礼貌的语气 \
    使用海盗风格\
    """
    
    pirate_result = transformer.transform_text_langchain(service_reply, service_style_pirate)
    print(pirate_result)
    print()
    
    # 示例4：客户评论分析
    print("===== 示例4: 客户评论分析 =====")
    
    customer_review = """\
    这款吹叶机非常神奇。 它有四个设置：\
    吹蜡烛、微风、风城、龙卷风。 \
    两天后就到了，正好赶上我妻子的\
    周年纪念礼物。 \
    我想我的妻子会喜欢它到说不出话来。 \
    到目前为止，我是唯一一个使用它的人，而且我一直\
    每隔一天早上用它来清理草坪上的叶子。 \
    它比其他吹叶机稍微贵一点，\
    但我认为它的额外功能是值得的。
    """
    
    analyzer = ReviewAnalyzer()
    
    print("简单提取结果:")
    simple_result = analyzer.extract_info_simple(customer_review)
    print(simple_result)
    print()
    
    print("结构化提取结果:")
    structured_result = analyzer.extract_info_structured(customer_review)
    print(structured_result)
    print(f"礼物: {structured_result.get('礼物')}")
    print(f"交货天数: {structured_result.get('交货天数')}")
    print(f"价钱: {structured_result.get('价钱')}")

if __name__ == "__main__":
    main()