from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 从helper模块导入工具函数
from helper.helper import create_client

# 定义大语言模型
def create_llm(temperature=0.0):
    """创建自定义语言模型接口，避开ChatOpenAI兼容性问题"""
    client = create_client()

    def invoke(messages):
        # 转换LangChain消息格式到通义千问支持的格式
        api_messages = []
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg
                # 将LangChain的角色名映射到通义千问支持的角色
                if role == "human":
                    role = "user"
                api_messages.append({"role": role, "content": content})
            else:
                # 将LangChain的消息类型映射到通义千问支持的角色
                role = msg.type
                if role == "human":
                    role = "user"
                api_messages.append({"role": role, "content": msg.content})

        api_response = client.chat.completions.create(
            model="qwen-turbo",
            messages=api_messages,
            temperature=temperature,
        )
        return api_response.choices[0].message.content

    return invoke

# 示例1: 使用提示模板转换邮件风格
def translate_email_style(email_text, target_style, temperature=0.0):
    """将邮件翻译成指定风格"""
    # 创建提示模板
    template_string = """把由三个反引号分隔的文本翻译成一种{style}风格。
    文本: ```{text}```
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)

    # 格式化消息
    messages = prompt_template.format_messages(
        style=target_style,
        text=email_text
    )

    # 调用模型获取响应
    llm = create_llm(temperature)
    response = llm(messages)

    return response

# 示例2: 使用Pydantic解析器提取评价中的信息
class ReviewInfo(BaseModel):
    """定义评价信息的模型"""
    gift: bool = Field(description="该商品是作为礼物送给别人的吗?")
    delivery_days: int = Field(description="产品需要多少天才能到达? 如果没有该信息, 则返回-1")
    price_value: str = Field(description="提取有关价值或价格的描述")

def extract_review_info(review_text, temperature=0.0):
    """从评价中提取结构化信息"""
    # 创建Pydantic输出解析器
    parser = PydanticOutputParser(pydantic_object=ReviewInfo)

    # 创建提示模板
    review_template = """
    对于以下文本，请提取以下信息：

    礼物：该商品是作为礼物送给别人的吗？ 
    交货天数：产品需要多少天才能到达？如果没有找到该信息，则输出-1。
    价格：提取有关价值或价格的任何描述。

    文本: {text}
    
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=review_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 格式化消息
    messages = prompt.format_messages(text=review_text)

    # 调用模型并解析响应
    llm = create_llm(temperature)
    response = llm(messages)

    return parser.parse(response)

def main():
    # 示例1: 邮件风格转换
    print("=== 邮件风格转换示例 ===")

    # 原始海盗风格邮件
    customer_email = """
    嗯呐，我现在可是火冒三丈，我那个搅拌机盖子竟然飞了出去，把我厨房的墙壁都溅上了果汁！
    更糟糕的是，保修条款可不包括清理我厨房的费用。
    伙计，赶紧给我过来！
    """

    # 转换成正式普通话风格
    formal_style = "正式普通话，用一个平静、尊敬、有礼貌的语调"
    formal_response = translate_email_style(customer_email, formal_style)
    print("\n原始邮件:")
    print(customer_email)
    print("\n转换为正式普通话:")
    print(formal_response)

    # 客服回复转换为海盗风格
    service_reply = """
    嘿，顾客，
    保修不包括厨房的清洁费用，
    因为您在启动搅拌机之前
    忘记盖上盖子而误用搅拌机,
    这是您的错。
    倒霉！再见！
    """

    pirate_style = "一个有礼貌的语气，使用海盗风格"
    pirate_response = translate_email_style(service_reply, pirate_style)
    print("\n客服原始回复:")
    print(service_reply)
    print("\n转换为海盗风格:")
    print(pirate_response)

    # 示例2: 提取评价信息
    print("\n\n=== 评价信息提取示例 ===")

    customer_review = """
    这款吹叶机非常神奇。它有四个设置：
    吹蜡烛、微风、风城、龙卷风。
    两天后就到了，正好赶上我妻子的
    周年纪念礼物。
    我想我的妻子会喜欢它到说不出话来。
    到目前为止，我是唯一一个使用它的人，而且我一直
    每隔一天早上用它来清理草坪上的叶子。
    它比其他吹叶机稍微贵一点，
    但我认为它的额外功能是值得的。
    """

    try:
        review_info = extract_review_info(customer_review)
        print("\n评价文本:")
        print(customer_review)
        print("\n提取的结构化信息:")
        print(f"是否是礼物: {review_info.gift}")
        print(f"交货天数: {review_info.delivery_days}")
        print(f"价格评价: {review_info.price_value}")
    except Exception as e:
        print(f"提取评价信息时出错: {e}")
        print("有可能是模型输出格式不正确导致解析失败")

if __name__ == "__main__":
    load_dotenv(find_dotenv())  # 加载环境变量
    main()