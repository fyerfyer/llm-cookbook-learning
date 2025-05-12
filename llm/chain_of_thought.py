from helper.helper import get_completion_from_messages

class ChainOfThoughtReasoner:
    def __init__(self, delimiter="===="):
        self.delimiter = delimiter
        self.system_message = self._create_system_message()

    def _create_system_message(self):
         return f"""
请按照以下步骤回答客户的提问。客户的提问将以{self.delimiter}分隔。

步骤 1:{self.delimiter}首先确定用户是否正在询问有关特定产品或产品的问题。产品类别不计入范围。

步骤 2:{self.delimiter}如果用户询问特定产品，请确认产品是否在以下列表中。所有可用产品：

产品：TechPro 超极本
类别：计算机和笔记本电脑
品牌：TechPro
型号：TP-UB100
保修期：1 年
评分：4.5
特点：13.3 英寸显示屏，8GB RAM，256GB SSD，Intel Core i5 处理器
描述：一款适用于日常使用的时尚轻便的超极本。
价格：$799.99

产品：BlueWave 游戏笔记本电脑
类别：计算机和笔记本电脑
品牌：BlueWave
型号：BW-GL200
保修期：2 年
评分：4.7
特点：15.6 英寸显示屏，16GB RAM，512GB SSD，NVIDIA GeForce RTX 3060
描述：一款高性能的游戏笔记本电脑，提供沉浸式体验。
价格：$1199.

产品：PowerLite 可转换笔记本电脑
类别：计算机和笔记本电脑
品牌：PowerLite
型号：PL-CV300
保修期：1年
评分：4.3
特点：14 英寸触摸屏，8GB RAM，256GB SSD，360 度铰链
描述：一款多功能可转换笔记本电脑，具有响应触摸屏。
价格：$699.99

产品：TechPro 台式电脑
类别：计算机和笔记本电脑
品牌：TechPro
型号：TP-DT500
保修期：1年
评分：4.4
特点：Intel Core i7 处理器，16GB RAM，1TB HDD，NVIDIA GeForce GTX 1660
描述：一款功能强大的台式电脑，适用于工作和娱乐。
价格：$999.99

产品：BlueWave Chromebook
类别：计算机和笔记本电脑
品牌：BlueWave
型号：BW-CB100
保修期：1 年
评分：4.1
特点：11.6 英寸显示屏，4GB RAM，32GB eMMC，Chrome OS
描述：一款紧凑而价格实惠的 Chromebook，适用于日常任务。
价格：$249.99

步骤 3:{self.delimiter} 如果消息中包含上述列表中的产品，请列出用户在消息中做出的任何假设，\
例如笔记本电脑 X 比笔记本电脑 Y 大，或者笔记本电脑 Z 有 2 年保修期。

步骤 4:{self.delimiter} 如果用户做出了任何假设，请根据产品信息确定假设是否正确。

步骤 5:{self.delimiter} 如果用户有任何错误的假设，请先礼貌地纠正客户的错误假设（如果适用）。\
只提及或引用可用产品列表中的产品，因为这是商店销售的唯一五款产品。以友好的口吻回答客户。

使用以下格式回答问题：
步骤 1: {self.delimiter} <步骤 1 的推理>
步骤 2: {self.delimiter} <步骤 2 的推理>
步骤 3: {self.delimiter} <步骤 3 的推理>
步骤 4: {self.delimiter} <步骤 4 的推理>
回复客户: {self.delimiter} <回复客户的内容>

请确保每个步骤上面的回答中中使用 {self.delimiter} 对步骤和步骤的推理进行分隔。
"""

    def process_query(self, user_message, temperature=0, show_reasoning=False):
        """
        处理用户查询，可以选择是否展示推理过程
        
        Args:
            user_message (str): 用户的查询文本
            temperature (float): 模型温度参数，控制随机性
            show_reasoning (bool): 是否显示推理过程
            
        Returns:
            str: 回答结果，根据show_reasoning决定是否包含推理过程
        """
        message = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"{self.delimiter}{user_message}{self.delimiter}"}
        ]

        response = get_completion_from_messages(message, temperature=temperature)
        if show_reasoning:
            return response
        else:
        # 仅返回最终回复，不包含推理过程
            try:
                if self.delimiter in response:
                    final_response = response.split(self.delimiter)[-1].strip()
                else:
                    final_response = response.split("回复客户:")[-1].strip()
                return final_response
            except Exception as e:
                return f"处理响应时出错: {e}\n原始响应: {response}"

class InternalMonologue:
    """
    内心独白类，用于执行任务但隐藏中间推理过程
    """
    def __init__(self, task_description=""):
        self.task_description = task_description
        self.system_message = self._create_system_message()

    def _create_system_message(self):
        return f"""
{self.task_description}

在回答问题时，请使用以下格式：

内心思考：
{{
    在这里进行你的详细分析和推理。
    考虑所有相关因素。
    逐步思考问题。
    这部分不会展示给用户。
}}

回答：
在这里只提供最终结论或回答，不要透露你的详细推理过程。
保持简洁明了。
"""

    def process_query(self, user_message, temperature=0, show_internal=False):
        """
        处理用户查询，隐藏内心独白部分
        
        Args:
            user_message (str): 用户的查询文本
            temperature (float): 模型温度参数
            show_internal (bool): 是否展示内心独白部分
            
        Returns:
            str: 响应结果
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message}
        ]

        response = get_completion_from_messages(messages, temperature=temperature)
        
        if show_internal:
            return response
        else:
            # 仅返回"回答："部分的内容
            try:
                if "回答：" in response:
                    final_response = response.split("回答：")[-1].strip()
                    return final_response
                else:
                    # 如果没有找到"回答："，返回整个响应
                    return response
            except Exception as e:
                return f"处理响应时出错: {e}\n原始响应: {response}"

def main():
    print("思维链推理演示:")
    reasoner = ChainOfThoughtReasoner()
    
    # 测试例子1 - 价格比较查询
    print("\n测试例子1: 产品价格比较")
    query1 = "BlueWave Chromebook 比 TechPro 台式电脑贵多少？"
    
    print("包含推理过程的回答:")
    result1_full = reasoner.process_query(query1, show_reasoning=True)
    print(result1_full)
    
    print("\n隐藏推理过程的回答:")
    result1_concise = reasoner.process_query(query1, show_reasoning=False)
    print(result1_concise)

    # 测试例子2 - 查询不存在的产品
    print("\n\n测试例子2: 查询不存在的产品")
    query2 = "你有电视机吗？"
    result2 = reasoner.process_query(query2, show_reasoning=False)
    print(result2)
    
    # 演示内心独白功能
    print("\n\n内心独白功能演示:")
    
    math_monologue = InternalMonologue(
        task_description="你是一位数学导师，帮助学生解决数学问题，但不直接给出答案，而是给予提示和指导。"
    )
    
    math_query = "如何求解方程：2x + 5 = 15"
    print("包含内心独白的完整回答:")
    full_response = math_monologue.process_query(math_query, show_internal=True)
    print(full_response)
    
    print("\n仅向用户展示的回答:")
    final_response = math_monologue.process_query(math_query, show_internal=False)
    print(final_response)


if __name__ == "__main__":
    main()