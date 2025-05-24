from helper.helper import get_completion
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import StructuredTool
from langchain.tools.python.tool import PythonREPLTool
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from datetime import date

class TongyiQianwenLLM(LLM):
    """
    通义千问自定义LLM实现，用于LangChain代理
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

class BuiltinToolsAgent:
    """
    使用LangChain内置工具的代理类
    支持数学计算和维基百科搜索
    """
    def __init__(self, temperature=0.0):
        self.llm = TongyiQianwenLLM(temperature=temperature)
        self.tools = load_tools(["llm-math", "wikipedia"], llm=self.llm)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )
    
    def run(self, query):
        """执行查询"""
        return self.agent(query)
    
    def calculate(self, expression):
        """执行数学计算"""
        return self.run(f"计算{expression}")
    
    def search_wikipedia(self, query):
        """搜索维基百科"""
        return self.run(query)

class PythonCodeAgent:
    """
    使用Python代码执行的代理类
    可以执行Python代码来解决问题
    """
    def __init__(self, temperature=0.0):
        self.llm = TongyiQianwenLLM(temperature=temperature)
        self.agent = create_python_agent(
            self.llm,
            tool=PythonREPLTool(),
            verbose=True
        )
    
    def run(self, query):
        """执行查询"""
        try:
            return self.agent.run(query)
        except Exception as e:
            error_msg = str(e)
            # 检查是否是输出解析异常
            if "OutputParserException" in error_msg:
                # 尝试提取最终答案 - 处理有无冒号的两种情况
                if "Final Answer:" in error_msg:
                    final_answer = error_msg.split("Final Answer:")[-1].strip()
                    return final_answer
                elif "Final Answer" in error_msg:
                    # 尝试提取没有冒号的最终答案
                    parts = error_msg.split("Final Answer")
                    if len(parts) > 1:
                        final_answer = parts[-1].strip()
                        return final_answer
            
            # 如果无法提取最终答案，尝试提取观察结果
            if "Observation:" in error_msg:
                # 查找最后一个Observation之后的内容
                obs_parts = error_msg.split("Observation:")
                last_obs = obs_parts[-1].strip()
                if last_obs:
                    return last_obs
                    
            # 处理同时包含动作和最终答案的情况
            if "Action:" in error_msg and "Action Input:" in error_msg:
                # 查找错误信息中最后出现的结果
                if "[" in error_msg and "]" in error_msg:
                    # 尝试提取结果数据结构
                    import re
                    matches = re.findall(r'\[.*?\]', error_msg)
                    if matches:
                        return matches[-1]  # 返回最后匹配的结果
        
        # 如果是其他异常或无法提取结果，尝试从执行结果中提取
        if "Observation:" in error_msg and "execute_result" in error_msg:
            # 尝试提取执行结果
            result_parts = error_msg.split("execute_result:")
            if len(result_parts) > 1:
                # 提取结果，通常在代码块之间
                result_text = result_parts[1]
                if "```" in result_text:
                    result = result_text.split("```")[1].strip()
                    return result
                
        # 如果仍然无法提取结果，返回一个友好的错误消息
        print(f"解析异常: {error_msg[:200]}...")  # 仅打印部分错误信息以便调试
        return "处理查询时出现错误，无法提取有效结果。请尝试重新表述您的请求。"
    
    def convert_names_to_pinyin(self, names_list):
        """将中文名字转换为拼音"""
        query = f"使用pinyin拼音库将这些客户名字转换为拼音，并打印输出列表: {names_list}"
        return self.run(query)
    
    def execute_python_task(self, task_description):
        """执行Python编程任务"""
        return self.run(task_description)

class CustomToolAgent:
    """
    使用自定义工具的代理类
    """
    def __init__(self, temperature=0.0):
        self.llm = TongyiQianwenLLM(temperature=temperature)
        
        # Create a tool from the function
        def get_time(input_str: str = "") -> str:
            """
            返回今天的日期，用于任何需要知道今天日期的问题。
            输入应该总是一个空字符串。
            """
            today = date.today()
            return f"今天的日期是：{today.strftime('%Y年%m月%d日')}"
        
        # Use StructuredTool and appropriate agent type
        self.tools = [StructuredTool.from_function(get_time)]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )
    
    def run(self, query):
        """执行查询"""
        return self.agent(query)
        
    def get_current_date(self):
        """获取当前日期"""
        return self.run("今天是几号？")

class WeatherToolAgent:
    """
    自定义天气工具代理示例
    """
    def __init__(self, temperature=0.0):
        self.llm = TongyiQianwenLLM(temperature=temperature)
        
        # Define the weather function directly
        def get_weather(city: str) -> str:
            """
            获取指定城市的天气信息。
            输入应该是城市名称，比如"北京"、"上海"等。
            这是一个模拟工具，实际应用中需要连接真实的天气API。
            """
            # 模拟天气数据
            weather_data = {
                "北京": "晴天，温度25°C",
                "上海": "多云，温度22°C", 
                "广州": "雨天，温度28°C",
                "深圳": "晴天，温度30°C"
            }
            return weather_data.get(city, f"抱歉，暂时无法获取{city}的天气信息")
        
        # Create a tool using StructuredTool
        self.tools = [StructuredTool.from_function(get_weather)]
        
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )
    
    def run(self, query):
        """执行查询"""
        return self.agent(query)
    
    def get_weather(self, city):
        """获取指定城市天气"""
        return self.run(f"{city}的天气怎么样？")

def main():
    print("===== LangChain代理演示程序 =====\n")
    
    # 示例1：使用内置工具进行数学计算和维基百科搜索
    print("1. 内置工具代理演示")
    print("-" * 50)
    
    builtin_agent = BuiltinToolsAgent()
    
    print("数学计算示例:")
    math_result = builtin_agent.calculate("300的25%")
    print(f"结果: {math_result}")
    print()
    
    print("维基百科搜索示例:")
    wiki_query = "Tom M. Mitchell是一位美国计算机科学家，也是卡内基梅隆大学（CMU）的创始人大学教授。他写了哪本书呢？"
    wiki_result = builtin_agent.search_wikipedia(wiki_query)
    print(f"结果: {wiki_result}")
    print()
    
    # 示例2：使用Python代码代理
    print("2. Python代码代理演示")
    print("-" * 50)
    
    python_agent = PythonCodeAgent()
    
    customer_list = ["小明", "小黄", "小红", "小蓝", "小橘", "小绿"]
    print(f"转换名字列表: {customer_list}")
    pinyin_result = python_agent.convert_names_to_pinyin(customer_list)
    print(f"拼音转换结果: {pinyin_result}")
    print()
    
    # 示例3：使用自定义时间工具
    print("3. 自定义工具代理演示")
    print("-" * 50)
    
    custom_agent = CustomToolAgent()
    
    print("获取当前日期:")
    date_result = custom_agent.get_current_date()
    print(f"结果: {date_result}")
    print()
    
    # 示例4：自定义天气工具演示
    print("4. 自定义天气工具演示")
    print("-" * 50)
    
    weather_agent = WeatherToolAgent()
    
    print("查询北京天气:")
    weather_result = weather_agent.get_weather("北京")
    print(f"结果: {weather_result}")
    print()
    
    # 示例5：Python代理执行其他编程任务
    print("5. Python代理编程任务演示")
    print("-" * 50)
    
    programming_task = """
    创建一个包含1到10的列表，然后计算所有偶数的平方和，并打印结果。
    """
    print(f"编程任务: {programming_task}")
    programming_result = python_agent.execute_python_task(programming_task)
    print(f"执行结果: {programming_result}")

if __name__ == "__main__":
    main()