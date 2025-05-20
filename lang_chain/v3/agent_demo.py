from typing import Union, Sequence
from dotenv import load_dotenv, find_dotenv
from datetime import date
import time
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("agent_demo")

# Import langgraph components
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool, BaseTool, Tool
from langchain_core.language_models import LanguageModelLike
from pydantic import BaseModel

# Import helper functions for Tongyi Qianwen
from helper.helper import create_client


class QwenModel(LanguageModelLike):
    """通义千问模型的LangGraph兼容接口"""
    
    def __init__(self, temperature=0.0):
        """初始化通义千问模型"""
        logger.info("Initializing QwenModel with temperature=%f", temperature)
        self.client = create_client()
        self.temperature = temperature
        self.tools = None
    
    def invoke(self, messages, config=None, **kwargs):
        """实现langchain兼容的invoke方法"""
        logger.info("QwenModel.invoke called with %d messages", len(messages) if isinstance(messages, list) else 1)
        # 转换LangChain消息格式到通义千问支持的格式
        api_messages = self._convert_messages(messages)
        logger.info("Converted messages: %s", json.dumps(api_messages))
        
        # 准备API参数
        api_kwargs = {"temperature": self.temperature}
        
        # 处理配置参数
        if config:
            # 如果提供了配置，则将其合并到kwargs中
            if isinstance(config, dict):
                # 过滤掉不支持的参数
                supported_params = [
                    "temperature", "max_tokens", "top_p", "frequency_penalty", 
                    "presence_penalty", "stop", "stream", "seed"
                ]
                filtered_config = {k: v for k, v in config.items() if k in supported_params}
                api_kwargs.update(filtered_config)
                logger.info("Added config params: %s", json.dumps(filtered_config))
        
        # 合并其他kwargs，也过滤不支持的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ["tags", "callbacks", "run_name", "verbose"]}
        api_kwargs.update(filtered_kwargs)
        logger.info("Final API kwargs: %s", json.dumps({k: str(v) for k, v in api_kwargs.items() if k != "tools"}))
        
        # 如果有绑定工具，添加到API调用中
        if self.tools:
            api_kwargs["tools"] = self.tools
            api_kwargs["tool_choice"] = "auto"
            logger.info("Using %d bound tools", len(self.tools))
        
        # 调用API
        try:
            logger.info("Calling Qwen API...")
            api_response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=api_messages,
                **api_kwargs
            )
            logger.info("API response received")
            
            # 检查是否有工具调用
            content = api_response.choices[0].message.content
            logger.info("Response content: %s", content[:100] + "..." if len(content) > 100 else content)
            
            tool_calls = getattr(api_response.choices[0].message, "tool_calls", None)
            if tool_calls:
                logger.info("Tool calls received: %d", len(tool_calls))
            
            # 返回AI消息，包含工具调用信息
            response_message = AIMessage(content=content)
            
            # 处理工具调用
            if tool_calls:
                processed_tool_calls = []
                for tool_call in tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments)
                        logger.info("Tool call: %s - Args: %s", tool_call.function.name, json.dumps(args))
                    except:
                        args = tool_call.function.arguments
                        logger.info("Tool call (unparsed args): %s - Args: %s", tool_call.function.name, args)
                        
                    processed_tool_calls.append({
                        "name": tool_call.function.name,
                        "args": args,
                        "id": tool_call.id,
                        "type": "tool_call"
                    })
                
                response_message.additional_kwargs = {"tool_calls": processed_tool_calls}
                logger.info("Processed %d tool calls", len(processed_tool_calls))
            
            return response_message
        
        except Exception as e:
            logger.error(f"API call error: {e}")
            logger.error(f"Parameters: {api_kwargs}")
            # 返回一个错误消息
            return AIMessage(content=f"调用模型时出错: {str(e)}")
    
    def _convert_messages(self, messages):
        """转换消息格式"""
        api_messages = []
        
        # 处理单个消息或消息列表
        if not isinstance(messages, list):
            # 如果是单个BaseMessage对象
            if isinstance(messages, BaseMessage):
                role = messages.type
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                if role not in ["system", "assistant", "user", "tool", "function"]:
                    role = "user"
                return [{"role": role, "content": messages.content}]
            
            # 如果是特殊输入如 ChatPromptValue
            if hasattr(messages, 'to_messages'):
                messages = messages.to_messages()
            else:
                # 单个非消息对象，当作用户消息处理
                return [{"role": "user", "content": str(messages)}]
        
        # 处理消息列表
        for msg in messages:
            if isinstance(msg, BaseMessage):
                role = msg.type
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                if role not in ["system", "assistant", "user", "tool", "function"]:
                    role = "user"
                
                content = msg.content
                api_message = {"role": role, "content": content}
                
                # 处理工具消息
                if role == "tool" and hasattr(msg, "tool_call_id"):
                    api_message["tool_call_id"] = msg.tool_call_id
                    
                api_messages.append(api_message)
            elif isinstance(msg, dict):
                # 处理字典格式的消息
                role = msg.get("role", "user")
                content = msg.get("content", "")
                api_message = {"role": role, "content": content}
                
                # 处理工具消息
                if role == "tool" and "tool_call_id" in msg:
                    api_message["tool_call_id"] = msg["tool_call_id"]
                
                api_messages.append(api_message)
            elif isinstance(msg, tuple):
                # 处理元组格式的消息
                role, content = msg
                if role == "human":
                    role = "user"
                api_messages.append({"role": role, "content": content})
        
        return api_messages
        
    def predict(self, text, **kwargs):
        """实现预测文本方法"""
        logger.info("QwenModel.predict called with text: %s", text[:50] + "..." if len(text) > 50 else text)
        message = [{"role": "user", "content": text}]
        response = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=message,
            temperature=self.temperature,
            **kwargs
        )
        logger.info("Predict response received")
        return response.choices[0].message.content
    
    def bind_tools(self, tools: Sequence[Union[BaseTool, Tool, BaseModel, type]]) -> "QwenModel":
        """绑定工具到模型，实现LangChain工具调用兼容"""
        logger.info("QwenModel.bind_tools called with %d tools", len(tools))
        # Create a new instance instead of deep copying
        new_model = QwenModel(temperature=self.temperature)
        # Share the client reference instead of copying it
        new_model.client = self.client
        
        # 将各种工具格式转换为通义千问API兼容的格式
        api_tools = []
        
        for tool in tools:
            if isinstance(tool, BaseTool) or (hasattr(tool, "name") and hasattr(tool, "description")):
                # 处理LangChain工具对象
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
                logger.info("Processing BaseTool: %s", tool.name)
                
                # 如果工具有args_schema
                if hasattr(tool, "args_schema") and tool.args_schema:
                    # 使用model_json_schema替代schema
                    try:
                        schema = tool.args_schema.model_json_schema()
                        logger.info("Using model_json_schema for tool %s", tool.name)
                    except AttributeError:
                        # 向后兼容: 如果model_json_schema不可用，尝试使用schema
                        schema = tool.args_schema.schema()
                        logger.info("Falling back to schema for tool %s", tool.name)
                        
                    tool_schema["function"]["parameters"] = {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }
                
                api_tools.append(tool_schema)
                
            elif callable(tool) and hasattr(tool, "__annotations__"):
                # 处理普通Python函数
                import inspect
                
                name = tool.__name__
                docstring = tool.__doc__ or ""
                signature = inspect.signature(tool)
                logger.info("Processing function tool: %s", name)
                
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                
                for param_name, param in signature.parameters.items():
                    if param_name == "self":
                        continue
                        
                    # 获取类型注解
                    param_type = tool.__annotations__.get(param_name, str)
                    
                    # 简单转换类型到JSON schema类型
                    json_type = "string"
                    if param_type == int:
                        json_type = "integer"
                    elif param_type == float:
                        json_type = "number"
                    elif param_type == bool:
                        json_type = "boolean"
                    elif param_type == list:
                        json_type = "array"
                    
                    # 添加参数定义
                    parameters["properties"][param_name] = {"type": json_type}
                    
                    # 如果没有默认值则为必填参数
                    if param.default == param.empty:
                        parameters["required"].append(param_name)
                
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": docstring,
                        "parameters": parameters
                    }
                }
                
                api_tools.append(tool_schema)
        
        new_model.tools = api_tools
        logger.info("Created %d API tool schemas", len(api_tools))
        return new_model


class AgentDemo:
    """代理演示类"""
    
    def __init__(self, temperature=0.0, verbose=True):
        """初始化代理演示类"""
        logger.info("Initializing AgentDemo with temperature=%f, verbose=%s", temperature, verbose)
        self.llm = QwenModel(temperature)
        self.verbose = verbose
    
    def create_math_wikipedia_agent(self):
        """创建使用数学和维基百科工具的代理"""
        logger.info("Creating math and Wikipedia agent")
        # 创建数学计算工具
        @tool
        def calculate(expression: str) -> str:
            """计算数学表达式。输入应该是一个有效的数学表达式。"""
            logger.info("Running calculate tool with expression: %s", expression)
            try:
                result = str(eval(expression))
                logger.info("Calculation result: %s", result)
                return result
            except Exception as e:
                error_msg = f"计算错误: {e}"
                logger.error(error_msg)
                return error_msg
        
        # 创建维基百科搜索工具
        @tool
        def wikipedia(query: str) -> str:
            """搜索维基百科。输入应该是一个搜索查询。"""
            # 简化实现，实际应使用Wikipedia API
            logger.info("Running Wikipedia tool with query: %s", query)
            time.sleep(0.5)  # 模拟API调用延迟
            
            if "Tom M. Mitchell" in query:
                result = """
                Tom Michael Mitchell (born August 9, 1951) is an American computer scientist and the 
                Founders University Professor at Carnegie Mellon University (CMU). He is a founder and 
                former Chair of the Machine Learning Department at CMU. Mitchell is known for his 
                contributions to the advancement of machine learning, artificial intelligence, and 
                cognitive neuroscience and is the author of the textbook Machine Learning.
                """
                logger.info("Wikipedia found info about Tom M. Mitchell")
                return result
            result = f"维基百科查询结果: 关于 {query} 的信息。"
            logger.info("Wikipedia returned generic result")
            return result
        
        # 使用langgraph创建代理
        logger.info("Creating React agent with math and Wikipedia tools")
        try:
            agent = create_react_agent(
                model=self.llm,
                tools=[calculate, wikipedia],
                prompt="你是一个有用的助手，可以回答数学计算和维基百科查询。",
            )
            logger.info("Math and Wikipedia agent created successfully")
            return agent
        except Exception as e:
            logger.error("Error creating math and Wikipedia agent: %s", str(e))
            raise
    
    def create_python_repl_agent(self):
        """创建使用Python REPL工具的代理"""
        logger.info("Creating Python REPL agent")
        # 创建Python执行工具
        @tool
        def python_repl(code: str) -> str:
            """执行Python代码。输入应该是有效的Python代码。"""
            logger.info("Running Python REPL tool with code: %s", code[:100] + "..." if len(code) > 100 else code)
            try:
                # 创建一个本地名称空间
                local_ns = {}
                global_ns = {"__builtins__": __builtins__}
                # 执行代码
                exec(code, global_ns, local_ns)
                # 捕获输出
                output = []
                for key, value in local_ns.items():
                    if key.startswith('_'):
                        continue
                    output.append(f"{key}: {value}")
                result = "\n".join(output) if output else "代码执行成功，但没有返回值"
                logger.info("Python execution result: %s", result[:100] + "..." if len(result) > 100 else result)
                return result
            except Exception as e:
                error_msg = f"代码执行错误: {e}"
                logger.error(error_msg)
                return error_msg
        
        # 创建Python REPL代理
        logger.info("Creating React agent with Python REPL tool")
        try:
            agent = create_react_agent(
                model=self.llm,
                tools=[python_repl],
                prompt="你是一个Python编程助手，能够编写和执行Python代码来解决问题。",
            )
            logger.info("Python REPL agent created successfully")
            return agent
        except Exception as e:
            logger.error("Error creating Python REPL agent: %s", str(e))
            raise
    
    def create_custom_tool_agent(self):
        """创建使用自定义工具的代理"""
        logger.info("Creating custom tool agent")
        # 创建日期工具
        @tool
        def get_date(text: str = "") -> str:
            """
            返回今天的日期，用于任何需要知道今天日期的问题。
            输入应该总是一个空字符串，
            这个函数将总是返回今天的日期，任何日期计算应该在这个函数之外进行。
            """
            today = str(date.today())
            logger.info("Date tool returned: %s", today)
            return today
        
        # 创建天气工具
        @tool
        def get_weather(location: str) -> str:
            """
            获取指定位置的天气信息。
            输入应该是一个地点名称。
            """
            # 简化实现，实际应调用天气API
            logger.info("Weather tool called for location: %s", location)
            result = f"{location}的天气: 晴朗，温度26°C。"
            logger.info("Weather tool returned: %s", result)
            return result
        
        # 创建自定义工具代理
        logger.info("Creating React agent with custom tools")
        try:
            agent = create_react_agent(
                model=self.llm,
                tools=[get_date, get_weather],
                prompt="你是一个助手，可以回答关于日期和天气的问题。",
            )
            logger.info("Custom tool agent created successfully")
            return agent
        except Exception as e:
            logger.error("Error creating custom tool agent: %s", str(e))
            raise
    
    def run_agent(self, agent, question):
        """运行代理并处理响应"""
        if self.verbose:
            print(f"\n问题: {question}")
            print("思考和行动过程:")
        
        # 运行代理
        try:
            logger.info("Running agent with question: %s", question)
            user_message = {"role": "user", "content": question}
            
            # Run the agent
            result = agent.invoke({"messages": [user_message]})
            logger.info("Agent returned result of type: %s", type(result).__name__)
            
            # Extract all relevant information for final response
            final_answer = self._process_final_result(result)
            
            if self.verbose:
                print(f"最终回答: {final_answer}\n")
            
            return final_answer
        
        except Exception as e:
            error_msg = f"处理代理回答时出错: {str(e)}"
            logger.error(error_msg)
            logger.exception("详细错误信息:")
            if self.verbose:
                print(f"最终回答: {error_msg}\n")
            return error_msg

    def _process_final_result(self, result):
        """Process the final result to extract tool results and generate a final answer"""
        logger.info("Processing final result to extract meaningful content")
        
        messages = []
        # Handle different result formats
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
        elif hasattr(result, "messages"):
            messages = result.messages
        elif isinstance(result, list):
            messages = result
        else:
            messages = [result] if hasattr(result, "content") else []
        
        # Find all tool results and relevant information
        tool_results = []
        original_question = ""
        
        # First find the original question
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human" and hasattr(msg, "content"):
                original_question = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                original_question = msg["content"]
                break
        
        # Then collect tool execution results
        for i, msg in enumerate(messages):
            # Look for tool messages
            if hasattr(msg, "type") and msg.type == "tool" and hasattr(msg, "content"):
                # Find the tool name from the preceding AIMessage's tool_calls
                tool_name = "unknown_tool"
                for j in range(i-1, -1, -1):
                    prev_msg = messages[j]
                    if (hasattr(prev_msg, "additional_kwargs") and 
                        "tool_calls" in prev_msg.additional_kwargs and 
                        len(prev_msg.additional_kwargs["tool_calls"]) > 0):
                        for tc in prev_msg.additional_kwargs["tool_calls"]:
                            if tc.get("id") == msg.tool_call_id:
                                tool_name = tc.get("name", "unknown_tool")
                                break
                        break
                
                tool_results.append(f"{tool_name}: {msg.content}")
            # Also check for tool execution results directly in the messages
            elif hasattr(msg, "name") and hasattr(msg, "content"):
                tool_results.append(f"{msg.name}: {msg.content}")
        
        # If we have tool results but the final message doesn't have content
        if messages and not (hasattr(messages[-1], "content") and messages[-1].content) and tool_results:
            # Make a second LLM call to summarize results
            logger.info("Final message has empty content, generating summary from tool results")
            
            # Create a more detailed prompt with the tool results
            summary_prompt = f"""
            Question: {original_question}
            
            Tool results:
            {chr(10).join(tool_results)}
            
            Based on these tool results, please provide a complete and direct answer to the original question.
            """
            
            logger.info(f"Summary prompt: {summary_prompt[:100]}...")
            response = self.llm.predict(summary_prompt)
            logger.info(f"Generated summary from tool results: {response[:100]}...")
            return response
        
        # If there's content in the final message
        elif messages and hasattr(messages[-1], "content") and messages[-1].content:
            return messages[-1].content
        
        # Final fallback
        return "无法生成回答。" if not tool_results else f"工具执行结果: {', '.join(tool_results)}"


def main():
    """主函数，演示各种代理功能"""
    load_dotenv(find_dotenv())  # 加载环境变量
    
    print("=== LangGraph 代理演示 ===\n")
    
    agent_demo = AgentDemo()
    
    # 演示使用数学和维基百科工具的代理
    print("=== 1. 使用数学和维基百科工具的代理 ===")
    try:
        logger.info("Creating math and Wikipedia agent")
        math_wiki_agent = agent_demo.create_math_wikipedia_agent()
        
        # 测试数学问题
        math_question = "计算300的25%"
        logger.info("Testing math question: %s", math_question)
        agent_demo.run_agent(math_wiki_agent, math_question)
        
        # 测试维基百科问题
        wiki_question = "Tom M. Mitchell是一位美国计算机科学家，也是卡内基梅隆大学（CMU）的创始人大学教授。他写了哪本书呢？"
        logger.info("Testing Wikipedia question: %s", wiki_question)
        agent_demo.run_agent(math_wiki_agent, wiki_question)
    except Exception as e:
        logger.error("Error in math/Wikipedia demo: %s", str(e))
        logger.exception("Detailed error:")
    
    # 演示使用Python REPL工具的代理
    print("=== 2. 使用Python REPL工具的代理 ===")
    try:
        logger.info("Creating Python REPL agent")
        python_agent = agent_demo.create_python_repl_agent()
        
        # 测试Python代码执行
        customer_list = ["小明", "小黄", "小红", "小蓝", "小橘", "小绿"]
        pinyin_question = f"将这些客户名字转换为拼音，并打印输出列表: {customer_list}。请使用拼音库完成这个任务。"
        logger.info("Testing Python question: %s", pinyin_question)
        agent_demo.run_agent(python_agent, pinyin_question)
    except Exception as e:
        logger.error("Error in Python REPL demo: %s", str(e))
        logger.exception("Detailed error:")
    
    # 演示使用自定义工具的代理
    print("=== 3. 使用自定义工具的代理 ===")
    try:
        logger.info("Creating custom tool agent")
        custom_agent = agent_demo.create_custom_tool_agent()
        
        # 测试日期工具
        date_question = "今天的日期是？"
        logger.info("Testing date question: %s", date_question)
        agent_demo.run_agent(custom_agent, date_question)
        
        # 测试天气工具
        weather_question = "北京的天气如何？"
        logger.info("Testing weather question: %s", weather_question)
        agent_demo.run_agent(custom_agent, weather_question)
    except Exception as e:
        logger.error("Error in custom tool demo: %s", str(e))
        logger.exception("Detailed error:")


if __name__ == "__main__":
    main()