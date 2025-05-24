### 使用langchain-0.0.235旧版时遇到的问题

因为llm-cookbook的示例代码和新版的langchain不太兼容，因此用0.0.235版本写了旧版的示例。遇到的问题如下：

1. 在agent实现中，直接使用`@tool`返回了下面的报错：

```bash
  File "E:\download\llm-cookbook-learning\.venv\lib\site-packages\langchain\agents\utils.py", line 10, in validate_tools_single_inpu    raise ValueError(
ValueError: ChatAgent does not support multi-input tool get_time_tool.
```

改成使用`StructuredTool`就没问题了。