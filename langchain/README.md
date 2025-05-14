### `model_demo`中遇到的一些错误

1. `StructuredOutputParser`这个解析器好像用不了了，所以使用了`PydanticOutputParser`。
2. **如果直接调用langchain封装的openai client的话，貌似会返回各种乱七八糟的不兼容错误**：

```bash
=== 邮件风格转换示例 ===
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "E:\download\llm-cookbook-learning\langchain\model_demo.py", line 143, in <module>
    main()
    ~~~~^^
  File "E:\download\llm-cookbook-learning\langchain\model_demo.py", line 95, in main
    formal_response = translate_email_style(customer_email, formal_style)
  File "E:\download\llm-cookbook-learning\langchain\model_demo.py", line 38, in translate_email_style
    llm = create_llm(temperature)
  File "E:\download\llm-cookbook-learning\langchain\model_demo.py", line 15, in create_llm
    return ChatOpenAI(
        temperature=temperature,
    ...<2 lines>...
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
  File "E:\download\llm-cookbook-learning\.venv\Lib\site-packages\langchain_core\load\serializable.py", line 130, in __init__
    super().__init__(*args, **kwargs)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\download\llm-cookbook-learning\.venv\Lib\site-packages\pydantic\main.py", line 253, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
  File "E:\download\llm-cookbook-learning\.venv\Lib\site-packages\langchain_core\language_models\base.py", line 94, in _get_verbosity
    return get_verbose()
  File "E:\download\llm-cookbook-learning\.venv\Lib\site-packages\langchain_core\globals.py", line 85, in get_verbose
    old_verbose = langchain.verbose
                  ^^^^^^^^^^^^^^^^^
AttributeError: module 'langchain' has no attribute 'verbose'
```

因此在`model_demo`中直接自己封装了llm client的invoke方法来处理模型调用。