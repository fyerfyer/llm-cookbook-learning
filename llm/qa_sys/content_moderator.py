from helper.helper import get_completion_from_messages

class ContentModerator:
    """内容审核类，负责检查用户输入和系统输出是否合规"""

    def check_content(self, input_text):
        """
        使用LLM进行内容审核

        Args:
            input_text (str): 需要审核的内容

        Returns:
            dict: 审核结果
        """
        system_message = """
        你是一个内容审核助手，你需要评估输入内容是否包含以下有害内容类别：
        - 暴力内容
        - 性内容
        - 仇恨言论
        - 自残或自杀
        - 恐怖主义
        - 其他有害内容
        
        请以JSON格式返回审核结果，格式如下：
        {
            "flagged": true/false,
            "categories": {
                "violence": true/false,
                "sexual": true/false,
                "hate": true/false,
                "self-harm": true/false,
                "terrorism": true/false,
                "other_harmful": true/false
            },
            "category_scores": {
                "violence": 0-1分数,
                "sexual": 0-1分数,
                "hate": 0-1分数,
                "self-harm": 0-1分数,
                "terrorism": 0-1分数,
                "other_harmful": 0-1分数
            }
        }
        
        只返回JSON格式，不要有任何其他解释。
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"请审核以下内容：\n\n{input_text}"}
        ]

        response = get_completion_from_messages(messages)

        # 尝试解析JSON响应
        try:
            import json
            import re

            # 尝试提取JSON部分
            json_match = re.search(r'({.*})', response.replace('\n', ''), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(response)

            return result

        except Exception as e:
            print(f"解析审核结果时出错: {e}")
            print(f"原始响应: {response}")
            return {
                "flagged": True,  # 解析失败时保守处理为有害内容
                "error": str(e),
                "raw_response": response
            }