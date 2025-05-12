from helper.helper import get_completion_from_messages

class ResponseGenerator:
    """生成用户查询的回答"""

    def __init__(self, delimiter="####"):
        self.delimiter = delimiter

    def generate_response(self, user_query, product_info, temperature=0):
        """
        生成对用户查询的回答

        Args:
            user_query (str): 用户查询
            product_info (str): 产品信息
            temperature (float): 模型温度参数

        Returns:
            str: 生成的回答
        """
        system_message = f"""
        您是一家大型电子商店的客户服务助理。
        请以友好和乐于助人的语气回答问题，并提供简洁明了的答案。
        请确保向用户提出相关的后续问题。
        仅使用提供的产品信息回答。
        如果您不确定答案，请说"我不确定"或"我没有这方面的信息"。
        请勿编造任何产品信息。
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{self.delimiter}{user_query}{self.delimiter}"},
            {"role": "assistant", "content": f"相关商品信息:\n{product_info}"}
        ]

        response = get_completion_from_messages(messages, temperature=temperature)
        return response

    def evaluate_response(self, user_query, assistant_response, temperature=0):
        """
        评估回答是否充分解决了用户查询

        Args:
            user_query (str): 用户查询
            assistant_response (str): 助手回答
            temperature (float): 模型温度参数

        Returns:
            bool: 回答是否充分
        """
        system_message = """
        您是一个客户服务质量评估员。
        您的任务是评估客服助理的回答是否充分解决了用户的问题。
        """

        user_message = f"""
        用户查询: {self.delimiter}{user_query}{self.delimiter}
        助手回答: {self.delimiter}{assistant_response}{self.delimiter}

        这个回答是否充分解决了用户的问题？
        如果充分解决，回答 Y
        如果没有充分解决，回答 N
        仅回答 Y 或 N，不要提供其他解释。
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        response = get_completion_from_messages(messages, temperature=temperature, max_tokens=10)
        return "Y" in response.upper()