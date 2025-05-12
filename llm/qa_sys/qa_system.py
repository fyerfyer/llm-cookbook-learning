from llm.qa_sys.content_moderator import ContentModerator
from llm.qa_sys.product_extractor import ProductExtractor
from llm.qa_sys.response_generator import ResponseGenerator
from llm.qa_sys.utils import add_message

class QASystem:
    """端到端问答系统"""

    def __init__(self, delimiter="####", debug=False):
        """
        初始化问答系统

        Args:
            delimiter (str): 分隔符
            debug (bool): 是否启用调试模式
        """
        self.delimiter = delimiter
        self.debug = debug
        self.moderator = ContentModerator()
        self.product_extractor = ProductExtractor(delimiter=delimiter)
        self.response_generator = ResponseGenerator(delimiter=delimiter)

    def process_user_message(self, user_input, all_messages=None):
        """
        处理用户消息的主函数

        Args:
            user_input (str): 用户输入
            all_messages (list): 历史消息列表

        Returns:
            tuple: (回答, 更新后的消息列表)
        """
        if all_messages is None:
            all_messages = [{'role': 'system', 'content': "You are Service Assistant"}]

        if self.debug:
            print("处理用户消息:", user_input)

        # 步骤1: 审核用户输入
        if self.debug:
            print("步骤1: 审核用户输入")

        moderation_result = self.moderator.check_content(user_input)
        if moderation_result.get('flagged', False):
            if self.debug:
                print("用户输入被审核拒绝")
            return "抱歉，您的请求不合规", all_messages

        if self.debug:
            print("用户输入通过审核")

        # 步骤2: 提取产品信息
        if self.debug:
            print("步骤2: 提取产品信息")

        category_and_product_list = self.product_extractor.extract_product_info(user_input)
        if self.debug:
            print("提取的产品信息:", category_and_product_list)

        # 步骤3: 获取产品详细信息
        if self.debug:
            print("步骤3: 获取产品详细信息")

        product_info = self.product_extractor.get_product_information(category_and_product_list)
        if self.debug:
            print("产品详细信息:", product_info)

        # 步骤4: 生成回答
        if self.debug:
            print("步骤4: 生成回答")

        # 将当前查询添加到消息列表
        all_messages = add_message(all_messages, 'user', user_input)

        assistant_response = self.response_generator.generate_response(user_input, product_info)
        if self.debug:
            print("生成的回答:", assistant_response)

        # 步骤5: 审核回答
        if self.debug:
            print("步骤5: 审核回答")

        response_moderation = self.moderator.check_content(assistant_response)
        if response_moderation.get('flagged', False):
            if self.debug:
                print("回答被审核拒绝")
            return "抱歉，我无法提供这方面的信息", all_messages

        if self.debug:
            print("回答通过审核")

        # 步骤6: 评估回答质量
        if self.debug:
            print("步骤6: 评估回答质量")

        is_satisfactory = self.response_generator.evaluate_response(user_input, assistant_response)
        if not is_satisfactory:
            if self.debug:
                print("回答质量不满足要求")
            final_response = "很抱歉，我无法提供您所需的信息。我将为您转接到一位人工客服代表以获取进一步帮助。"
        else:
            if self.debug:
                print("回答质量满足要求")
            final_response = assistant_response

        # 将回答添加到消息列表
        all_messages = add_message(all_messages, 'assistant', final_response)

        return final_response, all_messages

# 测试函数
def main():
    qa_system = QASystem(debug=True)

    print("电子商店客户服务助手\n")
    print("输入 'exit' 退出\n")

    messages = []
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'exit':
            print("感谢使用！再见！")
            break

        response, messages = qa_system.process_user_message(user_input, messages)
        print(f"\n助手: {response}")

if __name__ == "__main__":
    main()