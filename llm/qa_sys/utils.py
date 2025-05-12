import json

def format_conversation(messages):
    """
    将对话消息格式化为易读的文本

    Args:
        messages (list): 消息列表

    Returns:
        str: 格式化后的文本
    """
    result = ""
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')

        if role == 'system':
            result += f"系统: {content}\n\n"
        elif role == 'user':
            result += f"用户: {content}\n\n"
        elif role == 'assistant':
            result += f"助手: {content}\n\n"

    return result

def add_message(all_messages, role, content):
    """
    向消息列表添加新消息

    Args:
        all_messages (list): 消息列表
        role (str): 角色
        content (str): 内容

    Returns:
        list: 更新后的消息列表
    """
    all_messages.append({'role': role, 'content': content})
    return all_messages