import os 
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

def get_api_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['DASHSCOPE_API_KEY']

def create_client():
    api_key = get_api_key()
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def get_completion(prompt, model="qwen-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    return get_completion_from_messages(messages, model, temperature)

def get_completion_from_messages(messages, model="qwen-turbo", temperature=0, max_tokens=500):
    client = create_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_completion_and_token_count(messages, model="qwen-turbo", temperature=0, max_tokens=500):
    client = create_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        token_dict = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return content, token_dict
    except Exception as e:
        return f"Error: {str(e)}", None

def test_llm_call():
    client = create_client()
    
    try:
        response = client.chat.completions.create(
            model='qwen-turbo',
            messages=[{"role": "user", "content": "Hello, please introduce yourself"}]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {str(e)}")

def test_llm_call_with_message_and_token_count():
    messages = [
        {'role': 'system', 'content': '你是一个有帮助的助手'},
        {'role': 'user', 'content': '你是谁？'}
    ]

    content, token_dict = get_completion_and_token_count(messages)
    print("回复内容：", content)
    print("Token统计", token_dict)

# def main():
#     test_llm_call_with_message_and_token_count()

# if __name__ == "__main__":
#     main()