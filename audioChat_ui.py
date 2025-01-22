import gradio as gr
import random
from llama_cpp import Llama
from pywhispercpp.model import Model

# 定义模型名称和路径的映射
model_paths = {
    "Qwen2.5-3B-Instruct-Q4_0_4_4": "./Qwen2.5-3B-Instruct-Q4_0_4_4.gguf",
    "Qwen2.5-7B-Instruct-Q4_0_4_4": "./Qwen2.5-7B-Instruct-Q4_0_4_4.gguf",
}
# 创建一个函数来处理模型选择，并返回模型路径
def get_model_path(selected_model_name):
    return model_paths[selected_model_name]

model_path = "./Qwen2.5-3B-Instruct-Q4_0_4_4.gguf"  # 默认llm模型路径

#  使用 whisper.cpp
# 全局变量
whisper_model = None

def load_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = Model('./ggml-small-q4_0.bin')
# audio to text here
def audio_to_text(audio_path):
    """
    audio to text here，目前是 whisper.cpp
    Parameters:
    audio_path: str, 音频文件路径
    Returns:
    transcription.text: str, 音频转换的文本
    """
    load_model()  # 确保模型只加载一次
    if not audio_path:
        return None
    print(f"正在处理audio_path:{audio_path}")
    params = {
        'n_threads': 8,
        'language': 'auto',  
        'initial_prompt': '以下是普通话的句子',
    }
    result = whisper_model.transcribe(audio_path, **params)
    text = ''.join(segment.text for segment in result)
    print("Transcription text:", text)
    return text



class LLMResponse:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            chat_format="chatml",
            n_ctx=1024,
            n_threads=8,
        )

    def get_response(self, messages):
        # 创建聊天完成
        response = self.llm.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=512,
            top_p=0.8,  # 采样概率阈值
            top_k=100,  # 采样数量
            temperature=0.7,  # 温度
        )
        # 流式返回AI的响应
        for chunk in response:
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

def chat_completions(messages, gr_states, history, selected_model_name):
    """
    Chat completion here, using llama-cpp-python now.
    Parameters:
    messages: openai format messages
    gr_states: dict, global states
    history: list, conversation history
    Returns:
    gr_states: dict, updated global states
    history: list, updated conversation history
    """
    if not messages:
        return gr_states, history
    
    # Initialize the LLM model
    model_path = get_model_path(selected_model_name)  # 根据模型名称获取路径
    llm_response = LLMResponse(model_path)
    
    # 构建包含整个对话历史的消息列表
    full_messages = [
        {"role": "system", "content": gr_states["system_prompt"]},
    ]
    # 添加历史对话到消息列表中
    for usr, bot in history:
        full_messages.append({"role": "user", "content": usr})
        if bot:
            full_messages.append({"role": "assistant", "content": bot})
    
    # 添加最新的用户输入
    full_messages.append({"role": "user", "content": messages[-1]['content']})
    
    # Get the response from the LLM model
    ai_response = ""
    for content in llm_response.get_response(full_messages):
        ai_response += content
        gr_states["history"][-1][1] = ai_response  # 更新历史记录
        history[-1][1] = ai_response  # 更新 history
        yield gr_states, history  # 逐步返回更新后的 gr_states 和 history
    
    # Update the conversation history
    user_input = messages[-1]['content']
    if user_input:
        if not gr_states["history"]:
            gr_states["history"] = []
        gr_states["history"].append([user_input, ai_response])
        history.pop()
        history.append(gr_states["history"][-1])
    
    yield gr_states, history



def init_default_role():
    """
    初始化默认角色
    根据角色确定 system prompt
    """
    system_prompt = "你是一只会说话的青蛙，但无论说什么都爱在最后加上'呱唧呱唧'。"
    role = "一只用于演示的青蛙 🐸"
    role_description = "我是一只会说话的青蛙🐸，但无论说什么都爱在最后加上'呱唧呱唧'。"
    return role, role_description, system_prompt

def get_random_role():
    """
    随机获取一个角色，这里只是一个示例函数
    根据角色确定 system prompt
    """
    roles = [
        {
            "name": "0号小青蛙 🐸",
            "description": "我是一只会说话的青蛙🐸，但无论说什么都爱在最后加上'呱唧呱唧'。",
            "system_prompt": "你是一只会说话的青蛙，但无论说什么都爱在最后加上'呱唧呱唧'。"
        },
        {
            "name": "1号小老虎 🐯",
            "description": "我是一只勇敢的小老虎🐯，说话时总是充满力量，喜欢在句子末尾加上'嗷呜嗷呜'。",
            "system_prompt": "你是一只勇敢的小老虎，说话时总是充满力量，喜欢在句子末尾加上'嗷呜嗷呜'。"
        },
        {
            "name": "2号小猫咪 🐱",
            "description": "我是一只温柔的小猫咪🐱，说话时声音柔和，喜欢在句子末尾加上'喵喵喵喵'。",
            "system_prompt": "你是一只温柔的小猫咪，说话时声音柔和，喜欢在句子末尾加上'喵喵喵喵'。"
        },
        {
            "name": "3号小水牛 🐮",
            "description": "我是一只勤劳的小水牛🐮，说话时总是慢条斯理，喜欢在句子末尾加上'哞哞哞哞'。",
            "system_prompt": "你是一只勤劳的小水牛，说话时总是慢条斯理，喜欢在句子末尾加上'哞哞哞哞'。"
        },
        {
            "name": "4号小狗狗 🐶",
            "description": "我是一只忠诚的小狗狗🐶，说话时总是热情洋溢，喜欢在句子末尾加上'汪呜汪呜'。",
            "system_prompt": "你是一只忠诚的小狗狗，说话时总是热情洋溢，喜欢在句子末尾加上'汪呜汪呜'。"
        },
        {
            "name": "5号小喜鹊 🐦",
            "description": "我是一只快乐的小喜鹊🐦，说话时总是带着好消息，喜欢在句子末尾加上'喳喳唧唧'。",
            "system_prompt": "你是一只快乐的小喜鹊，说话时总是带着好消息，喜欢在句子末尾加上'喳喳唧唧'。"
        }
    ]
    
    i = random.randint(0, len(roles) - 1)
    selected_role = roles[i]
    
    role = selected_role["name"]
    role_description = selected_role["description"]
    system_prompt = selected_role["system_prompt"]
    
    return role, role_description, system_prompt



def format_messages(user_message, gr_states, history):
    """prepare the request data [messages] for the chatbot
    Parameters:
    user_message: str, 用户输入的消息
    gr_states: dict, {"system_prompt": str, "history": List, "user_prompt": str}
    history: list, 聊天记录，一个嵌套列表: [["用户消息", "bot回复"],["用户消息", "bot回复"]]
    """
    messages = [
        {
            "role": "system",
            "content": gr_states["system_prompt"],
        },
    ]
    history.append([user_message, None])
    if len(user_message) > 0:
        if not gr_states["history"]:
            gr_states["history"] = []
        gr_states["history"].append([user_message, None])
    for [usr, bot] in history:
        messages.append({
            "role": "user",
            "content": usr
        })
        if bot:
            messages.append({
                "role": "assistant",
                "content": bot
            })
    return messages, gr_states, history

def set_up(gr_states):
    """
    maybe 随机切换一个角色
    """
    role_name, role_description, system_prompt = get_random_role()
    gr_states = {"system_prompt": system_prompt, "history": []}
    role_info_display = f''' # {role_name}
    {role_description}
    '''
    history = []
    return history, gr_states, role_info_display, None


theme = gr.themes.Soft(
    font=['ui-sans-serif'],
    font_mono=['ui-monospace'],
)

with gr.Blocks(theme=theme) as demo:
    demo.title = 'Audio Chat 🎙️'
    gr.Markdown('''<center><strong style="font-size: 24px;">✨ Audio Chat 🎙️</strong></center>''')
    
    role_name, role_description, system_prompt = init_default_role()
    gr_states = gr.State({"system_prompt": system_prompt, "history": []})
    messages = gr.State(None)
    # with gr.Tab(label='demo'):
    with gr.Row():
        role_info_display = gr.Markdown(f''' # {role_name}
        {role_description}
        ''')
    with gr.Row():
        with gr.Column(scale=7):
            with gr.Row():
                chatbot = gr.Chatbot(label='聊天界面', value=[], height=500, visible=True)
            with gr.Row():
                with gr.Column(scale=2):
                    user_prompt = gr.Textbox(label='对话输入框（按Enter发送消息）', interactive=True, visible=True)
                    user_prompt_state = gr.State("")  # 创建一个状态变量来存储输入框的内容
                with gr.Column(scale=1):
                    with gr.Row():
                        send_btn = gr.Button("Send Message")
                    with gr.Row():
                        clear_btn = gr.Button("Clear Message")

        with gr.Column(scale=3):
            with gr.Row():
                change_btn = gr.Button("随机换一个角色")
            with gr.Row():
                default_role_btn = gr.Button("切换回默认角色")
            with gr.Row():
                model = gr.Dropdown(
                                label="Choose Model:",
                                choices=list(model_paths.keys()),  # 只显示模型的名称
                                value=list(model_paths.keys())[0] if model_paths else None,  # 默认选择第一个模型
                                interactive=True,
                                allow_custom_value=True,
                            )
            with gr.Row():
                input_audio = gr.Audio(label="语音输入框", type="filepath")
            with gr.Row():
                examples_list = [
                    [
                        "./audio_test/1.mp3",
                    ],
                    [
                        "./audio_test/2.mp3",
                    ],
                    [
                        "./audio_test/3.mp3",
                    ],
                    [
                        "./audio_test/4.mp3",
                    ],
                ]
                gr.Examples(examples=examples_list, inputs=[input_audio])



    # 定义clear message按钮的回调函数
    def clear_message(user_prompt):
        """
        清空输入框的回调函数
        """
        return ""

    # 将send message按钮与回调函数关联
    send_btn.click(
        format_messages,  # 第一步：格式化消息
        inputs=[user_prompt, gr_states, chatbot],
        outputs=[messages, gr_states, chatbot]
    ).then(
        chat_completions,  # 第二步：处理聊天完成
        inputs=[messages, gr_states, chatbot, model],
        outputs=[gr_states, chatbot]
    ).then(
        clear_message, 
        inputs=[user_prompt], 
        outputs=[user_prompt]
    )

    # 将clear message按钮与回调函数关联
    clear_btn.click(
        clear_message, 
        inputs=[user_prompt], 
        outputs=[user_prompt]
    )
    
    user_prompt.submit(
        format_messages, [user_prompt, gr_states, chatbot], [messages, gr_states, chatbot]).then(
        chat_completions, [messages, gr_states, chatbot, model], [gr_states, chatbot]).then(
        clear_message, 
        inputs=[user_prompt], 
        outputs=[user_prompt]
    )
    input_audio.change(audio_to_text, input_audio, user_prompt)
    change_btn.click(set_up, gr_states, [chatbot, gr_states, role_info_display])

    def reset_to_default_role(gr_states):
        """
        重置为默认角色的回调函数
        Parameters:
        gr_states: dict, 全局状态
        Returns:
        history: list, 清空的聊天记录
        gr_states: dict, 更新后的全局状态
        role_info_display: str, 更新后的角色信息显示
        audio: None, 清空音频输出
        """
        # 获取默认角色信息
        role_name, role_description, system_prompt = init_default_role()
        
        # 更新全局状态
        gr_states = {"system_prompt": system_prompt, "history": []}
        
        # 更新角色信息显示
        role_info_display = f''' # {role_name}
        {role_description}
        '''
        
        # 清空聊天记录
        history = []
        
        # 清空音频输出
        audio = None
        
        return history, gr_states, role_info_display, audio

    # 在 gr.Blocks 中关联按钮和回调函数
    default_role_btn.click(
        reset_to_default_role,  # 回调函数
        inputs=[gr_states],  # 输入
        outputs=[chatbot, gr_states, role_info_display]  # 输出
    )

    def update_model_path(selected_model_name):
        global model_path
        model_path = get_model_path(selected_model_name)
        return model_path
    model.change(
        update_model_path, 
        inputs=[model], 
        outputs=gr.State()  # 使用状态变量来存储模型路径
    )

demo.launch(server_port=9877, share=True)
