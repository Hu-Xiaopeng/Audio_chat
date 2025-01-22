import gradio as gr
import random
import os
from llama_cpp import Llama
from pywhispercpp.model import Model

# 定义模型名称和路径的映射
model_paths = {
    "Qwen2.5-3B-Instruct-Q4_0_4_4": "../../Qwen2.5-3B-Instruct-Q4_0_4_4.gguf",
    "Qwen2.5-7B-Instruct-Q4_0_4_4": "../../Qwen2.5-7B-Instruct-Q4_0_4_4.gguf",
}
# 创建一个函数来处理模型选择，并返回模型路径
def get_model_path(selected_model_name):
    return model_paths[selected_model_name]

model_path = "../../Qwen2.5-3B-Instruct-Q4_0_4_4.gguf"  # 默认llm模型路径

#  使用 whisper.cpp
# 全局变量
whisper_model = None

def load_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = Model('../../ggml-small-q4_0.bin')
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



# Assuming you are in the Kokoro-82M directory
from onnxruntime import InferenceSession
import torch
import numpy as np
from scipy.io.wavfile import write
import phonemizer
import re


def split_num(num):
    num = num.group()
    if "." in num:
        return num
    elif ":" in num:
        h, m = [int(n) for n in num.split(":")]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f"{h} oh {m}"
        return f"{h} {m}"
    year = int(num[:4])
    if year < 1100 or year % 1000 < 10:
        return num
    left, right = num[:2], int(num[2:4])
    s = "s" if num.endswith("s") else ""
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f"{left} hundred{s}"
        elif right < 10:
            return f"{left} oh {right}{s}"
    return f"{left} {right}{s}"


def flip_money(m):
    m = m.group()
    bill = "dollar" if m[0] == "$" else "pound"
    if m[-1].isalpha():
        return f"{m[1:]} {bill}s"
    elif "." not in m:
        s = "" if m[1:] == "1" else "s"
        return f"{m[1:]} {bill}{s}"
    b, c = m[1:].split(".")
    s = "" if b == "1" else "s"
    c = int(c.ljust(2, "0"))
    coins = (
        f"cent{'' if c == 1 else 's'}"
        if m[0] == "$"
        else ("penny" if c == 1 else "pence")
    )
    return f"{b} {bill}{s} and {c} {coins}"


def point_num(num):
    a, b = num.group().split(".")
    return " point ".join([a, " ".join(b)])


def normalize_text(text):
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace("«", chr(8220)).replace("»", chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = text.replace("(", "«").replace(")", "»")
    for a, b in zip("、。！，：；？", ",.!,:;?"):
        text = text.replace(a, b + " ")
    text = re.sub(r"[^\S \n]", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"(?<=\n) +(?=\n)", "", text)
    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
    text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
    text = re.sub(
        r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)", split_num, text
    )
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(
        r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
        flip_money,
        text,
    )
    text = re.sub(r"\d*\.\d+", point_num, text)
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    text = re.sub(r"(?<=\d)S", " S", text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", "s", text)
    text = re.sub(
        r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text
    )
    text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)
    return text.strip()


phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    ),
    b=phonemizer.backend.EspeakBackend(
        language="en-gb", preserve_punctuation=True, with_stress=True
    ),
)


def phonemize(text, lang, norm=True):
    if norm:
        text = normalize_text(text)
    ps = phonemizers[lang].phonemize([text])
    ps = ps[0] if ps else ""
    # https://en.wiktionary.org/wiki/kokoro#English
    ps = ps.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ").replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ")
    ps = ps.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k").replace("ɬ", "l")
    ps = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", ps)
    ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', "z", ps)
    if lang == "a":
        ps = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", ps)
    ps = "".join(filter(lambda p: p in VOCAB, ps))
    return ps.strip()


def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts


VOCAB = get_vocab()


def tokenize(ps):
    return [i for i in map(VOCAB.get, ps) if i is not None]


def generate(text, voicepack, lang="a", speed=1, ps=None):
    ps = ps or phonemize(text, lang)
    tokens = tokenize(ps)
    return tokens


# Tokens produced by phonemize() and tokenize() in kokoro.py
# tokens = [50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4]

VOICE_NAME = [
    "af",  # Default voice is a 50-50 mix of Bella & Sarah
    "af_bella",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
    "af_nicole",
    "af_sky",
][0]
VOICEPACK = torch.load(f"voices/{VOICE_NAME}.pt", weights_only=True)
print(f"Loaded voice: {VOICE_NAME}")

def process_tts(text, output_dir="output"):
    """
    将文本转换为语音并保存为音频文件。

    参数:
        text (str): 待转换的文本。
        output_dir (str): 保存音频文件的目录，默认为 "output"。

    返回:
        path (str): 保存音频文件的路径。
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    tokens = generate(text, VOICEPACK)

    # Context length is 512, but leave room for the pad token 0 at the start & end
    assert len(tokens) <= 510, len(tokens)

    # Style vector based on len(tokens), ref_s has shape (1, 256)
    ref_s = torch.load("voices/af.pt")[len(tokens)].numpy()

    # Add the pad ids, and reshape tokens, should now have shape (1, <=512)
    tokens = [[0, *tokens, 0]]

    sess = InferenceSession("kokoro-v0_19.onnx")

    audio = sess.run(
        None, dict(tokens=tokens, style=ref_s, speed=np.ones(1, dtype=np.float32))
    )[0]

    # 3️⃣ 保存音频文件
    output_path = os.path.join(output_dir, "output_audio.wav")
    write(output_path, 24000, audio)  # 采样率为 24000 Hz

    print(f"Audio saved to {output_path}")
    return output_path

def get_audio(gr_states, audio):
    """
    在gradio上渲染audio组件, 更新chatbot组件
    """
    if not gr_states["history"]:
        return audio
    
    response = gr_states["history"][-1][1] if len(gr_states["history"][-1]) > 1 else None
    # print(gr_states)
    if response == "Connection Error: 请求失败，请重试" or response is None:
        gr_states["history"].pop()
        return audio
    else:
        audio = process_tts(response)
        return audio


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
                chatbot = gr.Chatbot(label='聊天界面', value=[], height=600, visible=True)
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
            with gr.Row():
                audio = gr.Audio(label="output", interactive=False)



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
        get_audio,  # 第三步：处理音频输出
        inputs=[gr_states, audio],
        outputs=[audio]
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
        get_audio, [gr_states, audio], audio
    ).then(
        clear_message, 
        inputs=[user_prompt], 
        outputs=[user_prompt]
    )
    input_audio.change(audio_to_text, input_audio, user_prompt)
    change_btn.click(set_up, gr_states, [chatbot, gr_states, role_info_display, audio])

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
        outputs=[chatbot, gr_states, role_info_display, audio]  # 输出
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
