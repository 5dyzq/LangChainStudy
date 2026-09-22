from __future__ import annotations

"""DeepSeek 调用封装。

学习提示：
- 当前文件暂时不依赖 LangChain，目的是先理解大模型 API 的最小调用过程。
- 后续接入 LangChain 时，这一层可以替换为 LangChain 的 ChatModel，例如
  init_chat_model(...) 或对应 provider 的 ChatDeepSeek。
- 后续接入 LangGraph 时，DeepSeekClient 通常不会直接成为图节点；更常见的做法是：
  在某个节点函数中调用 LLM，然后把结果写回 LangGraph 的 state。

当前调用链：
prompt 构造函数 -> DeepSeekClient.chat(...) -> DeepSeek API -> 文本结果
"""

import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


@dataclass
class DeepSeekClient:
    """最小可用的 DeepSeek Chat 客户端占位实现。

    参数：
        api_key:
            DeepSeek API Key。若不传，则从环境变量 DEEPSEEK_API_KEY 读取。
            这样可以避免把密钥写入代码或上传到 GitHub。
        base_url:
            DeepSeek Chat Completions 接口地址。通常不需要修改。
            如果后续切换到兼容 OpenAI 协议的本地模型服务，也可以改这里。
        model:
            要调用的模型名称，例如 deepseek-chat。
            后续如果比较多个模型，可以在实验配置中控制这个字段。
    """

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None

    def chat(self, messages: list[dict], temperature: float = 0.2) -> str:
        """发送多轮消息并返回模型回复文本。

        参数：
            messages:
                Chat Completions 格式的消息列表。每个元素通常包含：
                - role: "system"、"user" 或 "assistant"
                - content: 消息文本

                示例：
                [
                    {"role": "system", "content": "你是数据分析助手"},
                    {"role": "user", "content": "请分析这些指标..."}
                ]

                在 LangChain 中，messages 会对应 SystemMessage、HumanMessage 等对象；
                在 LangGraph 中，messages 通常会作为 state 的一部分在节点间传递。

            temperature:
                控制输出随机性。越低越稳定，越高越发散。
                数据分析、实验报告建议使用 0.0 到 0.3；
                头脑风暴、方案探索可以使用 0.5 到 0.8。

        返回：
            模型生成的文本内容。
        """
        # 读取项目根目录下的 `.env` 文件
        load_dotenv()

        # `os.getenv("DEEPSEEK_API_KEY")`：去环境变量（.env 里）读 `DEEPSEEK_API_KEY`
        api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        base_url = self.base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/chat/completions"
        model = self.model or os.getenv("DEEPSEEK_MODEL") or "deepseek-flash"

        # 这里直接使用 requests，便于学习 API 的原始请求结构。
        # 后续接 LangChain 后，headers/json/错误处理通常由模型封装负责。

        # response = requests.post(...)发起 POST 请求，向 DeepSeek 接口发消息：
        #
        # - `base_url`：请求地址
        # - `headers` 请求头：
        #   - `Authorization: Bearer {api_key}`：API 鉴权，告诉服务器你是谁
        #   - `Content-Type: application/json`：告诉服务器，请求体是 JSON 格式
        # - `json={...}`：请求体，传给大模型的参数
        #   - `model`：模型名字
        #   - `messages`：对话历史（用户消息 + 助手历史）
        #   - `temperature`：温度，控制随机性，越大回答越发散
        # - `timeout=60`：60 秒超时，60 秒没返回直接报错，防止程序卡死
        response = requests.post(
            base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=60,
        )

        # 自动判断 HTTP 返回码：如果是 401（密钥错）、404、500 这类错误码，直接抛出异常，不用自己判断
        response.raise_for_status()

        # 把接口返回的 JSON 字符串，转换成 Python 字典，方便读取里面的数据。
        data = response.json()

        # 从返回数据里取出第一条模型回答的文本内容，作为函数结果返回。
        return data["choices"][0]["message"]["content"]
