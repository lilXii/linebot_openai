from flask import Flask, request, abort
import os
import traceback

print("DEBUG CHANNEL_ACCESS_TOKEN =", os.getenv("CHANNEL_ACCESS_TOKEN"))
print("DEBUG CHANNEL_SECRET =", os.getenv("CHANNEL_SECRET"))
print("DEBUG OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

import tempfile
import datetime
import openai
import time

app = Flask(__name__)
static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
openai.api_key = os.getenv('OPENAI_API_KEY')
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant for LINE users.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.jsonl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

rag_store = None
if ENABLE_RAG:
    try:
        from rag_utils import RAGStore

        rag_store = RAGStore(KNOWLEDGE_BASE_PATH, EMBEDDING_MODEL)
    except Exception:
        print("Failed to initialize RAG store")
        print(traceback.format_exc())
        rag_store = None


def GPT_response(text):
    context_messages = []
    if rag_store:
        try:
            chunks = rag_store.retrieve(text, top_k=RAG_TOP_K)
            if chunks:
                context_parts = []
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        source = chunk.get("doc_path") or "N/A"
                        chunk_text = chunk.get("text", "")
                    else:
                        source = KNOWLEDGE_BASE_PATH
                        chunk_text = str(chunk)
                    context_parts.append(
                        f"[資料{i + 1}] 來源: {source}\n{chunk_text}"
                    )
                context_text = "\n\n".join(context_parts)
                context_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "以下為檢索到的參考資料，回答時請優先使用這些內容，"
                            "若資料不足請誠實告知：\n"
                            f"{context_text}"
                        ),
                    }
                )
        except Exception:
            print("RAG retrieve failed")
            print(traceback.format_exc())

    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *context_messages,
            {"role": "user", "content": text},
        ],
    )
    print(response)
    answer = response['choices'][0]['message']['content'].strip()
    return answer


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text
    try:
        GPT_answer = GPT_response(msg)
        print(GPT_answer)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(GPT_answer))
    except Exception:
        print(traceback.format_exc())
        line_bot_api.reply_message(event.reply_token, TextSendMessage('你可能使用的 OPENAI API key 額度已超過，請稍後再試。'))
        

@handler.add(PostbackEvent)
def handle_postback(event):
    print(event.postback.data)


@handler.add(MemberJoinedEvent)
def welcome(event):
    uid = event.joined.members[0].user_id
    gid = event.source.group_id
    profile = line_bot_api.get_group_member_profile(gid, uid)
    name = profile.display_name
    message = TextSendMessage(text=f'{name}歡迎加入')
    line_bot_api.reply_message(event.reply_token, message)
        
        
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
