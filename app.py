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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))


def GPT_response(text):
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
