import requests
import json


def send(text):
    requests.post('https://hooks.slack.com/services/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', data = json.dumps({
        'text': text, # 投稿するテキスト
        'username': u'From Python Server', 
        'icon_emoji': u':ghost:', 
        # 'link_names': 1, # メンションを有効にする
    }))

if __name__ == "__main__":
    send(u'Complete All Command')