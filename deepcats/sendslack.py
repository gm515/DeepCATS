import requests


def slack_message(text, channel, username):
    """
    Slack integration to give slack message to chosen channel. Fill in slack
    hook url below to send to own slack channel.

    Params
    ------
    text : str
        String of text to post.
    channel : str
        String for the channel to post to.
    username : str
        String for the user posting the message.
    """
    # from urllib3 import request
    import json

    post = {"text": "{0}".format(text),
            "channel": "{0}".format(channel),
            "username": "{0}".format(username),
            "icon_url": "https://github.com/gm515/gm515.github.io/blob/master/Images/imperialstplogo.png?raw=true"}

    try:
        json_data = json.dumps(post)
        req = requests.post('https://hooks.slack.com/services/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',
                            data=json_data.encode('ascii'),
                            headers={'Content-Type': 'application/json'})
    except Exception as em:
        print("EXCEPTION: " + str(em))
