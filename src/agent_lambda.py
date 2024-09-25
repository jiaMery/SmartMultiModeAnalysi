import json
import boto3

# 创建 Connect 客户端
connect_client = boto3.client('connect')

# 创建 SES 客户端
ses_client = boto3.client("ses", region_name="us-east-1")  # 替换为您的 AWS 区域

def lambda_handler(event, context):
    agent = event['agent']
    actionGroup = event['actionGroup']
    function = event['function']
    #parameters = event.get('parameters', [])
    parameters = {param['name']: param['value'] for param in event['parameters']}

    print (parameters)

    text_to_speak = parameters.get('text_to_speak')

    print (text_to_speak)

    def callconnect(text_to_speak):
        # 从事件中获取文本参数
        if not text_to_speak:
            return {
                'statusCode': 400,
                'body': json.dumps('Missing text parameter')
            }

    # 定义邮件内容
    SENDER = "xxxxxx@xx.com"  # 替换为您的发件人邮箱地址
    RECIPIENT = "xxxxx@xx.com"  # 替换为您的收件人邮箱地址

    SUBJECT = "Alert!!"
    BODY_TEXT = text_to_speak

    # 定义邮件内容
    CHARSET = "UTF-8"

    # 创建邮件内容
    body = {
        "Text": {
            "Charset": CHARSET,
            "Data": BODY_TEXT,
        }
    }


     # 发送邮件
    response = ses_client.send_email(
        Destination={"ToAddresses": [RECIPIENT]},
        Message={
            "Body": body,
            "Subject": {"Charset": CHARSET, "Data": SUBJECT},
        },
        Source=SENDER,
        )
        # 打印响应
    print(response)


    result = callconnect(text_to_speak)

    responseBody =  {
        "TEXT": {
            "body": json.dumps(callconnect(text_to_speak))
        }
    }

    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': {
            'responseBody': responseBody
        }

    }

    function_response = {'response': action_response, 'messageVersion': event['messageVersion']}
    print("Response: {}".format(function_response))

    return function_response