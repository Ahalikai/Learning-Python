
# 参考资料 https://zhuanlan.zhihu.com/p/606488403

def regpt():
    import openai
    openai.api_key = 'Key'

    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=input("请输入你的问题:"),
        temperature=1,
        max_tokens=4000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    print(response.choices[0].text)

if __name__ == '__main__':
    for x in range(10000):
        regpt()