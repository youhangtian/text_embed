from zhipuai import ZhipuAI
import dashscope 
import erniebot 

ZHIPU_APIKEY = 'befff404d2bbc75f7424535a6c603700.AWkoilteOi1tSQY5'
QWEN_APIKEY = 'sk-93c1f90fc17846279f2a3846c5ff8374'
ERNIE_APIKEY = '6f1d966419c9773d1973a045ead5927d3bcb8c91'

dashscope.api_key = QWEN_APIKEY
erniebot.api_type = 'aistudio'
erniebot.access_token = ERNIE_APIKEY

class LLMAPI:
    def __init__(self, llm_name='zhipu'):
        assert llm_name in [
            'zhipu',
            'qwen',
            'ernie',
        ], f'do not support {llm_name}'
        self.llm_name = llm_name

    def chat(self, prompt):
        ret = {
            'answer': '',
            'usage': [0, 0, 0]
        }

        if self.llm_name == 'zhipu':
            client = ZhipuAI(api_key=ZHIPU_APIKEY)

            response = client.chat.completions.create(
                model='GLM-4',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }]
            )

            ret['answer'] = response.choices[0].message.content 
            usage = response.usage
            ret['usage'] = [usage.prompt_tokens, usage.completion_tokens, usage.total_tokens]

        elif self.llm_name == 'qwen':
            response = dashscope.Generation.call(
                model=dashscope.Generation.Models.qwen_max,
                prompt=prompt,
            )

            ret['answer'] = response.output['text']
            usage = response.usage
            ret['usage'] = [usage['input_tokens'], usage['output_tokens'], usage['total_tokens']]

        elif self.llm_name == 'ernie':
            response = erniebot.ChatCompletion.create(
                model='ernie-4.0',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }]
            )

            res = response.to_dict()['rbody']
            ret['answer'] = res['result']
            usage = res['usage']
            ret['usage'] = [usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']]

        return ret 
