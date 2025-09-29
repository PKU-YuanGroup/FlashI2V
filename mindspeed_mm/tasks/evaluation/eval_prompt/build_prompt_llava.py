from typing import Callable

from mindspeed_mm.tasks.evaluation.eval_prompt.build_prompt_base import BasePromptTemplate
from mindspeed_mm.tasks.evaluation.eval_datasets.datasets_base import datasets_type


class LlavaPromptTemplate(BasePromptTemplate):

    def __init__(self, use_custom_prompt):
        super().__init__(use_custom_prompt)

    def build_prompt(self, line, dump_image: Callable, dataset_name=None):
        target_path = dump_image(line)
        prompt = super()._build_prompt(line)
        message = [dict(type='image', value=s) for s in target_path]
        message.append(dict(type='text', value=prompt))

        return message

    @staticmethod
    def concat_list(message):
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                images.append(item['value'])
        return text, images

    def is_use_custom_prompt(self, dataset_name):
        if datasets_type[dataset_name] == 'MCQ':
            return True
        else:
            return False