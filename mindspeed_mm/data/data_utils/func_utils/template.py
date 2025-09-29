# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from transformers import PreTrainedTokenizer

from mindspeed_mm.data.data_utils.func_utils.formatters import EmptyFormatter, Formatter, StringFormatter
from mindspeed_mm.data.data_utils.func_utils.formatters import SLOTS
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.data.data_utils.func_utils.mm_plugin import get_mm_plugin

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .formatters import SLOTS
    from .mm_plugin import BasePlugin

logger = get_logger(__name__)


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_observation: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    mm_plugin: "BasePlugin"

    def encode_oneturn(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    def encode_multiturn(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def _encode(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: Sequence[Dict[str, str]],
            system: Optional[str]
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system:
                    elements += self.format_system.apply(content=system)

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(
                    content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(
                    content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(
                    content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(
                    content=message["content"])
            else:
                raise NotImplementedError(
                    "Unexpected role: {}".format(message["role"]))

            encoded_messages.append(
                self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    @staticmethod
    def _convert_elements_to_ids(tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem,
                                                  add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [
                    tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(
                    "Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids


TEMPLATES: Dict[str, "Template"] = {}


@dataclass
class RegisterParams:
    format_user: Optional["Formatter"] = None
    format_assistant: Optional["Formatter"] = None
    format_system: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None
    format_prefix: Optional["Formatter"] = None
    default_system: str = ""
    stop_words: Optional[Sequence[str]] = None
    efficient_eos: bool = False
    replace_eos: bool = False


def _register_template(
        name: str,
        params: RegisterParams,
        mm_plugin: "BasePlugin" = get_mm_plugin(name="base"),
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        RegisterParams(format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True),
    )
    ```
    """
    eos_slots = [] if params.efficient_eos else [{"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(
        slots=["{{content}}"] + eos_slots)

    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = Template(
        format_user=params.format_user or default_user_formatter,
        format_assistant=params.format_assistant or default_assistant_formatter,
        format_system=params.format_system or default_user_formatter,
        format_observation=params.format_observation or params.format_user or default_user_formatter,
        format_prefix=params.format_prefix or default_prefix_formatter,
        default_system=params.default_system,
        stop_words=[] if params.stop_words is None else params.stop_words,
        efficient_eos=params.efficient_eos,
        replace_eos=params.replace_eos,
        mm_plugin=mm_plugin,
    )


_register_template(
    name="qwen2vl",
    params=RegisterParams(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        replace_eos=True),
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")
)

# copied from qwen template
_register_template(
    name="qwen2_omni",
    params=RegisterParams(
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
        ),

        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        replace_eos=True),
    mm_plugin=get_mm_plugin(
        name="qwen2_omni", audio_token="<|AUDIO|>", image_token="<|IMAGE|>", video_token="<|VIDEO|>")

)


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    if tokenizer.eos_token == eos_token:
        return


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", template: str) -> "Template":
    r"""
    Gets chat template and fixes the tokenizer.
    """

    template = TEMPLATES.get(template, None)
    if template is None:
        raise ValueError(
            "Template {} does not exist.".format(template))

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError(
                "Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning(
                "New tokens have been added, make sure `resize_vocab` is True.")

    return template


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning(
            "New tokens have been added, make sure `resize_vocab` is True.")
