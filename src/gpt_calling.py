import os
import sys

import inspect
import re
from colorama import Fore

import openai
import time
import random
import requests
import base64
from io import BytesIO
from PIL import Image

from .tools import *

from openai.error import (
    APIError,
    OpenAIError,
    RateLimitError,
    APIConnectionError,
    AuthenticationError,
)


def set_api_key():
    if os.environ.get("OPENAI_API_TYPE", "none") == "azure":
        openai.api_key      = os.environ.get("OPENAI_API_KEY")
        openai.api_type     = os.environ.get("OPENAI_API_TYPE", "azure")
        openai.api_version  = os.environ.get("OPENAI_API_VERSION", "2023-03-15-preview")
        openai.api_base     = os.environ.get("OPENAI_API_BASE", "")
    else:
        openai.api_key      = os.environ.get("OPENAI_API_KEY")


def call_openai_completion(engine, messages, temperature=1.0, seed=42):
    # https://platform.openai.com/docs/api-reference/chat/create
    set_api_key()

    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=engine,
                seed=seed,
                messages=messages,
                temperature=temperature,
            )
            response["choices"][0]["message"].to_dict()["content"]
            return response
        except (
                APIError,
                OpenAIError,
                RateLimitError,
                APIConnectionError,
                AuthenticationError,
                KeyError):

            print("RateLimitError / API Error, retrying...")
            time.sleep(2)
            continue
        except Exception as e:
            print("Error:", e)
            exit(-1)


def get_tool_name_description(load_dict):
    models = {}
    # Load Basic Foundation Models
    for class_name, device in load_dict.items():
        models[class_name] = globals()[class_name](device=device)

    # Load Template Foundation Models
    for class_name, module in globals().items():
        if getattr(module, 'template_model', False):
            template_required_names = {k for k in inspect.signature(
                module.__init__).parameters.keys() if k != 'self'}
            loaded_names = set([type(e).__name__ for e in models.values()])
            if template_required_names.issubset(loaded_names):
                models[class_name] = globals()[class_name](
                    **{name: models[name] for name in template_required_names})

    tool_name_description = {}
    tool_name_cookbook = {}
    tools = []
    for instance in models.values():
        for e in dir(instance):
            if e.startswith('inference'):
                func = getattr(instance, e)

                tool_name_description[func.name] = func.description
                tool_name_cookbook[func.name] = func.cookbook
                tools.append(func)

    return tools, tool_name_description, tool_name_cookbook


def tool_function_call(
        load_dict,
        task,
        feedback=None,
        llm_engine="gpt-35-turbo",
        tool_name=None,
        tool_input=None):
    tools, tool_name_description, tool_name_cookbook = get_tool_name_description(
        load_dict)
    # print(f"All the Available Functions: {tool_name_description}")

    old_tool_name = tool_name
    old_tool_input = tool_input

    prompt = f"I have a list of tools that you can call. Here is the list:\n" + \
        f"\n```\n{tool_name_description}\n```\n" + \
        f"I have one task `{task}` to do, could you tell me which tool I should call? You should only respond with ONE tool name."

    response = call_openai_completion(engine=llm_engine, messages=[
                                      {"role": "user", "content": prompt}])

    response_message = response["choices"][0]["message"]
    response_message = response_message.to_dict()["content"]

    tool_name = None
    tool = None

    for _tool, name in zip(tools, tool_name_description.keys()):
        matched_func_name = re.findall(name.lower(), response_message.lower())
        if len(matched_func_name) > 0:
            tool_name = name
            tool = _tool
            break

    if tool_name is None or tool is None:
        print(Fore.RED + "No tool used. Return input task.")
        return task, None, None

    feedback_string = ""
    if feedback is not None and feedback != "":
        feedback_string = f"Feedback to improve the editing quality is \n```\n{feedback}\n```\n"
        feedback_string += f"The former used tool is `{old_tool_name}`. The input to the tool is `{old_tool_input}`.\n"

    cookbook = tool_name_cookbook[tool_name]
    prompt = f"Now I choose the tool `{tool_name}` to do the task `{task}`. " + \
        f"The manual of this tool is \n```{cookbook}```\n" + \
        f"You response should be detailed, e.g., if you need to `add a hat`, you should enrich the instruction like `add a hat to the image of a cute dog`. " + \
        f"Now design the tool input for this task. " + \
        f"{feedback_string}" + \
        f"You MUST ONLY respond in the format of tool name and tool input, sperated by @@. " + \
        f"For example, if the tool name is 'resize-image' and the tool input is 'image/xyzm.png <-> image/resized_image.png <-> 800', you should response 'resize-image @@ image/xyzm.png <-> image/resized_image.png <-> 800'. " + \
        f"All input image paths must exist and do not fake the input image paths.\n"

    response = call_openai_completion(engine=llm_engine, messages=[
                                      {"role": "user", "content": prompt}])
    response_message = response["choices"][0]["message"]
    response_message = response_message.to_dict()["content"]
    tool_input = response_message.split("@@")[1]

    print(
        Fore.CYAN +
        f"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" +
        f"tool call function input:\n {prompt}\n" +
        f"\n\n\n{tool_name}" +
        f"\n\n{tool_input}" +
        f"\n\n{feedback}" +
        f"\n\nNum of tokens: {response['usage']['prompt_tokens']} + {response['usage']['completion_tokens']} = {response['usage']['total_tokens']}" +
        f"\n---------------------------------------------------------\n" +
        f"answers: {response_message}\n" +
        f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" +
        Fore.RESET)

    try:
        results = tool(tool_input.strip("'").strip('"'))
        return results, tool_name, tool_input
    except Exception as e:
        print(Fore.RED + f"Error: {e}, return tool input" + Fore.RESET)
        return tool_input, None, None


def tool_function_callv1(
        load_dict, task, feedback=None,
        llm_engine="gpt-35-turbo",
        tool_name=None, tool_input=None,
        former_exec_io="",
        log_func=print,):
    tools, tool_name_description, tool_name_cookbook = get_tool_name_description(
        load_dict)

    old_tool_name = tool_name
    old_tool_input = tool_input

    task = task.replace("\n", " ")
    prompt = f"I have a list of tools that you can call. Here is the list:\n" + \
             f"\n```\n{tool_name_description}\n```\n" + \
             f"I have one task `{task}` to do, could you tell me which tool I should call? You should only respond with ONE tool name."

    response = call_openai_completion(engine=llm_engine, messages=[
                                      {"role": "user", "content": prompt}])

    response_message = response["choices"][0]["message"]
    response_message = response_message.to_dict()["content"]

    tool_name = None
    tool = None

    for _tool, name in zip(tools, tool_name_description.keys()):
        matched_func_name = re.findall(name.lower(), response_message.lower())
        if len(matched_func_name) > 0:
            tool_name = name
            tool = _tool
            break

    if tool_name is None or tool is None:
        log_func(Fore.RED + "No tool used. Return input task.")
        return task, None, None

    feedback_string = ""
    if feedback is not None and feedback != "":
        feedback_string = f"Feedback to improve the editing quality is \n```\n{feedback}\n```\n"
        feedback_string += f"The tool used in last round is `{old_tool_name}`. The input to the tool is `{old_tool_input}`.\n"

    if former_exec_io != "":
        feedback_string += f"The former execution input/output in current round is \n```\n{former_exec_io}\n```\n"

    cookbook = tool_name_cookbook[tool_name]
    prompt = f"Now I choose the tool `{tool_name}` to do the task `{task}`. " + \
        f"The manual of this tool is \n```{cookbook}```\n" + \
        f"You response should be detailed, e.g., if you need to `add a hat`, you should enrich the instruction like `add a hat to the image of a cute dog`. " + \
        f"Now design the tool input for this task. " + \
        f"{feedback_string}" + \
        f"You MUST ONLY respond in the format of tool name and tool input, sperated by @@. " + \
        f"For example, if the tool name is 'tool1' and the tool input is 'arg1 <-> arg2 <-> arg3', you should response 'tool1 @@ arg1 <-> arg2 <-> arg3'. " + \
        f"All input image paths must exist and do not fake the input image paths.\n"

    response = call_openai_completion(engine=llm_engine, messages=[
                                      {"role": "user", "content": prompt}])
    response_message = response["choices"][0]["message"]
    response_message = response_message.to_dict()["content"]
    tool_input = response_message.split("@@")[1]

    log_func(
        Fore.CYAN +
        f"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" +
        f"tool call function input:\n {prompt}\n" +
        f"\n\n\n{tool_name}" +
        f"\n\n{tool_input}" +
        f"\n\n{feedback}" +
        f"\n\nNum of tokens: {response['usage']['prompt_tokens']} + {response['usage']['completion_tokens']} = {response['usage']['total_tokens']}" +
        f"\n---------------------------------------------------------\n" +
        f"answers: {response_message}\n" +
        f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" +
        Fore.RESET)

    try:
        results = tool(tool_input.strip("'").strip('"'))
        return results, tool_name, tool_input
    except Exception as e:
        log_func(
            Fore.RED +
            f"Error: {e}, return tool input: {tool_input}" +
            Fore.RESET)
        # import pdb; pdb.set_trace()
        return tool_input, None, None


def parse_feedback_for_each_task(
    input_image,
    input_text,
    plans,
    feedback,
    *,
    max_trials=3,
    llm_engine="gpt-4",
    temperature=1.0,
    other_plan_feedbacks=None,
):

    tries = 0
    while True:
        try:
            prompt = f"I want to edit the image `{input_image}` to meet the requirements of the input text `{input_text}`. " + \
                f"The current plan is:\n\n" + \
                f"\n".join([f"{i+1}. {p}" for i, p in enumerate(plans)]) + \
                f"\n\nThe feedback is:\n\n" + \
                f"\n\n{feedback}"

            if other_plan_feedbacks is not None:
                prompt += f"\n\nWe also have the following plan and feedback from others:\n"
                for _plan, _feedback in other_plan_feedbacks:
                    _plan_string = "\n".join(
                        [f"{idx+1}. {subtask}" for idx, subtask in enumerate(_plan)])
                    # reflect_prompt += f"Feedback from plan `{_plan_string}`: `{_feedback}`.\n"
                    prompt += f"\n\nPlan: `{_plan_string}`\nFeedback: `{_feedback}`.\n"

            prompt += f"\n\nPlease give me extract the feedback to each plan step." + \
                f"Your response should be concise and clear. "

            response = call_openai_completion(
                engine=llm_engine,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=temperature,
            )

            response_message = response["choices"][0]["message"]
            response_message = response_message.to_dict()["content"]

            # parse the output
            feedback_for_per_subtask = []
            for line in response_message.split("\n"):
                line = line.strip()
                # is digit
                if line != "" and line[0].isdigit():
                    feedback_for_per_subtask.append(line)
                # else:
                #     feedback_for_per_subtask[-1] += " " + line
            assert len(feedback_for_per_subtask) == len(plans)
            return feedback_for_per_subtask

        except Exception as e:
            print("Error:", e)
            tries += 1
            if tries >= max_trials:
                print("Max trials reached, return empty feedback.")
                return [""] * len(plans)
            else:
                print(f"Retry {tries} times...")
                time.sleep(5)
                continue


def reflector_multi_turn(
    input_image="image/test.png",
    input_text='"Resize the shorter side to 640, add a rainbow on the sky, replace sorghum with a field of sunflowers, and add a rustic wooden barn to the horizon line."',
    current_plan_feedback={
        "plan": [
            "Resize the image to have its shortest side at 640 pixels using the tool 'resize-image'. Input: 'input image path', 'output image path', '640'.",
            "Apply a rainbow to the sky of the image using the tool 'instructdiffusion'.",
            "Replace sorghum with a field of sunflowers on the image using the tool 'grounding-dino-inpainting'.",
            "Add a rustic wooden barn to the horizon line of the image using the tool 'instructdiffusion'."],
        "feedback": "The feedback indicates that the edited image showcases a field of sunflowers under a beautiful sunset, with a dramatic atmosphere created through clouds in the sky. The sunflowers are spread across the field and become the main focus due to their vibrant color against the sunset backdrop. However, elements such as a rainbow or a rustic wooden barn on the horizon line are not evident from the feedback given.\n\nRegarding the comparison with the original image of a sorghum field, there's a clear transformation to a sunflower field, which is successfully carried out. There's a shift in the orientation and perspective of sunflowers in the edited image as compared to the original, which adds to the visual experience. The left part of the edited picture shows a frontal view of sunflowers, while the right part displays a side view. Nevertheless, the sunset seems more prominent in the left part than in the right one.\n\nThe overall quality and coherence of the edited image compared to the original image are maintained. The differences in perspectives and orientations create a unique experience without compromising the quality of the image.\n\nFor improvements, consider adding more elements specified in the task such as a rainbow or a rustic wooden barn to the scene for more complexity. Also, strive for consistency in the appearance of the sunset across the entire edited image to maintain coherence.",
    },
    previous_plan_feedbacks=[],
    tool_names='instructdiffusion, resize-image, edict-editing, grounding-dino-inpainting',
    tool_descriptions={
        'instructdiffusion': 'Useful when you want to edit the image with a text instruction. Can be used for local editing like adding/removing/replacing objects, and global editing like changing the style of the image. ',
        'resize-image': 'resize the image to the given resolution/long side. Useful when you want to resize the image to make the long side to a specific number. ',
        'edict-editing': 'Useful when you want to edit the image with a text instruction. ',
        'grounding-dino-inpainting': 'Useful when you want to edit the image with a text instruction. (usually for object replacement). Detect the object in the image using prompt, and inpaint the object with the given text instruction. '},
    tool_cookbook={
        'instructdiffusion': 'The random seed is a int value, playing a crucial role in the diversity and variability of the generated images. Default is 42. It is encouraged to change the random seed. The text classifier-free guidance (txt-cfg) is a float value, strictly greater than 1.0 and less than 10.0, default is 5.0. The image classifier-free guidance (img-cfg) is a float value, strictly greater than 0.0 and less than 2.0, default is 1.25. A larger txt-cfg value results in an output image showing more editing effects, while a larger img-cfg value leads to an output image more similar to the input image.The input to this tool should be a <-> separated string of six, representing `input image path`, `target image path`, `random seed`, `text classifier-free guidance (txt-cfg)`, `image classifier-free guidance (img-cfg)`, and `text`. no other special instructions are needed.',
        'resize-image': 'receives image_path and the resolution as input. The input to this tool should be a <-> separated string of three, representing the `input image path`, `target image path` and the `resolution` (int value). no other special instructions are needed.',
        'edict-editing': 'receives image_path, base_prompt, and edit_prompt as input. The base prompt is the text describing the input image. The edit prompt is the text describing the edited image. For example, if I want to add a hat to the image of a cute dog, the base prompt could be `a cute dog`, and the edit prompt could be `a cute dog with a hat`. Then the input to this tool should be a <-> separated string of four, representing the `input image path`, the `target image path`, the `base prompt`, and the `edit prompt`. no other special instructions are needed.',
        'grounding-dino-inpainting': 'receives `image path`, `save path`, `detect prompt`, and `inpaint prompt` as input. The `detect prompt` is the text describing the object to be detected. The `inpaint prompt` is the text describing the object to be inpainted. For example, if I want to add change the black car to a cute dog, the `detect prompt` could be `black car`, and the `inpaint prompt` could be `a cute dog`. Then the input to this tool should be a <-> separated string of four, representing the `input image path`, the `target image path`, the `detect prompt`, and the `inpaint prompt`. '},
    llm_engine="gpt-35-turbo",
    log_func=print,
):

    prompt_template = """I want to edit the image given the editing request. Please help me decompose this task into several subtasks. All images should be saved at the same folder `image/`

        Each subtask should be short and specific that can be done by only a single tool. Each subtask should only be tried once. Each subtask should be described in a single line.

        The tool must be one of the following:
        ```
        {tool_names}
        ```

        The detailed description of each tool is as follows:
        ```
        {tool_descriptions}
        ```

        And I have decomposed this task into the following plan (subtask with related tool):
        ```
        {subtasks}
        ```

        The feedback of the above plan is:
        ```
        {feedback}
        ```

        The input image is {input_image_path}. The editing request is `{input_text}`.
    """

    prompt = prompt_template.format(
        input_image_path=input_image,
        input_text=input_text,
        tool_names=tool_names,
        tool_descriptions="\n".join(
            [f"{name}: {tool_descriptions[name]}" for name in tool_descriptions.keys()]),
        # tool_descriptions="\n".join([f"{name}: {tool_descriptions[name]}\n{tool_cookbook[name]}" for name in tool_descriptions.keys()]),
        subtasks="\n".join([f"{idx+1}. {subtask}" for idx,
                            subtask in enumerate(current_plan_feedback['plan'])]),
        feedback=current_plan_feedback['feedback'],
    )

    prompt += f"We also have the following plan and feedback from others:\n"
    for _plan, _feedback in previous_plan_feedbacks:
        _plan_string = "\n".join(
            [f"{idx+1}. {subtask}" for idx, subtask in enumerate(_plan)])
        # reflect_prompt += f"Feedback from plan `{_plan_string}`: `{_feedback}`.\n"
        prompt += f"-------------------------------------------------------------------\n" + \
            f"Plan: `{_plan_string}`\nFeedback: `{_feedback}`.\n"

    new_prompt = prompt + f"Comparing my plan and others' plan, " + \
        f"Do you think I should change the order of the subtasks or change the content of subtasks? " + \
        f"If yes, please tell me the new plan to improve the editing quality. " + \
        f"You should choose one specific tool for each subtask and resize the image first. " + \
        f"If you think I should only change the tool or the input to the tool, please respond with 'no'.\n"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    messages.append({"role": "user", "content": new_prompt})

    response = call_openai_completion(engine=llm_engine, messages=messages)
    response_message = response["choices"][0]["message"].to_dict()["content"]

    plans = []
    feedbacks = parse_feedback_for_each_task(
        input_image=input_image,
        input_text=input_text,
        plans=current_plan_feedback['plan'],
        feedback=current_plan_feedback['feedback'],
        llm_engine=llm_engine,
    )

    feedbacks_after_reflection = []

    # print(Fore.BLUE + f"Response_message: {response_message}" + Fore.RESET)
    stdout = Fore.BLUE + \
        f"\n---------------------------------------------\n" + \
        f"Prompt:\n {new_prompt}\n" + \
        f"Plans: \n{current_plan_feedback['plan']}\n" + \
        f"Feedbacks per subtask: \n{feedbacks}\n" + \
        f"\n\nNum of tokens: {response['usage']['prompt_tokens']} + {response['usage']['completion_tokens']} = {response['usage']['total_tokens']}" + \
        f"\n---------------------------------------------\n" + \
        Fore.RESET
    log_func(stdout)

    if response_message.lower() == 'no' or response_message.lower() == 'no.':
        for subtask, feedback in zip(current_plan_feedback['plan'], feedbacks):

            new_prompt = prompt + f"Comparing my plan and others' plan, " + \
                f"Do you think I should change the tool in the subtask `{subtask}`, whose feedback is {feedback}? " + \
                f"It is encouraged to change the tool if you think the current tool is not suitable for the subtask. " + \
                f"For example, if `instructdiffusion` cannot handle the task <add a horse>, you can try `grounding-dino-inpainting` or `edict-editing`. " + \
                f"If you think I should change the tool, please tell me the new subtask to improve the editing quality. " + \
                f"If no, please respond with 'no'.\n"
            _new_message = messages + \
                [{"role": "assistant", "content": response_message}, {"role": "user", "content": new_prompt},]
            response = call_openai_completion(
                engine=llm_engine, messages=_new_message)
            response_message = response["choices"][0]["message"].to_dict()[
                "content"]

            print(
                Fore.GREEN +
                f"Response_message: {response_message}" +
                Fore.RESET)

            if response_message.lower() == 'no' or response_message.lower() == 'no.':
                plans.append(subtask)
                feedbacks_after_reflection.append(feedback)

            else:
                new_message_1 = _new_message + [{"role": "assistant", "content": response_message}, {
                    "role": "user", "content": "Give me the subtask with new tool."},]
                response_message = call_openai_completion(engine=llm_engine, messages=new_message_1)[
                    "choices"][0]["message"].to_dict()["content"]

                plans.append(response_message)
                feedbacks_after_reflection.append("")

    else:
        for line in response_message.split("\n"):
            line = line.strip()
            if len(line) > 1 and line[0].isdigit() and line[1] == ".":
                plans.append(line[2:].strip())
                feedbacks_after_reflection.append("")

    _plan_string = "\n".join(
        [f"{idx+1}. {subtask}" for idx, subtask in enumerate(plans)])
    stdout = Fore.BLUE + \
        f"\n---------------------------------------------\n" + \
        f"New Plans: \n{_plan_string}\n" + \
        f"Feedbacks after reflection: {feedbacks_after_reflection}\n" + \
        f"---------------------------------------------\n" + \
        Fore.RESET
    log_func(stdout)

    try:
        assert len(plans) == len(feedbacks_after_reflection)
    except BaseException:
        import pdb
        pdb.set_trace()
    return plans, feedbacks_after_reflection
