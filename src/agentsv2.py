import os
import sys

import inspect
import re

from colorama import Fore
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.gpt_calling import (
    tool_function_callv1,
    call_openai_completion,
    reflector_multi_turn
)
from src.tools import *


def stream_logging(filename, string, verbose=True):
    if verbose:
        print(string)
    with open(filename, "a", encoding='utf-8') as f:
        f.write(string)


class BaseAgent(object):
    def __init__(self, toolset, llm_engine="gpt-35-turbo", role="Planner"):

        assert role in [
            "Planner",
            "Reflector",
            "Tool-executor",
            "Discriminator",
            "Summarizer"]
        # planner is used to make precise plan, using which tool
        # reflector is used to decide if is plan is good and can meet the request, besides, when we get a feedback of result, need to decide which step can be modified.
        # tool executor uses tool to get results, need to be expert to use
        # differnt tool, explore the implicit parameter according to each
        # tool's specific instruction instead of only tool descrition.
        assert llm_engine in [
            "gpt-4",
            "gpt-35-turbo",
            "gpt-4-32k",
            "gpt-4-turbo"]
        self.llm_engine = llm_engine
        # e.g., toolset = {'VisualQuestionAnswering':'cuda:0',
        # 'ImageCaptioning':'cuda:1',...}

        self.toolset = toolset
        self._init_tools(toolset)

    def _init_tools(self, load_dict):
        models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {
                    k for k in inspect.signature(
                        module.__init__).parameters.keys() if k != 'self'}
                loaded_names = set([type(e).__name__ for e in models.values()])
                if template_required_names.issubset(loaded_names):
                    models[class_name] = globals()[class_name](
                        **{name: models[name] for name in template_required_names})

        self.models = models

        self.tool_name_description = {}
        self.tool_name_cookbook = {}
        for instance in models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    # tools.append(Tool(name=func.name, description=func.description, func=func))
                    self.tool_name_description[func.name] = func.description
                    self.tool_name_cookbook[func.name] = func.cookbook

        # return tools


class Planner(BaseAgent):
    def __init__(self, toolset, llm_engine, role="Planner"):
        super().__init__(toolset, llm_engine, role=role)

    def __call__(
            self,
            input_image,
            input_text,
            detailed_caption="",
            log_func=print):
        ADDITIONAL_INFO = """The detailed description of the input image is `{detailed_caption}`."""
        PLANNING_TEMPLATE = """I want to edit the image using user's editing request. Please help me decompose this task into several subtasks. All images should be saved at the same folder `image/`

Each subtask should be short and specific that can be done by only a single tool. Each subtask should only be tried once. Each subtask should be described in a single line.
The tool should be one of the following:
```
    {tool_names}
```

The detailed description of each tool is as follows:
```
    {tool_descriptions}
```

For example, if the task is `rotate the image, and create a vintage-style portrait of a person with a hat, and adjust the image to have a sepia tone.` given input image, I can decompose this task into the following steps:
1. rotate the image; 2. Add a vintage-style hat to the person in the image; 3. Apply a sepia tone filter to the entire image.

Do not include specific input/output image path in subtasks. If you have to resize the image, put it at the first. The final response should be concise and clear.

The input image is {input_image}. User's editing request is `{input_text}`. {additional_info}

Now give me the plan."""

        self.user_input_text = input_text
        self.user_input_image = input_image

        input_text = PLANNING_TEMPLATE.format(
            input_image=input_image,
            input_text=input_text,
            tool_names=", ".join(self.tool_name_description.keys()),
            tool_descriptions="\n".join(
                [f"{name}: {description}" for name, description in self.tool_name_description.items()]),
            # tool_descriptions_cookbook="\n".join([f"{name}: {self.tool_name_description[name]}\n{self.tool_name_cookbook[name]}" for name in self.tool_name_description.keys()]),
            additional_info=ADDITIONAL_INFO.format(
                detailed_caption=detailed_caption) if len(detailed_caption) > 0 else "",
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ]

        response = call_openai_completion(
            engine=self.llm_engine,
            messages=messages,
            temperature=0.8)
        response_message = response["choices"][0]["message"]
        response_message = response_message.to_dict()["content"]

        stdout = Fore.RED + \
            f"\n---------------------------------------------\n" + \
            f"Input Text: \n{input_text}\n" + \
            f"\nNum of tokens: {response['usage']['prompt_tokens']} + {response['usage']['completion_tokens']} = {response['usage']['total_tokens']}" + \
            f"\n---------------------------------------------\n" + \
            Fore.RESET
        log_func(stdout)

        # parse the response to a list of subtasks
        sub_tasks = []
        for line in response_message.split("\n"):
            # if starts with a number and a dot
            if len(line) > 1 and line[0].isdigit() and line[1] == ".":
                sub_tasks.append(line[2:].strip())

        stdout = Fore.BLUE + \
            f"\n---------------------------------------------\n" + \
            f"Initial Plans: \n" + \
            "\n".join(sub_tasks) + \
            f"\n---------------------------------------------\n" + \
            Fore.RESET
        log_func(stdout)

        return sub_tasks


class Reflector(BaseAgent):
    def __init__(self, toolset, llm_engine, role="Reflector"):
        super().__init__(toolset, llm_engine, role=role)

    def __call__(
            self,
            input_image,
            input_text,
            current_plan_feedback,
            previous_plan_feedback=None,
            log_func=print):
        # feedback is None when it is the first time to reflect the plan
        if current_plan_feedback is None or previous_plan_feedback is None:
            # only verify if the planning is good.
            pass
        else:
            new_plans, new_feedbacks = reflector_multi_turn(
                input_image=input_image,
                input_text=input_text,
                current_plan_feedback=current_plan_feedback,
                previous_plan_feedbacks=previous_plan_feedback,
                tool_names=", ".join(self.tool_name_description.keys()),
                tool_descriptions=self.tool_name_description,
                tool_cookbook=self.tool_name_cookbook,
                llm_engine=self.llm_engine,
                log_func=log_func,
            )

        return new_plans, new_feedbacks


class ToolExecutor(BaseAgent):
    def __init__(self, toolset, llm_engine, role="Planner"):
        super().__init__(toolset, llm_engine, role=role)

        self.prefix = """You should choose only one tool and execute it to finish the subtask. You should only think and act once.
        All input image paths MUST exist.

        The tool should be one of the following:
        ```
        {tool_names}
        ```

        The detailed description of each tool is as follows:
        ```
        {tool_descriptions}
        ```
        """

        self.temp_images = []
        self.tool_names = []
        self.tool_inputs = []
        self.subtasks = []

        self.temp_images_current_round = []

    def execute_step(
            self,
            image_path: str,
            target_image_path: str,
            subtask: str,
            feedback: str = None,
            log_func=print,
            tool_name: str = None,
            tool_input: str = None):

        # subtask = f"\nThe subtask is `{subtask.strip()}`. \nYour response should be concise and clear. " + \
        #           f"The input image path can be set as `{image_path}` if required. Target image path can be set as `{target_image_path}` if required."
        # subtask = f"\nThe subtask is `{subtask.strip()}`. \nYour response should be concise and clear. "
        subtask = f"\nThe subtask is `{subtask.strip()}`. \nYour response should be concise and clear. " + \
                  f"save path can be set as `{target_image_path}` if required. " + \
                  f"The input should rely on the previous subtask. "

        # former exec io
        former_exec_io = f"The initial input image is `{self.temp_images_current_round[0]}`. \n"
        for i in range(len(self.tool_names)):
            former_exec_io += f"The subtask is `{self.subtasks[i]}`. \n" + \
                              f"The execute command is {self.tool_names[i]} @@ {self.tool_inputs[i]}\n" + \
                              f"The output of the subtask is `{self.temp_images_current_round[i + 1]}`.\n\n"

        result, tool_name, tool_input = tool_function_callv1(
            load_dict=self.toolset, task=subtask,
            tool_name=tool_name, tool_input=tool_input,
            former_exec_io=former_exec_io,
            llm_engine=self.llm_engine, feedback=feedback,
            log_func=log_func)

        stdout = Fore.MAGENTA + \
            f"\n---------------------------------------------\n" + \
            f"Subtask: \n{subtask}\n" + \
            f"\nResult of execution: \n{result}\n" + \
            f"\nSpecific Command: \n{tool_name} @@ {tool_input}\n" + \
            f"---------------------------------------------\n" + \
            Fore.RESET
        log_func(stdout)

        _img = re.findall(r"image/.*\.(?:png|jpg)", result)[0]
        if not os.path.exists(_img):
            _img = self.temp_images_current_round[-1]
        self.temp_images.append(_img)
        self.temp_images_current_round.append(_img)
        self.tool_names.append(tool_name)
        self.tool_inputs.append(tool_input)

        return result, tool_name, tool_input

    def __call__(
            self,
            plan,
            input_image,
            input_text="",
            save_prefix="image/step",
            feedback=None,
            log_func=print,
            tool_names=None,
            tool_inputs=None):
        assert isinstance(plan, list)
        self.temp_images.append(input_image)

        self.tool_names = []
        self.tool_inputs = []
        self.temp_images_current_round = [input_image]
        self.subtasks = plan

        feedback = [None] * len(plan) if feedback is None else feedback
        tool_names = [None] * len(plan) if tool_names is None else tool_names
        tool_inputs = [None] * \
            len(plan) if tool_inputs is None else tool_inputs

        for subtask, feedback, tool_name, tool_input in zip(
                plan, feedback, tool_names, tool_inputs):
            self.execute_step(
                subtask=subtask, image_path=self.temp_images[-1],
                target_image_path=f"{save_prefix}_{len(self.temp_images) - 1}.png",
                tool_name=tool_name, tool_input=tool_input,
                feedback=feedback, log_func=log_func)
            # input_image = f"image/step_{idx}.png" if os.path.exists(f"image/step_{idx}.png") else input_image
            # if os.path.exists(f"image/step_{idx}.png"):
            #     input_image = f"image/step_{idx}.png"
            #     self.temp_images.append(input_image)

        return self.temp_images, self.tool_names, self.tool_inputs


class GeneratorAgent:

    def __init__(self, agent_list):

        # agent_list = [
        #     {
        #         "role": "Planner",
        #         "llm_engine": "gpt-35-turbo",
        #         "toolset": {"InstructDiffusion": 'cuda:0', "LLaVA": 'cuda:0'}
        #     },
        #     ...
        # ]

        for agent in agent_list:
            if agent["role"] == "Planner":
                self.planner = Planner(
                    toolset=agent["toolset"],
                    llm_engine=agent["llm_engine"],
                    role=agent["role"])
            if agent["role"] == "Tool-executor":
                self.tool_executor = ToolExecutor(
                    toolset=agent["toolset"],
                    llm_engine=agent["llm_engine"],
                    role=agent["role"])
            if agent["role"] == "Reflector":
                self.reflector = Reflector(toolset=agent["toolset"],
                                           llm_engine=agent["llm_engine"],
                                           role=agent["role"])

        self.planning_results = []
        self.temp_images = []

        self.tool_names = []
        self.tool_inputs = []

        self.input_image_discription = ""
        self.input_image_path = ""

        self.save_prefix = "image/step"
        self.agend_id = 0

        self.best_images = []
        self.best_image_feedbacks = []

    def set_agent_id(self, agent_id):
        self.agend_id = agent_id

    def set_save_prefix(self, save_prefix):
        self.save_prefix = save_prefix

    def save_best_image(self, best_image, feedback):
        self.best_images.append(best_image)
        self.best_image_feedbacks.append(feedback)

    def forward(
            self,
            input_image,
            input_text,
            detailed_caption="",
            log_func=print):
        self.input_image_path = input_image

        # generate a plan
        self.planning_results.append(
            self.planner(
                input_image,
                input_text,
                detailed_caption=detailed_caption,
                log_func=log_func))

        # execute the plan
        self.temp_images, tool_names, tool_inputs = self.tool_executor(
            self.planning_results[-1],
            input_image, input_text,
            save_prefix=self.save_prefix,
            log_func=log_func)
        self.tool_names.append(tool_names)
        self.tool_inputs.append(tool_inputs)
        return self.temp_images[-1], tool_names, tool_inputs

    def modification(self, input_image, input_text, feedback, log_func=print):

        # TBD, suppose we have feedback, this does not include any debate or compete part
        # 1. send feedback to reflector
        planning_results = feedback["planning_results"]
        feedbacks = feedback["feedbacks"]

        current_plan_feedback = {
            "plan": planning_results[self.agend_id][-1], "feedback": feedbacks[self.agend_id]}

        previous_plan_feedback = []
        for idx, (plan, feedback) in enumerate(
                zip(planning_results, feedbacks)):
            if idx != self.agend_id:
                previous_plan_feedback.append((plan[-1], feedback))
                break

        new_plan, new_feedack = self.reflector(
            input_image, input_text,
            current_plan_feedback=current_plan_feedback,
            previous_plan_feedback=previous_plan_feedback, log_func=log_func)
        self.planning_results.append(new_plan)

        # 2. execute the plan
        self.temp_images, tool_names, tool_inputs = self.tool_executor(
            self.planning_results[-1], input_image,
            input_text, save_prefix=self.save_prefix,
            tool_names=self.tool_names[-1],
            tool_inputs=self.tool_inputs[-1],
            feedback=new_feedack, log_func=log_func)
        self.tool_names.append(tool_names)
        self.tool_inputs.append(tool_inputs)
        return self.temp_images[-1], tool_names, tool_inputs

    def _generate_detailed_caption(self, input_image, log_func=print):
        return LLaVA(
            device="cuda:0").inference(
            inputs=f"{input_image} <-> Provide a one-sentence caption for the provided image.")

    def __call__(
            self,
            input_image,
            input_text,
            feedback=None,
            log_func=print) -> Any:
        if self.input_image_discription is None or len(
                self.input_image_discription) == 0:
            self.input_image_discription = LLaVA(device="cuda:0").inference(
                inputs=f"{input_image} <-> Provide a one-sentence caption for the provided image.")

        if feedback is None or len(self.planning_results) == 0:
            return self.forward(
                input_image,
                input_text,
                self.input_image_discription,
                log_func=log_func)
        else:
            return self.modification(
                input_image, input_text, feedback, log_func=log_func)


class Discriminator(BaseAgent):
    def __init__(self, toolset, llm_engine, role="discriminator"):
        super().__init__(toolset, llm_engine, role=role)

        self._init_prompt()
        self.prompt = """We want to edit the image using instruction `{input_text}`. The detailed description of the input image is `{detailed_caption}`.

Suppose we have edited the image, please design some questions to ask human to judge the quality of the edited image.

For example, if the task is `transform the daytime cityscape photo into a nighttime scene with lit streetlights and a full moon.`. The description of the image is `A cityscape photo with a busy street, tall buildings, and people walking around.` Then the questions can be like: 1. Is the original cityscape photo taken during the day or night?\n2.Are there streetlights visible in the original image?\n3.Is the moon present in the photo?

The questions should be concrete and clear and do not include hallucination. **Yes or No questions are preferred**. Reduce overlap between the questions. Do not include questions about image size/resolution. The questions should be composed of local and global editing effects. You need to ensure that parts unrelated to the editing requirements remain unchanged.\nThe number of questions should be less than five. Each question should be in a single line. The final response should be concise and clear.

Now give me the questions."""

    def _init_prompt(self):
        self.quality_prompt = """Given the image path {image_path} tell me the aesthetic score of the image.
        Your response should be concise and clear and do not include hallucination."""
        self.effectiveness_prompt = """Given the edited image path `{image_path}` and user input `{input_text}`, describe the edited image in detail. Tell us what is effectively changed that matches the requirements.
        Your response should be concise and clear and do not include hallucination."""
        self.unchanged_prompt = """Given the edited image path `{image_path}` and user input `{input_text}`, describe the edited image in detail and tell us what is unchanged.
        Your response should be concise and clear and do not include hallucination."""

        self.image_difference_prompt = """Given the source image path `{source_image_path}` and the edited image path `{edited_image_path}`, describe the difference between them.  You cannot use a non-existent image path on your own.
        Your response should be concise and clear and do not include hallucination."""
        self.image_common_point_prompt = """Given the source image path `{source_image_path}` and the edited image path `{edited_image_path}`, describe the common points between them.  You cannot use a non-existent image path on your own.
        Your response should be concise and clear and do not include hallucination."""

    def _design_questions(self, detailed_caption="", input_text=""):
        prompt = self.prompt.format(
            detailed_caption=detailed_caption,
            input_text=input_text,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response = call_openai_completion(
            engine=self.llm_engine, messages=messages)
        response_message = response["choices"][0]["message"]
        response_message = response_message.to_dict()["content"]

        return response_message

    def _answer_questions(
            self,
            input_image,
            input_text,
            output_image,
            questions_to_answer):
        question_answer_list = []
        for line in questions_to_answer.split("\n"):
            line = line.strip()
            if len(line) > 0 and line[0].isdigit() and line[1] == ".":
                _question = line[2:].strip()
                _new_question = f"The input image is `{input_image}`. The edited image is `{output_image}`. Answer question about the edited image: {_question}."
                # get answer
                _answer, _, _ = tool_function_callv1(
                    load_dict=self.toolset, task=_new_question, llm_engine=self.llm_engine)

                # if is float number
                if isinstance(_answer, float):
                    _answer = f"The aesthetic score is {_answer:.2f}."

                question_answer_list.append((_question, _answer))
        return question_answer_list

    def __call__(
            self,
            input_image,
            input_text,
            output_image,
            questions_to_answer,
            log_func=print):
        # question_quality = self.quality_prompt.format(image_path=output_image)
        question_answers = self._answer_questions(
            input_image, input_text, output_image, questions_to_answer)

        # import pdb; pdb.set_trace()
        self.answers = "\n".join(
            [
                f"The question is: {question}\nThe answer is: {answer}" for question,
                answer in question_answers])

        stdout = Fore.CYAN + \
            f"\n---------------------------------------------\n" + \
            f"Question and Answers: \n{self.answers}\n" + \
            f"---------------------------------------------\n" + \
            Fore.RESET
        log_func(stdout)

        self.quality_feedback = ""
        self.effectiveness_feedback = self.answers
        self.unchanged_feedback = ""

        return self.quality_feedback, self.effectiveness_feedback, self.unchanged_feedback


class Summarizer(BaseAgent):
    # summarize and interctive with human
    def __init__(self, toolset, llm_engine, role="Summarizer"):
        super().__init__(toolset, llm_engine, role=role)
        self.summarize_prompt = """Given comprehensive feedbacks \n```\n{feedback}\n```\n from human, summarize them and generate overall feedback for further improvement, including local and global editing effects, and overall quality. Numerical quality metrics should be included in the response. Your response should be short, concrete, and clear. Do not include hallucination."""
        self.interactive_prompt = ""

    def summarize(self, feedback_list, log_func=print):

        if isinstance(feedback_list, str):
            feedback_list = [feedback_list]

        summarize_prompt = self.summarize_prompt.format(
            feedback=". ".join(feedback_list))

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": summarize_prompt},
        ]

        response = call_openai_completion(
            engine=self.llm_engine, messages=messages)
        response_message = response["choices"][0]["message"]
        response_message = response_message.to_dict()["content"]

        stdout = Fore.CYAN + \
            f"\n---------------------------------------------\n" + \
            f"Summarize: \n{response_message}\n" + \
            f"\nNum of tokens: {response['usage']['prompt_tokens']} + {response['usage']['completion_tokens']} = {response['usage']['total_tokens']}" + \
            f"\n---------------------------------------------\n" + \
            Fore.RESET
        log_func(stdout)

        return response_message

    def interct_with_human(self):
        # chat with human to get feedback, direct employ chatgpt is okay.
        input_text = input(
            "Please input your feedback: (If you think the feedback is good enough, you can respond by 'None')\n)")
        return input_text.strip()


class DiscriminatorAgent():
    # The discriminator agent takes input of input_image, input_text, result,
    # interactive with human and generate a feedback.

    def __init__(self, agent_list):

        for agent in agent_list:
            if agent["role"] == "Discriminator":
                self.discriminator = Discriminator(
                    toolset=agent["toolset"],
                    llm_engine=agent["llm_engine"],
                    role=agent["role"])

            if agent["role"] == "Summarizer":
                self.summarizer = Summarizer(
                    toolset=agent["toolset"],
                    llm_engine=agent["llm_engine"],
                    role=agent["role"])

        self.feedback = []
        self.agend_id = 0

    def set_agent_id(self, agent_id):
        self.agend_id = agent_id

    def __call__(
            self,
            input_text,
            input_image,
            result,
            detailed_caption="",
            interct_with_human=False,
            log_func=print):
        _questions_to_answer = self.discriminator._design_questions(
            detailed_caption=detailed_caption, input_text=input_text)

        self.quality_feedback, self.effectiveness_feedback, self.unchanged_feedback = \
            self.discriminator(input_image, input_text, result,
                               _questions_to_answer, log_func=log_func)

        if interct_with_human:
            while True:
                try:
                    effective_feedback = input(
                        Fore.GREEN +
                        f"Please input your feedback: (If you think the feedback is good enough, you can respond by 'None')\n" +
                        f"Generated Effectiveness Feedback: " +
                        Fore.RESET +
                        f"{self.effectiveness_feedback}\n")
                    break
                except KeyboardInterrupt:
                    continue

            if effective_feedback.strip() != "None":
                self.effectiveness_feedback = effective_feedback.strip()

        summary = self.summarizer.summarize(
            [self.quality_feedback,
             self.effectiveness_feedback,
             self.unchanged_feedback],
            log_func=log_func)

        if interct_with_human:
            while True:
                try:
                    user_summary = input(
                        Fore.GREEN +
                        f"Please input your feedback: (If you think the feedback is good enough, you can respond by 'None')\n" +
                        f"Generated Summary: " +
                        Fore.RESET +
                        f"{summary}\n")
                    break
                except KeyboardInterrupt:
                    continue

            if user_summary.strip() != "None":
                summary = user_summary.strip()
        self.feedback.append(summary)

        return self.feedback[-1]


def judge_terminal_criterion(
        input_image,
        input_instruction,
        caption,
        imgs,
        feedbacks,
        llm_engine="gpt-4"):
    prompt = f"I want to edit the input image `{input_image}` to meet editing request `{input_instruction}`. "
    prompt += f"The detailed description of the input image is `{caption}`. \n"
    prompt += f"We have some edited images and feedbacks on the editing effects as follows. \n"
    prompt += "-------------------------------------------------\n"
    for img, feedback in zip(imgs, feedbacks):
        prompt += f"Edited image: `{img[-1]}`. Feedback on the edited image: `{feedback}`. \n\n"
    prompt += "-------------------------------------------------\n"
    prompt += f"Please judge whether the edited images meet the editing request `{input_instruction}`. "
    prompt += f"You should pick the best edited image and respond with the image path. For example, if you think x/y.png is the best, you should respond by 'x/y.png'."
    prompt += f"If you think the picked image is good enough, you can respond by adding `best @`. For example, if you think x/y.png is the best, you can respond by 'best @ x/y.png'. Your response should be short and clear. The image path in response must exist."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = call_openai_completion(engine=llm_engine, messages=messages)
    response_message = response["choices"][0]["message"]
    response_message = response_message.to_dict()["content"]

    for img, feedback in zip(imgs, feedbacks):
        if img[-1] in response_message:

            if "best @" in response_message.strip():
                # return response_message.strip().split("best @")[-1].strip(),
                # 1
                return img[-1], feedback, 1
            else:
                return img[-1], feedback, 0

    image_path = imgs[-1][-1]  # chosen the last one, here is randomness

    return image_path, "", 0
