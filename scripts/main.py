import os
import sys

from functools import partial
from colorama import Fore
import csv
import click
import shutil
from datetime import datetime
import pytz
import requests
from io import BytesIO
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.agentsv2 import (
    judge_terminal_criterion,
    GeneratorAgent,
    DiscriminatorAgent,
    stream_logging,
)
from utils import seed_everything


@click.group()
def cli():
    r"""Editing Agents CLI"""


def agent_api(image_path,
              instruction,
              num_agents=2,
              tag="debug",
              num_rounds=5,
              tool_list="InstructDiffusion,Resize",):

    from PIL import Image
    # set random seed
    seed_everything(42)

    # llm_engine = llm_engine1 = "gpt-4" # "gpt-35-turbo"
    toolset = {tool_name: 'cuda:0' for tool_name in tool_list.split(",")}
    gen_agent_list = [
        {
            "role": "Planner",
            "llm_engine": "gpt-4",
            "toolset": toolset
        },
        {
            "role": "Tool-executor",
            "llm_engine": "gpt-4-turbo",
            "toolset": toolset
        },
        {
            "role": "Reflector",
            "llm_engine": "gpt-4",
            "toolset": toolset
        }
    ]

    dis_agent_list = [{"role": "Discriminator",
                       "llm_engine": "gpt-4-turbo",
                       "toolset": {"AestheticScore": 'cuda:0',
                                     "LLaVA": 'cuda:0',
                                     "ImageDifferenceLLaVA": 'cuda:0'}},
                      {"role": "Summarizer",
                       "llm_engine": "gpt-4-turbo",
                       "toolset": {"AestheticScore": 'cuda:0',
                                   "LLaVA": 'cuda:0'}}]

    # csv_path = "examples.csv"
    img_caption_instruction_list = [
        (image_path, instruction)
    ]

    for _img, _ins in img_caption_instruction_list:
        os.makedirs("image", exist_ok=True)

        input_text = _ins
        input_image = _img

        if input_image.startswith("http"):
            response = requests.get(input_image)
            input_image = Image.open(BytesIO(response.content))

            input_image.save("image/test.png")
            input_image = "image/test.png"

        else:
            assert os.path.exists(input_image)
            input_image = Image.open(input_image)
            input_image.save("image/test.png")
            input_image = "image/test.png"

        # total_iters = 5
        feedback_from_last_turn = None
        interact_with_human = False

        gen_agents = [GeneratorAgent(gen_agent_list)
                      for _ in range(num_agents)]
        dis_agents = [DiscriminatorAgent(dis_agent_list)
                      for _ in range(num_agents)]

        logger_files = [f"agent{i + 1}.log" for i in range(num_agents)]
        time_string = datetime.now(
            pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.environ.get(
            "LOGDIR", f"logs/{tag}-{os.path.basename(_img).split('.')[0]}-{time_string}")
        os.makedirs(log_dir, exist_ok=True)
        logger_files = [os.path.join(log_dir, logger_file)
                        for logger_file in logger_files]

        # set agent id
        for i in range(len(gen_agents)):
            gen_agents[i].set_agent_id(i)
            dis_agents[i].set_agent_id(i)

        all_results = []

        try:
            # if True:
            for _round in range(num_rounds):

                for i in range(len(gen_agents)):
                    gen_agents[i].set_save_prefix(f"image/agent{i + 1}")

                    _log_func = partial(stream_logging, logger_files[i])
                    _log_func(
                        Fore.CYAN +
                        f"\n++++++++++++++++++++++++++++++++++++++++++++++\n" +
                        f"Round - {_round}\n" +
                        f"++++++++++++++++++++++++++++++++++++++++++++++\n" +
                        Fore.RESET
                    )

                    chat_string_gen, tool_names, tool_inputs = gen_agents[i](
                        input_text=input_text,
                        input_image=input_image,
                        feedback=all_results,
                        log_func=_log_func)

                    feedback_from_last_turn = dis_agents[i](
                        input_text=input_text,
                        input_image=input_image,
                        result=chat_string_gen,
                        interct_with_human=interact_with_human,
                        log_func=_log_func,
                    )

                all_results = dict(
                    feedbacks=[
                        dis_agent.feedback for dis_agent in dis_agents], planning_results=[
                        gen_agent.planning_results for gen_agent in gen_agents], imgs=[
                        gen_agent.temp_images for gen_agent in gen_agents], )

                best_result, feedback, to_terminal = judge_terminal_criterion(
                    input_image, input_text,
                    caption=gen_agents[0].input_image_discription,
                    imgs=all_results["imgs"],
                    feedbacks=all_results["feedbacks"],
                    llm_engine="gpt-35-turbo")

                gen_agents[0].save_best_image(best_result, feedback)
                gen_agents[1].save_best_image(best_result, feedback)

                stdout = Fore.GREEN + \
                    f"Best result: {best_result}" + Fore.RESET
                stream_logging(logger_files[0], stdout, verbose=True)
                stream_logging(logger_files[1], stdout, verbose=False)

                # load best image
                best_image = None
                try:
                    best_image = Image.open(best_result).convert("RGB")
                except BaseException:
                    best_image = None
                    for _img in gen_agents[0].best_images[::-1]:
                        if os.path.exists(_img):
                            best_image = Image.open(_img).convert("RGB")
                            break

                if to_terminal > 0:
                    stdout = Fore.GREEN + f"Terminal criterion is met." + Fore.RESET
                    stream_logging(logger_files[0], stdout, verbose=True)
                    stream_logging(logger_files[1], stdout, verbose=False)

                    # cp image folder to log dir
                    shutil.copytree("image", os.path.join(log_dir, "image"))
                    shutil.rmtree("image")
                    # exit(0)
                    break

            if os.path.exists("image"):
                shutil.copytree("image", os.path.join(log_dir, "image"))
                shutil.rmtree("image")

            return best_image

        except BaseException:

            if os.path.exists("image"):
                shutil.copytree("image", os.path.join(log_dir, "image"))
                shutil.rmtree("image")

            continue


# warp api
@cli.command()
@click.option("--image-path", default="outputs/OIP.jpg", help="image path")
@click.option("--instruction", default="replace the leopard with a cute cat", help="instruction")
@click.option("--num-agents", default=2, help="Number of agents")
@click.option("--tag", default="debug", help="tag")
@click.option("--num-rounds", default=5, help="num rounds")
@click.option("--tool-list",
                default="InstructDiffusion,Resize",
                help="tool list")
def run(image_path="outputs/OIP.jpg",
        instruction="replace the leopard with a cute cat",
        num_agents=2,
        tag="debug",
        num_rounds=5,
        tool_list="InstructDiffusion,Resize",):

    _ = agent_api(image_path=image_path,
                  instruction=instruction,
                  num_agents=num_agents,
                  tag=tag,
                  num_rounds=num_rounds,
                  tool_list=tool_list)



@cli.command()
@click.option("--num-agents", default=2, help="Number of agents")
@click.option("--split-id", default=0, help="split-training-data")
@click.option("--splits", default=1, help="total splits")
@click.option("--debug", default=False, help="debug mode")
@click.option("--tag", default="debug", help="tag")
@click.option("--csv-path", default="examples.csv", help="csv path")
@click.option("--tool-list",
              default="InstructDiffusion,Resize",
              help="tool list")
@click.option("--max-items", default=100, help="max items")
@click.option("--num-rounds", default=1, help="num rounds")
def parallel_runningv1(num_agents=2,
                       split_id=0,
                       splits=1,
                       debug=False,
                       tag="debug",
                       csv_path="examples.csv",
                       tool_list="InstructDiffusion,Resize",
                       max_items=100,
                       num_rounds=1):

    from PIL import Image
    # set random seed
    seed_everything(42)

    # llm_engine = llm_engine1 = "gpt-4" # "gpt-35-turbo"
    toolset = {tool_name: 'cuda:0' for tool_name in tool_list.split(",")}
    gen_agent_list = [
        {
            "role": "Planner",
            "llm_engine": "gpt-4",
            "toolset": toolset
        },
        {
            "role": "Tool-executor",
            "llm_engine": "gpt-4-turbo",
            "toolset": toolset
        },
        {
            "role": "Reflector",
            "llm_engine": "gpt-4",
            "toolset": toolset
        }
    ]

    dis_agent_list = [{"role": "Discriminator",
                       "llm_engine": "gpt-4-turbo",
                       "toolset": {"AestheticScore": 'cuda:0',
                                     "LLaVA": 'cuda:0',
                                     "ImageDifferenceLLaVA": 'cuda:0'}},
                      {"role": "Summarizer",
                       "llm_engine": "gpt-4-turbo",
                       "toolset": {"AestheticScore": 'cuda:0',
                                   "LLaVA": 'cuda:0'}}]

    img_instruction_list = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_instruction_list.append((row['image'], row['instruction']))

    img_instruction_list = img_instruction_list[:max_items]
    img_instruction_list = img_instruction_list[split_id::splits]

    # img_instruction_list = [
    #     (
    #         "outputs/OIP.jpg",
    #         "replace the leopard with a cute cat"
    #     )
    # ]

    for _img, _ins in img_instruction_list:
        os.makedirs("image", exist_ok=True)

        input_text = _ins
        input_image = _img
        input_image = os.path.join(
            "/home/t-thang/cache/benchmark", input_image)

        if input_image.startswith("http"):
            response = requests.get(input_image)
            input_image = Image.open(BytesIO(response.content))

            input_image.save("image/test.png")
            input_image = "image/test.png"

        else:
            assert os.path.exists(input_image)
            input_image = Image.open(input_image)
            input_image.save("image/test.png")
            input_image = "image/test.png"

        feedback_from_last_turn = None
        interact_with_human = False

        gen_agents = [GeneratorAgent(gen_agent_list)
                      for _ in range(num_agents)]
        dis_agents = [DiscriminatorAgent(dis_agent_list)
                      for _ in range(num_agents)]

        logger_files = [f"agent{i + 1}.log" for i in range(num_agents)]
        time_string = datetime.now(
            pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.environ.get(
            "LOGDIR", f"logs/{tag}-{_img.replace('/', '-').split('.')[0]}-{time_string}")
        os.makedirs(log_dir, exist_ok=True)
        logger_files = [os.path.join(log_dir, logger_file)
                        for logger_file in logger_files]

        # set agent id
        for i in range(len(gen_agents)):
            gen_agents[i].set_agent_id(i)
            dis_agents[i].set_agent_id(i)

        all_results = []

        try:
            # if True:
            for _round in range(num_rounds):

                for i in range(len(gen_agents)):
                    gen_agents[i].set_save_prefix(f"image/agent{i + 1}")

                    _log_func = partial(stream_logging, logger_files[i])
                    _log_func(
                        Fore.CYAN +
                        f"\n++++++++++++++++++++++++++++++++++++++++++++++\n" +
                        f"Round - {_round}\n" +
                        f"Input image: {input_image}\n" +
                        f"Instruction: {_ins}\n" +
                        f"++++++++++++++++++++++++++++++++++++++++++++++\n" +
                        Fore.RESET
                    )

                    chat_string_gen, tool_names, tool_inputs = gen_agents[i](
                        input_text=input_text,
                        input_image=input_image,
                        feedback=all_results,
                        log_func=_log_func)

                    feedback_from_last_turn = dis_agents[i](
                        input_text=input_text,
                        input_image=input_image,
                        result=chat_string_gen,
                        interct_with_human=interact_with_human,
                        log_func=_log_func,
                    )

                all_results = dict(
                    feedbacks=[
                        dis_agent.feedback for dis_agent in dis_agents], planning_results=[
                        gen_agent.planning_results for gen_agent in gen_agents], imgs=[
                        gen_agent.temp_images for gen_agent in gen_agents], )

                best_result, feedback, to_terminal = judge_terminal_criterion(
                    input_image, input_text,
                    caption=gen_agents[0].input_image_discription,
                    imgs=all_results["imgs"],
                    feedbacks=all_results["feedbacks"],
                    llm_engine="gpt-35-turbo")

                gen_agents[0].save_best_image(best_result, feedback)
                gen_agents[1].save_best_image(best_result, feedback)

                stdout = Fore.GREEN + \
                    f"Best result: {best_result}" + Fore.RESET
                stream_logging(logger_files[0], stdout, verbose=True)
                stream_logging(logger_files[1], stdout, verbose=False)

                # load best image
                best_image = None
                try:
                    best_image = Image.open(best_result).convert("RGB")
                except BaseException:
                    best_image = None
                    for _img in gen_agents[0].best_images[::-1]:
                        if os.path.exists(_img):
                            best_image = Image.open(_img).convert("RGB")
                            break

                if to_terminal > 0:
                    stdout = Fore.GREEN + f"Terminal criterion is met." + Fore.RESET
                    stream_logging(logger_files[0], stdout, verbose=True)
                    stream_logging(logger_files[1], stdout, verbose=False)

                    # cp image folder to log dir
                    shutil.copytree("image", os.path.join(log_dir, "image"))
                    shutil.rmtree("image")
                    # exit(0)
                    break

            if os.path.exists("image"):
                shutil.copytree("image", os.path.join(log_dir, "image"))
                shutil.rmtree("image")

            # return best_image

        except BaseException:

            if os.path.exists("image"):
                shutil.copytree("image", os.path.join(log_dir, "image"))
                shutil.rmtree("image")

            continue


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
