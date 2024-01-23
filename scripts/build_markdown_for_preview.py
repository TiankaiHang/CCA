import os
import re

import click
import shutil
from tqdm import tqdm


@click.group()
def cli():
    r"""Build markdown files for preview."""


def replace_image_path(line, pattern, new_pattern):
    img_paths = re.findall(pattern, line)
    img_paths = list(set(img_paths))
    img_paths = [img_path for img_path in img_paths if (
        img_path.startswith('image/'))]
    for img_path in img_paths:
        line = line.replace(img_path, new_pattern.format(img_path))
    return line


@cli.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True))
@click.option('--log-pattern', '-p', default='TAG1', type=str, help='log pattern')
def buildall(input_dir, log_pattern="TAG1"):
    r"""Build markdown files for preview."""

    sub_dirs = os.listdir(input_dir)
    sub_dirs = [_dir for _dir in sub_dirs if log_pattern in _dir]
    pattern = r'\b[\w\.-]+(?:/[\w\.-]+)*\.(?:png|jpg|jpeg)\b'

    for sub_dir in tqdm(sub_dirs, total=len(sub_dirs)):
        _input_dir = os.path.join(input_dir, sub_dir)

        if not (os.path.exists(os.path.join(_input_dir, 'agent1.log')) and os.path.exists(os.path.join(_input_dir, 'agent2.log'))):
            # remove the directory
            shutil.rmtree(_input_dir)
            continue

        agent_1_log = os.path.join(_input_dir, 'agent1.log')
        agent_1_md = os.path.join(_input_dir, 'agent1.md')

        agent_2_log = os.path.join(_input_dir, 'agent2.log')
        agent_2_md = os.path.join(_input_dir, 'agent2.md')

        lines_for_agent_1_md = []
        with open(agent_1_log, 'r') as f:
            agent_1_lines = f.readlines()
            # parse image from
            # ```
            # Result of execution:
            # image/agent1_12.png
            # ```
            for i in range(len(agent_1_lines)):

                if i > 0 and agent_1_lines[i-1].startswith('Result of execution:'):
                    lines_for_agent_1_md.append(replace_image_path(
                        agent_1_lines[i], pattern, '![image]({})'))
                elif "Best result:" in agent_1_lines[i].strip():
                    new_line = agent_1_lines[i].strip().replace('`', ' ')
                    new_line = replace_image_path(
                        new_line, pattern, '![image]({})')
                    lines_for_agent_1_md.append('\n' + new_line + '\n')
                elif agent_1_lines[i].strip().startswith("Input image"):
                    new_line = agent_1_lines[i].strip().replace('`', ' ')
                    new_line = replace_image_path(
                        new_line, pattern, '![image]({})')
                    lines_for_agent_1_md.append('\n' + new_line + '\n')
                else:
                    lines_for_agent_1_md.append("+ " + agent_1_lines[i])

        with open(agent_1_md, 'w') as f:
            f.writelines(lines_for_agent_1_md)

        lines_for_agent_2_md = []
        with open(agent_2_log, 'r') as f:
            agent_2_lines = f.readlines()
            # parse image from
            # ```
            # Result of execution:
            # image/agent2_12.png
            # ```
            for i in range(len(agent_2_lines)):
                if i > 0 and agent_2_lines[i-1].startswith('Result of execution:'):
                    lines_for_agent_2_md.append(replace_image_path(
                        agent_2_lines[i], pattern, '![image]({})'))

                elif "Best result:" in agent_2_lines[i].strip():
                    new_line = agent_2_lines[i].strip().replace('`', '')
                    new_line = replace_image_path(
                        new_line, pattern, '![image]({})')
                    lines_for_agent_2_md.append('\n' + new_line + '\n')
                elif agent_2_lines[i].strip().startswith("Input image"):
                    new_line = agent_2_lines[i].strip().replace('`', ' ')
                    new_line = replace_image_path(
                        new_line, pattern, '![image]({})')
                    lines_for_agent_2_md.append('\n' + new_line + '\n')
                else:
                    lines_for_agent_2_md.append("+ " + agent_2_lines[i])

        with open(agent_2_md, 'w') as f:
            f.writelines(lines_for_agent_2_md)


if __name__ == '__main__':
    r"""
    python scripts/build_markdown_for_preview.py buildall -i logs/
    """
    cli()
