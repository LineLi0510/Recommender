from typing import List

from domain.entities.train_data import TrainData
from domain.runner.runner_commands.runner_command import RunnerCommand


class RunnerInvoker:
    @staticmethod
    def run_commands(runner_commands: List[RunnerCommand], data_set: TrainData):
        for command in runner_commands:
            command.execute(data_set=data_set)