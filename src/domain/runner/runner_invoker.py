from typing import List

from domain.entities.train_data import TrainData
from domain.runner.runner_commands.runner_command import RunnerCommand


class RunnerInvoker:
    def __init__(self, runner_commands: List[RunnerCommand]):
        self._runner_commands: List[RunnerCommand] = runner_commands


    def run_commands(self, data_set: TrainData):
        for command in self._runner_commands:
            command.execute(data_set=data_set)
