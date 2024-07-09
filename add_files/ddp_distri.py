# Copyright The PyTorch Lightning team.
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
from typing import Dict

import torch

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from pytorch_lightning.utilities.seed import isolate_rng
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn import Module
from pytorch_lightning.strategies.ddp import DDPStrategy
import logging
log = logging.getLogger(__name__)

class DDP_Distri_Strategy(DDPStrategy):
    """DDP2 behaves like DP in one node, but synchronization across nodes behaves like in DDP."""

    strategy_name = "ddp_distri"

    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        device_ids = self.determine_ddp_device_ids()
        print("*********************************device_ids***********************************")
        # print('device_ids:', device_ids)
        log.detail(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
        return DistributedDataParallel(module=model, **self._ddp_kwargs)
    
    def model_to_device(self):
        log.detail(f"{self.__class__.__name__}: moving model to device [{self.root_device}]...")
        print("*********************************Moving model to device***********************************")
        device1 = 'cuda:0'
        device2 = 'cuda:1'
        self.model.to(device1)
        self.model.trans_stage_model.to(device2)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
