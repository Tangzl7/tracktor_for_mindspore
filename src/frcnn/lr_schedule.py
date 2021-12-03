# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""lr generator for fasterrcnn"""
import math

def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate

def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate

def dynamic_lr(config, steps_per_epoch):
    """dynamic learning rate generator"""
    base_lr = config.base_lr
    total_steps = steps_per_epoch * (config.epoch_size + 1)
    lr = []
    for i in range(total_steps):
        if i < min(1000, steps_per_epoch):
            alpha = float(i) / config.warmup_step
            lr.append(base_lr * (config.warmup_ratio * (1 - alpha) + alpha))
        else:
            lr.append((0.1 ** int(i / (steps_per_epoch * 20))) * base_lr)
    return lr

def dynamic_lr_1(config, steps_per_epoch):
    """dynamic learning rate generator"""
    base_lr = config.base_lr
    total_steps = steps_per_epoch * (config.epoch_size + 1)
    lr = []
    for i in range(total_steps):
        lr.append((0.1 ** int(i / (steps_per_epoch * 10))) * base_lr)

    return lr