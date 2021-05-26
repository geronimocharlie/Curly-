# Curly-

Reimplementation of curriculum designt with the  paired algorithm([see the paper](https://arxiv.org/pdf/2012.02096.pdf)) with gridworlds. Based on the Really (reinforcement learinng using ray) framework.

**main.py** - main loop


**train.py** - contains ppo training loop (agents and adversarial) + training utilities


**models.py** - contains models (Gonist for protagonist and antagonist & Adversarial)


**wrapper.py** - class for wrapping adversarial output into environment parameters


**gridworld_global_multi0.py** - play around copy of gridworld allowing multiple blocks
