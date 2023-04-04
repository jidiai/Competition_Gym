# Competition_Gym


## Dependency

>conda create -n jidi_gym python=3.7.5

>conda activate jidi_gym

>pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

---

## How to test submission

You can locally test your submission. At Jidi platform, we evaluate your submission as same as *run_log.py*

For example, to play with Acrobot, run:

>python run_log.py --env_name 'classic_Acrobot-v1'

To play with Hopper, run:

>python run_log.py --env_name 'gym_Hopper-v2'

in which you are controlling agent 1 which is green.

### Note: To play with Hopper, please first install mujoco210: [mujoco210](https://github.com/deepmind/mujoco/releases/tag/2.1.0)

---

## Ready to submit

Random policy --> *agents/random/submission.py*