from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

from .stag_hunt import StagHunt
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

from .maze import Maze
REGISTRY["maze"] = partial(env_fn, env=Maze)

from .stag_maze import StagMaze
REGISTRY["stag_maze"] = partial(env_fn, env=StagMaze)

from .stag_maze_p import StagMaze
REGISTRY["stag_maze_p"] = partial(env_fn, env=StagMaze)

from .stag_maze_p_mst import StagMaze
REGISTRY["stag_maze_p_mst"] = partial(env_fn, env=StagMaze)