"""Subclass the BabyAI Environment to generate an instruction that can be satisfied with the current environment state"""

from minigrid.envs.babyai.goto import GoToLocal
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc

# Object Types that are relevant: "Key", "Ball", "Box", "Door", "Goal"
# Colors "red", "green", "blue", "purple", "yellow", "grey
KEY_ACTIONS = ["pickup", "putnext", "goto"]
BALL_ACTIONS = ["pickup", "putnext", "goto"]
BOX_ACTIONS = ["pickup", "putnext", "goto"]
DOOR_ACTIONS = ["open", "goto"]

OBJ2ACTION_SPACE = {
    "key": KEY_ACTIONS,
    "ball": BALL_ACTIONS,
    "box": BOX_ACTIONS,
    "door": DOOR_ACTIONS,
}


class GoToLocalExtended(GoToLocal):
    """An extended version of the GoToLocal class."""

    def sample_new_instruction(self):
        """
        Samples a new instruction for the environment.

        This method selects a random object from the environment and generates a new instruction
        to go to that object. The generated instruction is stored in the `instrs` attribute and
        the corresponding mission string is stored in the `mission` attribute.

        Returns:
            None
        """
        objs = self._get_objs()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
        self.mission = self.instrs.surface(self)
        self.instrs.reset_verifier(self)

    def _get_objs(self):
        """
        Get the objects in the room.

        Returns:
            list: A list of objects in the room.
        """
        room = self.room_grid[0][0]
        return room.objs
