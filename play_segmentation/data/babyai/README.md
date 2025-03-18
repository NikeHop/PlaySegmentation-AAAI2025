# 🗂️ BabyAI Dataset

Generate BabyAI trajectories (play data & annotated dataset).

**Note:** Activate the play_segmentation conda environment to run the code.

## Data Generation

To generate a dataset of trajectories containing multiple instructions run:

```
python generate.py --config ./config/gen_data.yaml
```

This process generates two datasets: one with unsegmented trajectories and another with segmented trajectories containing their subparts. The data is stored as a dictionary of lists, where each list entry corresponds to a trajectory, saving relevant data for each.


**Segmented Trajectories**
- data["images"][i]: [blosc.array](https://www.blosc.org/python-blosc/reference.html). Packed numpy.array, containing RGB environment observations (Tx64x64x3) of the trajectory.
- data["states"][i]: Packed numpy.array, containing environment states (Tx8x8x3) of the trajectory.
- data["instructions"][i]: instruction (str)
- data["instruction_types"][i]: instruction type (dict)
- data["actions"][i]: list of actions the agent is taking at timestep t (0-2).
- data["directions"][i]: list of directions the agent is pointing to at timestep t (0-3).
- data["rewards"][i]: list of rewards the agent receives at timestep t (\[0-1\]).

**Unsegmented Trajectories**
- data["images"][i]: list of blosc.arrays
- data["states"][i]: list of blosc.arrays
- data["instructions"][i]: list of instructions
- data["instruction_types"][i]: list of instruction-types
- data["actions"][i]: list of list of actions the agent is taking
- data["directions"][i]: list of list of directions the agent is pointing to.
- data["rewards"][i]: list of list of rewards.


## Data Setup paper
To recreate the data setup in the paper, run:

`bash ./scripts/create_datasets.sh`.
