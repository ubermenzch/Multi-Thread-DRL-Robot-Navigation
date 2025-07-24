from ros_nodes import (
    UnitreeCmdVelPublisher,
)
from SAC.SAC import SAC
class Agent:
    def __init__(self):
        self.cmd_vel_publisher=UnitreeCmdVelPublisher()
        self.gps=None
        self
def main(args=None):
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 25  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    if torch.cuda.is_available():
        print("GPU CUDA Ready")
    else:
        print("CPU CUDA Ready")
    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        is_unitree_dog=True,
    )  # instantiate a model
    state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get state a state representation from returned data from the environment

def calc_target_agent_angle():

if __name__ == "__main__":
    main()