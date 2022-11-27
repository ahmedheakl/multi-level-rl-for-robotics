[![Main Tests](https://github.com/ahmedheakl/Mutli_Level_RL_Robotics/workflows/test/badge.svg)](https://github.com/ahmedheakl/Mutli_Level_RL_Robotics/actions)

# Highrl: Multi-level Reinforcement Learning for Robotics Navigation

Highrl is a library for training robots using RL under the scheme of multi-level RL. The library has numerous features from generating random environment, training agents to generate curriculum learning schemes, or train robots in pre-defined environments.

The robot can explore a synthetic [gym](https://www.gymlibrary.dev/) environment using lidar vision (either in flat or rings format). The robot is trained to reach the goal with obstacles in between to hinder its movement and simulate the real-world environment. A teacher (an RL agent) will be trained to synthesize the perfect curriculum learning for the robot, so that the robot will solve maps with certain difficulties in minimal time.  

The robot model is implemented with a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) for the feature extractor and an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) for both the value and policy networks. The teacher model is implemented with an LSTM network for the feature extractor and an MLP for value/policy network. The robot model is fed with the lidar data and outputs the velocity ```(vx, vy)``` of the robot. The teacher model is fed with data of the last session for the robot that the teacher is training, and outputs the configurations for the next environment to train the robot. At each step of the teacher, a new robot will be generated with probability of 10% and will be trained for a fixed number of steps. You can find the models in ```src/highrl/policy```. 


## Installation

Please note that the library is **only tested on Linux distributions**. If you want to install it on Windows, you can use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install higrl library.

```bash
pip install highrl
```

## Usage

```bash
highrl -h # get available arguments
```
### Configurations
**--robot-config**: path of configuration file of robot environment (relative path). 

**--teacher-config**: path of configuration file of teacher environment (relative path). 

**--mode**: choose train or test mode.

**--env-mode**: choose whether to train the robot alone or in the presence of a teacher to generate curriculum learning. 

**--render-each**: the frequency of rendering for robot environment (integer).

**--output-path**: relative path to output results for robot mode. 

**--lidar-mode**: mode to process lidar flat=1D, rings=2D.

### Example
```bash
highrl --render-each=50 --output-dir=~/Desktop
```

<p align="center">
  <img height="300" width="500" src="https://github.com/ahmedheakl/multi-level-rl-for-robotics/blob/main/imgs/highrl-img.png">
</p>


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
