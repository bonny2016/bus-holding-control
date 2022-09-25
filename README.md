## About The Project
Implementation of paper "Dynamic Bus Holding Control Using Spatial-Temporal Data -- A Deep Reinforcement Learning Approach"

This paper proposes a deep reinforcement learning (DRL) approach that dynamically determines the dispatching of bus services at the starting bus stop for a high-frequency bus service line. 

### About The Data
We use data from BRT (Bus Rapid Transit) Line 2 of Xiamen city, China on a random day in June 2018. It is a high-freqency bus line system which is a good candidate for real-time control. The environment is simulated based on real passenger swiping records and bus travel time information, with some pre-processing steps to simulate passenger arrival times based on corresponding boarding time, since passenger arrival times are not directly recorded in swiping records. A uniform distribution of arrivals during two consecutive buses at each stop is assumed.

### Getting Started
This project use PyTorch as machine learning framework and PTAN (https://github.com/Shmuma/ptan) as reinforcement learning base framework. Please refer to the "requirement.txt" file for my develop environemnt setup. 

## Usage
* To train a control model, execute the command:
  ```
  python train_model.py [--name architectName] [--run modelName]. 
  ```
  where --name parameter represent the DQN architecture to be used. Three options are available: fc(fully-connected) (default), att(self-attention), and cnn. 
  for example, thhe following command trains a self-attention based reinforcement learning model, and saves result models in folder ".\saves\my-att-model".
  ```
  python train_model.py --name att --run my-att-model
  ```
  
* To run a trained dynamic model or choose a static dispatching method using bus data and see metrics, execute the command:
   ```
   python run_model.py [--name modelName] [--model modelFile]
   ```
   where modelName can be either one of the dynamic models (att/fc/cnn) we trained, or one of the static models (uniform/schedule/manual): 

   For example, the following command will load a fully-connected dynamic model and run with our bus data.
   ```
   python run_model.py --name fc --model ./saves/fc-state_1_staging_0_episode_1100_/val_reward-366.800.data 
   ```
 
