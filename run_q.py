
from q_seq2seq import * 
import matplotlib.pyplot as plt

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    for (l,p) in points.items():
        plt.plot(p, label=l)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    training, testing = loadData()
    agent_params = {}
    trainer_params = {}
    critic_params = {}

    trainer_params['n_iter'] = int(5e6)
    trainer_params['train_n_iter'] = 1
    trainer_params['train_batch_size'] = 32

    agent_params['batch_size'] = trainer_params['train_batch_size']
    agent_params['learning_starts'] = int(1e4)
    agent_params['target_update_freq'] = int(3e4)
    agent_params['learning_freq'] = 4
    agent_params['optimizer_spec'] = OptimizerSpec(constructor=optim.Adam, optim_kwargs=dict(lr=1),learning_rate_schedule=lambda epoch: 1e-3)
    agent_params['exploration_schedule'] = PiecewiseSchedule([(0, 1), (5e5 * 0.1, 0.02),], outside_value=0.02)
    agent_params['train'] = training
    agent_params['test'] = testing
    agent_params['replay_buffer_size'] = int(5e5)
    agent_params['frame_history_len'] = 1
    agent_params['critic_params'] = critic_params
    
    
    critic_params['grad_norm_clipping'] = 10

    trainer = Trainer(agent_params, trainer_params)
    trainer.run()
    showPlot({"Reward":trainer.reward})
