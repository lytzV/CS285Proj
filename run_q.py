
from q_seq2seq import *
import matplotlib  
matplotlib.use('TkAgg')
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
    try:
        training = pd.read_csv('data/sighan50train.csv').to_numpy()
        testing = pd.read_csv('data/sighan50test.csv').to_numpy()
        print(training[0])
    except Exception as e:
        print(e)
        training, testing = loadData()
        pd.DataFrame(training).to_csv("data/sighan50train.csv", index=False)
        pd.DataFrame(testing).to_csv("data/sighan50test.csv", index=False)
        print(training[0])
    agent_params = {}
    trainer_params = {}
    critic_params = {}

    multiplier = 0.1

    trainer_params['n_iter'] = int(5e6*multiplier)
    trainer_params['train_n_iter'] = 1
    trainer_params['train_batch_size'] = 32
    trainer_params['multiplier'] = multiplier

    agent_params['batch_size'] = trainer_params['train_batch_size']
    agent_params['learning_starts'] = int(1e4*multiplier)
    agent_params['target_update_freq'] = int(3e4*multiplier)
    agent_params['learning_freq'] = 4
    agent_params['optimizer_spec'] = OptimizerSpec(constructor=optim.Adam, optim_kwargs=dict(lr=1),learning_rate_schedule=lambda epoch: 1e-3)
    agent_params['exploration_schedule'] = PiecewiseSchedule([(0, 1), (5e5 * 0.1, 0.02),], outside_value=0.02)
    agent_params['train'] = training
    agent_params['test'] = testing
    agent_params['replay_buffer_size'] = int(5e5*multiplier)
    agent_params['frame_history_len'] = 1
    agent_params['critic_params'] = critic_params
    
    
    critic_params['grad_norm_clipping'] = 10

    trainer = Trainer(agent_params, trainer_params)
    trainer.run()
    showPlot({"Train":trainer.reward})
    showPlot({"Precision":trainer.precision, "Recall":trainer.recall, "Fhalf":trainer.fhalf})
