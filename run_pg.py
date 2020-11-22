
from pg_seq2seq import * 
import matplotlib.pyplot as plt

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    for (l,p) in points.items():
        plt.plot(p, label=l)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    training, testing = loadData()
    agent_params = {}
    trainer_params = {}
    critic_params = {}

    multiplier = 0.2

    trainer_params['n_iter'] = int(5e4*multiplier)
    trainer_params['train_n_iter'] = 1
    trainer_params['train_batch_size'] = 32
    trainer_params['multiplier'] = multiplier

    trainer_params['batch_size'] = trainer_params['train_batch_size']
    trainer_params['train'] = training
    trainer_params['test'] = testing
    
    trainer = Trainer(agent_params, trainer_params)
    trainer.run()
    showPlot({"Train":trainer.reward, "Eval":trainer.eval_rewards})
