args = {
    "atari": {
        "num_train_steps": 7e9,
        "nenvs": 4,
        "num_runner_steps": 128,
        "gamma": 0.99,
        "lambda_": 0.95,
        "num_epochs": 5,
        "num_minibatches": 4,
        "cliprange": 0.1,
        "value_loss_coef": 0.25,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "lr": 2.5e-4,
        "optimizer_epsilon": 1e-5,
        "search_space": "impala",

        "num_learner_steps": 10,
        "num_nas_runner_steps": 3,


        "nas_lr": 1e-3,
        "nas_entropy_coef": 1e-4,
        "nas_optimizer_epsilon":1e-5,
        "nas_baseline_momentum":0.2,

        "spos_total_iters": 10,
        "spos_learning_rate": .5,
        "spos_momentum": .9,
        "spos_weight_decay": 4e-5,
    }
}