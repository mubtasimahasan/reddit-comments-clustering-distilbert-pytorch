import yaml
import os
import torch


class CFG:
    def __init__(self):
        config_path = "./configs/config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.seed = config["seed"]
        self.model_name = config["model_name"]
        self.num_labels = config["num_labels"]
        self.max_length = config["max_length"]
        self.debug = config["debug"]
        self.n_epochs = config["n_epochs"]
        self.max_steps = config["max_steps"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.warmup_steps = config["warmup_steps"]
        self.weight_decay = config["weight_decay"]
        self.logging_steps = config["logging_steps"]
        self.save_steps = config["save_steps"]
        self.evaluation_strategy = config["evaluation_strategy"]
        self.load_best_model_at_end = config["load_best_model_at_end"]
        self.device = config["device"]
        self.report_to = config["report_to"]

        if self.debug:
            self.max_steps = 200
        
        if not torch.cuda.is_available():
            self.device = "cpu"

CFG = CFG()
