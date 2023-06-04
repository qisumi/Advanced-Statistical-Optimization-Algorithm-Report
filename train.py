from Model.MCTAgent import MCTTrainer
import torch
import numpy as np
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    trainer = MCTTrainer(device=device,
                         use_checkpoint=False,
                         board_size=3,
                         n_inrow=3)
    trainer.train()