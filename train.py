from Model.MCTAgent import MCTTrainer
import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    trainer = MCTTrainer(device=device,
                         use_checkpoint=True,
                         board_size=6,
                         n_inrow=4)
    trainer.train()