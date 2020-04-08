

"""
Author: Jumpei.Takubo

This code is totally referred to the original tutorial.
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__doc__ == """This file is intended for learning LSTM and apply it to DQN"""


torch.manual_seed(1)



class LSTM_cell:
    pass

def main():
    lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
    inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # initialize the hidden state.
    hidden = (torch.randn(1, 1, 3),
            torch.randn(1, 1, 3))
    import pdb; pdb.set_trace()
    for i in inputs:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, hidden = lstm(i.view(1, 1, -1), hidden)
        print(out)
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)

if __name__ == "__main__":
    main()
    pass
    

