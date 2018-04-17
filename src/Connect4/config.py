import logging


class Config:

    def __init__(self, is_multithread=True):
        self.is_multithread = is_multithread

        # MISC
        self.Logger_Level = logging.INFO
        # Logger_Format = '%(asctime)s %(levelname)s %(message)s'
        self.Logger_Format = '%(asctime)s %(message)s'
        # Logger_Format = '%(message)s'

        # game environment parameters
        self.Width = 7
        self.Height = 6

        # network parameters
        self.Network_Metadata = [{'filters': 42, 'kernel_size': (4, 4)}, {'filters': 42, 'kernel_size': (4, 4)},
                            {'filters': 42, 'kernel_size': (4, 4)}, {'filters': 42, 'kernel_size': (4, 4)},
                            {'filters': 42, 'kernel_size': (4, 4)},
                            {'filters': 42, 'kernel_size': (4, 4)}]
        self.Reg_Const = 0.01
        self.Learning_Rate = 0.00025
        self.Root_Path = 'C:\TensorFlow\Workspace2'

        # monte carlo tree search parameters
        self.Cpuct = 1
        self.Temperature = 0.01 # this value should be in the range of [0, 1], the lower the value, the certainer the probability
        self.Dir_Epsilon = 0.25
        self.Dir_Alpha = 0.03

        # training parameters
        if self.is_multithread:
            self.Episode_Num = 0
        else:
            self.Episode_Num = 100 # not used by multithreading
        self.MCTS_Num = 100
        self.Memory_Size = 20000
        self.Iteration_Num = 500
        self.Sample_Size = 64
        if self.is_multithread:
            self.Epochs_Num = 2
        else:
            self.Epochs_Num = 10
        if self.is_multithread:
            self.Batch_Size = 16
        else:
            self.Batch_Size = 16
        self.Compete_Game_Num = 30
        self.Best_Network_Threshold = 0.6
        self.Validation_Split = 0.1

        # multi-threading parameters
        self.Fit_Interval = 3
        self.Comparison_Interval = 5
        self.Comparison_Long_Wait = 3600
        self.Min_Memory_Size_Before_Fit = int(self.Memory_Size * 0.4) # must greater than Sample_Size
        self.New_Best_Network_Memory_Clean_Rate = 1

        # Test parameters
        self.Test_MCTS_Num = 100
