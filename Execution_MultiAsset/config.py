
class Config:
    def __init__(
            self,
            batch_size: int = 64,
            valid_size: int = 256,
            num_iterations: int = 4000,
            logging_frequency: int = 100,
            verbose: bool = True
    ):
        '''
        :param batch_size: batch size
        :param valid_size: validation sample size
        :param num_iterations: # of training iter
        :param logging_frequency: # logging training info in every __ iters
        :param verbose:
        '''
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_iterations = num_iterations
        self.logging_frequency = logging_frequency
        self.verbose = verbose


class MultiAssetExecConfig(Config):
    def __init__(
            self,
            d_asset: int = 3,
            total_time: float = 1/250,
            num_time_interval: int = 100,
            num_iterations: int = 4000,
            num_hidden_layers: int = 3,
            num_neuron: int = 32,
            batch_size: int = 64,
            valid_size: int = 256,
            logging_frequency: int = 100,
            verbose: bool = True
    ):
        '''
        :param d_asset: # of assets to exeucte
        :param total_time: # physical time in years
        :param num_time_interval: # of discrete time inervals
        :param num_iterations: # of training iter
        :param num_hidden_layers: # of hideen layers in NN
        :param num_neuron: # of neurons in each hideen layer
        :param batch_size:
        :param valid_size:
        :param logging_frequency:
        :param verbose:
        '''
        super().__init__(batch_size, valid_size, num_iterations, logging_frequency, verbose)
        self.d_asset = d_asset
        self.dim = 2 * self.d_asset + 1  # 2d+1, input to NN
        self.total_time = total_time   # in years
        self.num_time_interval = num_time_interval
        self.num_iterations = num_iterations
        self.num_neurons = [self.dim] + [num_neuron for _ in range(num_hidden_layers)] + [self.d_asset]

def get_config(
        name: str,
        d_asset: int = 3,
        total_time: float = 1/250,
        num_time_interval: int = 100,
        num_iterations: int = 4000,
        num_hidden_layers: int = 3,
        num_neuron: int = 32,
        batch_size: int = 64,
        valid_size: int = 256,
        logging_frequency: int = 100,
        verbose: bool = True
):
    try:
        return globals()[name + 'Config'](
            d_asset=d_asset,
            total_time=total_time,
            num_time_interval=num_time_interval,
            num_iterations=num_iterations,
            num_hidden_layers=num_hidden_layers,
            num_neuron=num_neuron,
            batch_size=batch_size,
            valid_size=valid_size,
            logging_frequency=logging_frequency,
            verbose=verbose
        )
    except KeyError:
        raise KeyError(f"Config {name} not found.")
