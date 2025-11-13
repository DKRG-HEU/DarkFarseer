import argparse
import functools


def get_args():
    """
    Build and parse default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="AIR36", help="select from [METR-LA, PEMS-BAY, PEMS03, PEMS04, AIR36, NREL-PA, USHCN]")
    parser.add_argument("--partition", type=str, default="6/2/2")
    parser.add_argument("--virtual_node_rate", type=float, default=0.25)
    parser.add_argument("--time_window", type=int, default=24)

    parser.add_argument('--store_path', type=str, default='model_files/', help="model file storage path")

    # Optimizer param
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--clip', type=float, default=5.0, help="Gradient clip")

    # Trainer params
    parser.add_argument('--epochs', type=int, default=300, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Training epochs")
    parser.add_argument('--device', type=str, default="cuda:0", help='CUDA Visible Devices')
    parser.add_argument('--seed', type=int, default=0, help="Controlling the distribution of virtual nodes")
    parser.add_argument('--num_run', type=int, default=10, help="Number of experiments")

    # Model params
    parser.add_argument('--mu', type=float, default=0.0, help="Control the BCCs sparsity")
    parser.add_argument('--eta', type=float, default=0.01, help="Trade-off param of VCCL")
    parser.add_argument('--beta', type=float, default=0.1, help="Drop edge rate for each virtual node")
    parser.add_argument('--hidden_size', type=int, default=64, help="Hidden dimension")

    use_default = True
    if use_default:
        args, _ = parser.parse_known_args()

        dataset_args_map = {
            "PEMS03": DEFAULT_PEMS03_args,
            "PEMS04": DEFAULT_PEMS04_args,
            "METR-LA": DEFAULT_LA_args,
            "PEMS-BAY": DEFAULT_BAY_args,
            "AIR36": DEFAULT_AIR36_args,
            "NREL-PA": DEFAULT_NREL_PA_args,
            "USHCN": DEFAULT_USHCN_args,
        }

        custom_args = dataset_args_map.get(args.dataset, {})
        parser.set_defaults(**custom_args)

    args = parser.parse_args()

    return args


# Specific args

DEFAULT_PEMS03_args = {
    "mu": 0.0,
    "eta": 0.01,
}

DEFAULT_PEMS04_args = {
    "mu": 0.0,
    "eta": 0.01,
}

DEFAULT_LA_args = {
    "mu": 0.9,
    "eta": 0.01,
    "beta":0.1,
}

DEFAULT_BAY_args = {
    "mu": 0.7,
    "eta": 0.007,
    "beta":0.1,
}

DEFAULT_AIR36_args = {
    "mu": 0.7,
    "eta": 0.01,
    "beta":0.1,
}

DEFAULT_NREL_PA_args = {
    "mu": 0.85,
    "eta": 0.001,
    "beta":0.1,
}

DEFAULT_USHCN_args = {
    "mu": 0.9,
    "eta": 0.001,
    "beta":0.1,
}


if __name__ == "__main__":
    args = get_args()
    print(args.mu)
