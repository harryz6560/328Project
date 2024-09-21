import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

from A1_submission import logistic_regression, tune_hyper_parameter

torch.multiprocessing.set_sharing_strategy('file_system')


def compute_score(acc, acc_thresh):
    min_thres, max_thres = acc_thresh
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


def test(
        model,
        dataset_name,
        device,

):
    if dataset_name == "MNIST":
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))

    elif dataset_name == "CIFAR10":
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    else:
        raise AssertionError(f"Invalid dataset: {dataset_name}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True)

    model.eval()
    num_correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)
            total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

    acc = float(num_correct) / total
    return acc


class Args:
    """
    command-line arguments
    """

    """    
    'MNIST': run on MNIST dataset (part 1)
    'CIFAR10': run on CIFAR10 dataset (part 2)
    """
    dataset = "MNIST"
    # dataset = "CIFAR10"

    """
    'logistic': run logistic regression on the specified dataset (parts 1 and 2)
    'tune': run hyper parameter tuning (part 3)
    """
    mode = 'logistic'
    # mode = 'tune'

    """
    metric with respect to which hyper parameters are to be tuned
    'acc': validation classification accuracy
    'loss': validation loss
    """
    target_metric = 'acc'
    # target_metric = 'loss'

    """
    set to 0 to run on cpu
    """
    gpu = 1


def main():
    args = Args()
    try:
        import paramparse
        paramparse.process(args)
    except ImportError:
        pass

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    acc_thresh = dict(
        MNIST=(0.84, 0.94),
        CIFAR10=(0.30, 0.40),
    )

    if args.mode == 'logistic':
        start = timeit.default_timer()
        results = logistic_regression(args.dataset, device)
        model = results['model']

        if model is None:
            print('model is None')
            return

        stop = timeit.default_timer()
        run_time = stop - start

        accuracy = test(
            model,
            args.dataset,
            device,
        )

        score = compute_score(accuracy, acc_thresh[args.dataset])
        result = OrderedDict(
            accuracy=accuracy,
            score=score,
            run_time=run_time
        )
        print(f"result on {args.dataset}:")
        for key in result:
            print(f"\t{key}: {result[key]}")
    elif args.mode == 'tune':
        start = timeit.default_timer()
        best_params, best_metric = tune_hyper_parameter(
            args.dataset, args.target_metric, device)
        stop = timeit.default_timer()
        run_time = stop - start
        print()
        print(f"Best {args.target_metric}: {best_metric:.4f}")
        print(f"Best params:\n{best_params}")
        print(f"runtime of tune_hyper_parameter: {run_time}")
    else:
        raise AssertionError(f'invalid mode: {args.mode}')


if __name__ == "__main__":
    main()
