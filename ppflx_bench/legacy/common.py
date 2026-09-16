import argparse
import os
import random
import torch
import shutil
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch.nn.functional
from collections import OrderedDict
from ppflx.core.security import *
from ppflx.core.security import _concrete_quantize, simulate_concrete_encrypt
from ppflx.core.zkp import zkp_commit_model, ZKPLayer
import io
import zlib
import struct
import logging
import gc
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Track emulated Concrete payload size for benchmarking (bytes)







def write_yaml(data, file_write="toyaml.yml", data1=None):
    """
    A function to write YAML file

    :param data: data to write in the YAML file
    :param file_write: path to save the YAML file
    :param data1: data to add in the YAML file (if we want to add data in the YAML file without overwriting it)
    :return: data (the data to write in the YAML file)
    """
    import yaml

    def accumul_time(time_key, data, data1):
        if time_key in data1 and time_key in data:
            data[time_key] += data1[time_key]

        return data

    path_yaml = os.path.join(*file_write.split("/")[:-1])
    print("save yaml in ", path_yaml)
    os.makedirs(path_yaml, exist_ok=True)
    with open(file_write, "w") as f:
        if data1:
            data = accumul_time("Train_time", data, data1)
            data = accumul_time("Test_time", data, data1)
            data = {**data1, **data}

        yaml.dump(data, f)

    return data


def read_yaml(yaml_file="config.yml"):
    """
    A function to read YAML file

    :param yaml_file: path to the YAML file
    :return: config (the data in the YAML file)
    """
    import yaml

    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    return config


def choice_device(device):
    """
    A function to choose the device

    :param device: the device to choose (cpu, gpu or mps)
    """
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and device != "cpu"
    ):
        """
        on Mac :
        - torch.backends.mps.is_available() ensures that the current MacOS version is at least 12.3+
        - torch.backends.mps.is_built() ensures that the current current PyTorch installation was built with MPS activated.
        """
        device = "mps"

    else:
        device = "cpu"

    return device


def classes_string(name_dataset):
    """
    A function to get the classes of the dataset

    :param name_dataset: the name of the dataset
    :return: classes (the classes of the dataset) in a tuple
    """
    if name_dataset == "cifar":
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    elif name_dataset == "animaux":
        classes = ("cat", "dog")

    elif name_dataset == "breast":
        classes = ("0", "1")

    elif name_dataset == "histo":
        classes = ("0", "1")

    elif name_dataset == "healthcare":
        classes = ("No Disease", "Disease")

    elif name_dataset == "creditcard":
        classes = ("Legitimate", "Fraud")

    elif name_dataset == "stock":
        classes = ("Down", "Up")

    else:
        print("Warning problem : unspecified dataset")
        return ()

    return classes


def parsing(description="PyTorch ImageNet Training"):
    """
    A function to work with command line arguments (argparse)

    :param description: the description of the model
    :return: argparser object (the arguments of the model)
    """
    # To define the argparse arguments
    # Create the top-level parser
    parent_parser = argparse.ArgumentParser(description="common", add_help=False)
    parent_parser.add_argument("--max_epochs", type=int, default=1)
    parent_parser.add_argument("--number_clients", type=int, default=2)
    parent_parser.add_argument(
        "--length", type=int, default=None, help="size at the entrance of the model"
    )
    parent_parser.add_argument("--batch_size", type=int, default=64)
    parent_parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="- Choice of the device between cpu and gpu "
        "(cuda if compatible Nvidia and mps if on mac\n"
        "- The choice the output may be cpu even if you choose the gpu if the latter isn't "
        "compatible",
    )
    parent_parser.add_argument(
        "--dataset", default="cifar", help="choice of the dataset (cifar10 by default)"
    )
    parent_parser.add_argument(
        "--data_path", type=str, default="./data/", help="Path to the training data"
    )
    parent_parser.add_argument(
        "--data_path_val", type=str, default=None, help="Path to the validation dataset"
    )
    parent_parser.add_argument(
        "--model_save", type=str, default="", help="Path to save the central model"
    )
    parent_parser.add_argument(
        "--yaml_path",
        type=str,
        default="./results/results.yml",
        help="Path to save the metrics results",
    )
    parent_parser.add_argument("--seed", type=int, default=42)
    parent_parser.add_argument("--num_workers", type=int, default=0)
    parent_parser.add_argument(
        "--split",
        default=10,
        type=int,
        help="ratio (in percent) of the training dataset that will be used for the test "
        "(default : 10)",
    )
    parent_parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="learning rate for the central model" "(default : 0.001)",
    )
    parent_parser.add_argument(
        "--he",
        default=False,
        required=False,
        action="store_true",
        dest="he",
        help="True if we want to use the homomorphic encryption (by default : False)",
    )
    parent_parser.add_argument(
        "--he_backend",
        type=str,
        choices=["tenseal", "concrete", "concrete_tfhe"],
        default="tenseal",
        help="HE library to use when --he is enabled: 'tenseal' (default), 'concrete' (simulated), or 'concrete_tfhe' (actual FHE)",
    )
    parent_parser.add_argument(
        "--zkp",
        default=False,
        required=False,
        action="store_true",
        dest="zkp",
        help="True if we want to use zero-knowledge proofs (by default : False)",
    )
    parent_parser.add_argument(
        "--path_keys",
        type=str,
        default="keys/he_tenseal/secret_context.bin",
        help="Path to get the combo private/public keys",
    )
    parent_parser.add_argument(
        "--path_public_key",
        type=str,
        default="keys/he_tenseal/public_context.bin",
        help="Path to get the the public key",
    )
    parent_parser.add_argument(
        "--path_crypted",
        type=str,
        default="server_weights.bin",
        help="Path to save the crypted (and not crypted) weights",
    )
    parent_parser.add_argument(
        "--zkp_params",
        type=str,
        default="keys/zkp/zkp_params.json",
        help="Path to get/save the ZKP parameters",
    )
    parent_parser.add_argument(
        "--zkp_backend",
        type=str,
        choices=["pedersen", "gnark"],
        default=os.environ.get("FL_ZKP_BACKEND", "gnark"),
        help="ZKP backend to use: 'gnark' (default) or 'pedersen'",
    )
    parent_parser.add_argument(
        "--dp",
        default=False,
        required=False,
        action="store_true",
        dest="dp",
        help="True if we want to use differential privacy (by default : False)",
    )
    parent_parser.add_argument(
        "--dp_params",
        type=str,
        default="keys/dp/dp_params.json",
        help="Path to get/save the DP parameters",
    )
    parent_parser.add_argument(
        "--benchmark",
        default=False,
        required=False,
        action="store_true",
        dest="benchmark",
        help="True if we want to enable detailed benchmarking (by default : False)",
    )

    # Create the specific commands for the "classic ML"
    parser_ml = argparse.ArgumentParser(description="classic", add_help=False)
    parser_ml.add_argument(
        "--matrix_path",
        type=str,
        default=None,
        help="Path to save the confusion matrix",
    )
    parser_ml.add_argument(
        "--roc_path", type=str, default=None, help="Path to save the roc figures"
    )
    parser_ml.add_argument(
        "--save_results", type=str, default=None, help="Path to save the results"
    )

    # Create the specific commands for the "server"
    parser_server = argparse.ArgumentParser(description="server", add_help=False)
    parser_server.add_argument("--frac_fit", type=float, default=1.0)
    parser_server.add_argument("--frac_eval", type=float, default=0.5)
    parser_server.add_argument("--min_fit_clients", type=int, default=2)
    parser_server.add_argument("--min_eval_clients", type=int, default=None)
    parser_server.add_argument("--min_avail_clients", type=int, default=2)

    parser_server.add_argument(
        "--rounds", default=3, type=int, help="number of rounds (default : 3)"
    )

    # Create the specific commands for the "client"
    parser_client = argparse.ArgumentParser(description="client", add_help=False)
    parser_client.add_argument(
        "--id_client", type=str, default=None, help="client id (by default None)"
    )

    # Create the final parser
    main_parser = argparse.ArgumentParser(description=description)

    # Create the subparser of the final parser
    service_subparsers = main_parser.add_subparsers(
        title="service", dest="service_command"
    )

    # Add specific command for each choice (classic, client, server and simulation)
    classic_subparser = service_subparsers.add_parser(
        "run", help="classic ML", parents=[parent_parser, parser_ml]
    )

    # python client.py client
    client_subparser = service_subparsers.add_parser(
        "client", help="client", parents=[parent_parser, parser_ml, parser_client]
    )

    # python server.py server
    server_subparser = service_subparsers.add_parser(
        "server", help="server", parents=[parent_parser, parser_server]
    )

    simul_subparser = service_subparsers.add_parser(
        "simulation",
        help="client",
        parents=[parent_parser, parser_server, parser_ml, parser_client],
    )

    return main_parser


# functions for the dataset creation
def supp_ds_store(path):
    """
    Delete the hidden file ".DS_Store" created on macOS

    :param path: path to the folder where the hidden file ".DS_Store" is
    """
    for i in os.listdir(path):
        if i == ".DS_Store":
            print("Deleting of the hidden file '.DS_Store'")
            os.remove(path + "/" + i)


def create_files_train_test(path_init, path_final, splitter):
    """
    Split the dataset from path_init into two datasets : train and test in path_final
    with the splitter ratio (in %). Example : if splitter = 10, 10% of the initial dataset will be in the test dataset.

    :param path_init: path of the initial dataset
    :param path_final: path of the final dataset
    :param splitter: ratio (in %) of the initial dataset that will be in the test dataset.
    """
    # Move a file from rep1 to rep2
    for classe in os.listdir(path_init):
        list_init = os.listdir(path_init + "/" + classe)
        size_test = int(len(list_init) * splitter / 100)
        print("Before : ", len(list_init))
        for _ in range(size_test):
            e = random.choice(list_init)  # random choice of the path of an image
            list_init.remove(e)
            shutil.move(path_init + classe + "/" + e, path_final + classe + "/" + e)

        print("After", path_init + classe, ":", len(os.listdir(path_init + classe)))
        print(path_final + classe, ":", len(os.listdir(path_final + classe)))


def save_matrix(y_true, y_pred, path, classes):
    """
    Save the confusion matrix in the path given in argument.

    :param y_true: true labels (real labels)
    :param y_pred: predicted labels (labels predicted by the model)
    :param path: path to save the confusion matrix
    :param classes: list of the classes
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sn
    # To get the confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # To normalize the confusion matrix
    cf_matrix_normalized = cf_matrix / np.sum(cf_matrix) * 10

    # To round up the values in the matrix
    cf_matrix_round = np.round(cf_matrix_normalized, 2)

    # To plot the matrix
    df_cm = pd.DataFrame(
        cf_matrix_round, index=[i for i in classes], columns=[i for i in classes]
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)

    plt.savefig(path)
    plt.close()


def save_roc(targets, y_proba, path, nbr_classes):
    """
    Save the roc curve in the path given in argument.

    :param targets: true labels (real labels)
    :param y_proba: predicted labels (labels predicted by the model)
    :param path: path to save the roc curve
    :param nbr_classes: number of classes
    """
    import matplotlib.pyplot as plt
    y_true = np.zeros(
        shape=(len(targets), nbr_classes)
    )  # array-like of shape (n_samples, n_classes)
    for i in range(len(targets)):
        y_true[i, targets[i]] = 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nbr_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nbr_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nbr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nbr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    lw = 2
    for i in range(nbr_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label="Worst case")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC) Curve OvR")  # One vs Rest
    plt.legend(loc="lower right")  # loc="best"

    plt.savefig(path)
    plt.close()


def save_graphs(path_save, local_epoch, results, end_file=""):
    """
    Save the graphs in the path given in argument.

    :param path_save: path to save the graphs
    :param local_epoch: number of epochs
    :param results: results of the model (accuracy and loss)
    :param end_file: end of the name of the file
    """
    os.makedirs(path_save, exist_ok=True)  # to create folders results
    print("save graph in ", path_save)
    # plot training curves (train and validation)
    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc"], results["val_acc"]],
        "Epochs",
        "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=path_save + "Accuracy_curves" + end_file,
    )

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss"], results["val_loss"]],
        "Epochs",
        "Loss",
        curve_labels=["Training loss", "Validation loss"],
        title="Loss curves",
        path=path_save + "Loss_curves" + end_file,
    )


def plot_graph(
    list_xplot, list_yplot, x_label, y_label, curve_labels, title, path=None
):
    """
    Plot the graph of the list of points (list_xplot, list_yplot)
    :param list_xplot: list of list of points to plot (one line per curve)
    :param list_yplot: list of list of points to plot (one line per curve)
    :param x_label: label of the x axis
    :param y_label: label of the y axis
    :param curve_labels: list of labels of the curves (curve names)
    :param title: title of the graph
    :param path: path to save the graph
    """
    import matplotlib.pyplot as plt
    lw = 2

    plt.figure()
    for i in range(len(curve_labels)):
        plt.plot(list_xplot[i], list_yplot[i], lw=lw, label=curve_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if curve_labels:
        plt.legend(loc="lower right")

    if path:
        plt.savefig(path)






# /////////////////////// Concrete TFHE & Inference Integration \\\\\\\\\\\\\\






def get_concrete_inference_paths() -> Dict[str, str]:
    """
    Get paths for Concrete-ML inference deployment.

    Returns:
        Dict with 'client_dir' and 'server_dir' paths
    """
    output_root = os.environ.get("FL_CONCRETE_INFERENCE_DIR", "./concrete_inference")
    return {
        "client_dir": os.path.join(output_root, "client"),
        "server_dir": os.path.join(output_root, "server"),
    }


def setup_concrete_encrypted_inference(
    model: Any,
    calibration_data: np.ndarray,
    n_bits: int = 8,
    output_dir: Optional[str] = None,
) -> str:
    """
    Setup Concrete-ML FHE model for encrypted inference (Option B).

    Compiles model once; clients then encrypt test data and get predictions
    from server without server seeing plaintext inputs/outputs.

    Args:
        model: Trained PyTorch model
        calibration_data: Representative data for quantization
        n_bits: Quantization bit-width
        output_dir: Where to save compiled artifacts

    Returns:
        Path to compiled model directory

    Example:
        >>> from core.common import setup_concrete_encrypted_inference
        >>> compiled_path = setup_concrete_encrypted_inference(
        ...     model, calibration_data, n_bits=8
        ... )
        >>> # Server loads from compiled_path
        >>> from core.security import get_concrete_inference_server
        >>> server = get_concrete_inference_server(compiled_path)
    """
    try:
        from .security import compile_model_to_concrete_fhe

        if output_dir is None:
            output_dir = get_concrete_inference_paths()["server_dir"]

        logger.info(f"Compiling model to Concrete-ML FHE: {output_dir}")
        compiled_path = compile_model_to_concrete_fhe(
            model=model,
            calibration_data=calibration_data,
            output_dir=output_dir,
            n_bits=n_bits,
        )

        logger.info(f"Model compiled successfully to {compiled_path}")
        return compiled_path
    except Exception as e:
        logger.error(f"Concrete-ML compilation failed: {e}")
        raise


def get_parameters_with_concrete_inference(
    net,
    inference_mode: bool = False,
) -> List[np.ndarray]:
    """
    Get model parameters for Concrete-ML inference deployment.

    Converts model to format suitable for FHE inference circuits.

    Args:
        net: PyTorch model
        inference_mode: Whether to prepare for inference (vs training)

    Returns:
        List of parameters as numpy arrays
    """
    params = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return params


# Tutorial and Integration Helpers


def demo_concrete_tfhe_aggregation(
    num_clients: int = 3,
    bit_width: int = 8,
) -> Dict[str, Any]:
    """
    Demo Concrete TFHE parameter aggregation (Option C).

    Shows how clients encrypt params, server aggregates encrypted,
    clients decrypt result - all without server seeing plaintext.

    Args:
        num_clients: Number of clients
        bit_width: Quantization bit-width

    Returns:
        Dict with timing and verification results
    """
    try:
        from .security import get_concrete_aggregation_context
        from .concrete_agg import ConcreteAggregator
        import time

        logger.info(f"Demo: Concrete TFHE aggregation with {num_clients} clients")

        # Setup context
        context = get_concrete_aggregation_context(
            bit_width=bit_width,
            vector_size=1000,
            enable_fhe=False,  # Use simulation for demo
        )

        # Generate keys for each client
        client_keys = [context.generate_keys() for _ in range(num_clients)]

        # Create dummy parameters
        dummy_params = {
            f"layer_{i}": np.random.randn(100).astype(np.float32) for i in range(5)
        }

        # Encrypt
        start = time.time()
        encrypted_list = []
        for private_key, _ in client_keys:
            encrypted_params = []
            for tensor in dummy_params.values():
                enc = context.encrypt_tensor(tensor, private_key)
                encrypted_params.append(enc)
            encrypted_list.append(encrypted_params)
        encrypt_time = time.time() - start

        # Aggregate
        start = time.time()
        aggregator = ConcreteAggregator(context, num_clients=num_clients)
        weight = 1.0 / num_clients
        for encrypted_params in encrypted_list:
            aggregator.add_encrypted(encrypted_params, weight=weight)
        aggregated = aggregator.get_result()
        aggregate_time = time.time() - start

        # Decrypt
        start = time.time()
        private_key, _ = client_keys[0]
        decrypted = [context.decrypt_tensor(enc, private_key) for enc in aggregated]
        decrypt_time = time.time() - start

        return {
            "num_clients": num_clients,
            "bit_width": bit_width,
            "encrypt_time": encrypt_time,
            "aggregate_time": aggregate_time,
            "decrypt_time": decrypt_time,
            "total_time": encrypt_time + aggregate_time + decrypt_time,
            "encrypted_params_count": len(aggregated),
        }
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def demo_concrete_encrypted_inference(
    model: Any,
    test_data: np.ndarray,
) -> Dict[str, Any]:
    """
    Demo Concrete-ML encrypted inference (Option B).

    Shows client encrypting test input, server running inference
    on encrypted data, client decrypting encrypted result.

    Args:
        model: PyTorch model
        test_data: Test data for inference

    Returns:
        Dict with timing and correctness results
    """
    try:
        from .concrete_inference import FHEInferenceSimulator
        import time

        logger.info("Demo: Concrete-ML encrypted inference")

        # Use simulator for demo (avoids compilation)
        simulator = FHEInferenceSimulator(model, quantization_params={"n_bits": 8})

        # Simulate inference
        start = time.time()
        predictions = simulator.predict(
            test_data[:5] if len(test_data) > 5 else test_data
        )
        inference_time = time.time() - start

        return {
            "predictions_shape": predictions.shape,
            "inference_time": inference_time,
            "samples": len(test_data[:5]),
            "time_per_sample": inference_time / len(test_data[:5]),
        }
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
