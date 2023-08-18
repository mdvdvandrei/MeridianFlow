import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models
from DataSet import MyDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
from threading import Thread
from queue import Empty, Queue
import threading
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from models.models import resnext50_32x4d, mobilenet_v2, shufflenet_small
from torch.utils.tensorboard import SummaryWriter
import wandb

run = wandb.init(project='my_project')
run.save()

def find_directories(path, pattern, maxdepth):
    """
    Function to find directories that match a given pattern.

    Args:
    - path (str): The path in which to search.
    - pattern (str): The pattern to filter directories.
    - maxdepth (int): The maximum depth of the search.

    Returns:
    - directories (list): List of directories matching the pattern.
    """
    directories = []
    pattern = pattern.replace('*', '')  
    for root, dirs, files in os.walk(path):
        if root[len(path):].count(os.sep) < maxdepth:
            for directory in dirs:
                if pattern in directory:
                    directories.append(os.path.join(root, directory))
    return directories

tb_period = 8

run_prefix = 'resnext50_32x4d'

existing_logs_directories = [d for d in find_directories('./logs', '%s_run*' % run_prefix, maxdepth=2)]

prev_runs = [os.path.basename(os.path.split(d)[0]) for d in existing_logs_directories]
prev_runs = [int(s.replace('%s_run' % run_prefix, '')) for s in prev_runs]

if len(prev_runs) > 0:
    curr_run = np.max(prev_runs) + 1
else:
    curr_run = 1

curr_run = len(os.listdir('/app/amedvedev/scripts/TBoard')) + 1

curr_run = '%s_run%04d' % (run_prefix, curr_run)
logs_basepath = os.path.join('./logs', curr_run)
tb_basepath = os.path.join('./TBoard', curr_run)
print(curr_run)
run.name = curr_run

device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device1)
tb_writer = SummaryWriter(log_dir=tb_basepath)

checkpoints_basepath = '/app/amedvedev/data/models'

mean = 8619244861.558443
std = 4320463472.088931

def start_train(model):
    """
    Function to enable training of all layers in the model.

    Args:
    - model (torch.nn.Module): The model to enable training for.
    """
    for param in model.parameters():
        param.requires_grad = True

class thread_killer(object):
    """
    Boolean object for signaling a worker thread to terminate.
    """

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill

def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """
    Function to load batches of data into the queue from a dataset generator.

    Args:
    - tokill (callable): A function that returns a boolean indicating whether to terminate the thread.
    - batches_queue (Queue): The queue to store the batches of data.
    - dataset_generator (generator): The generator that yields data batches.

    Returns:
    None
    """
    while tokill() == False:
        for img, landmarks in dataset_generator:
            batches_queue.put((img, landmarks), block=True)
            if tokill() == True:
                return

def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue):
    """
    Function to transfer batches of data to GPU.

    Args:
    - tokill (callable): A function that returns a boolean indicating whether to terminate the thread.
    - cuda_batches_queue (Queue): The queue to store the GPU-transferred batches.
    - batches_queue (Queue): The queue from which to retrieve the data batches.

    Returns:
    None
    """
    while tokill() == False:
        (img, flow) = batches_queue.get(block=True)
        img = torch.from_numpy(img).float()
        img = img.type(torch.float32)

        normalized_flow = (flow - mean) / std

        normalized_flow = torch.from_numpy(normalized_flow)
        img = Variable(img.float()).to(device1)
        flow = Variable(normalized_flow.float()).to(device1)
        cuda_batches_queue.put((img, flow), block=True)

        if tokill() == True:
            return


batch_size = 8
dataset_train = MyDataset(
    root_dir='/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d',
    batch_size=batch_size, val_mark=False)
dataset_test = MyDataset(
    root_dir='/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d',
    batch_size=batch_size, val_mark=True)

def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_function: torch.nn.Module,
                       cuda_batches_queue: Queue,
                       Per_Step_Epoch: int,
                       current_epoch: int):
    """
    Function to perform a single training epoch.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - optimizer (torch.optim.Optimizer): The optimizer to update the model's parameters.
    - loss_function (torch.nn.Module): The loss function to compute the training loss.
    - cuda_batches_queue (Queue): The queue storing GPU-transferred data batches.
    - Per_Step_Epoch (int): The number of steps per training epoch.
    - current_epoch (int): The current epoch number.

    Returns:
    - loss_mean (float): The mean training loss for the epoch.
    """
    model.train()
    loss_values = []
    pbar = tqdm(total=Per_Step_Epoch)
    for batch_idx in range(int(Per_Step_Epoch)):
        data_image, target = cuda_batches_queue.get(block=True)

        target = Variable(target)

        optimizer.zero_grad()
        data_out = model(data_image)
        loss = loss_function(data_out, target.view(-1, 1))
        loss_values.append(loss.item())

        loss.backward()
        optimizer.step()

        pbar.update(1)
        pbar.set_postfix({'loss': loss.item()})

        '''
        for tag, param in model.named_parameters():
            #tb_writer.add_histogram('grad/%s'%tag, param.grad.data.cpu().numpy(), current_epoch)
            #tb_writer.add_histogram('weight/%s' % tag, param.data.cpu().numpy(), current_epoch)
            wandb.log({f"{tag}_grad": wandb.Histogram(param.grad.cpu().data.numpy())})
            wandb.log({f"{tag}_weight": wandb.Histogram(param.cpu().data.numpy())})
        '''


    pbar.close()

    loss_mean = np.mean(loss_values)
    return loss_mean

def validate_single_epoch(model: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          cuda_batches_queue: Queue,
                          Per_Step_Epoch: int,
                          current_epoch: int):
    """
    Function to perform a single validation epoch.

    Args:
    - model (torch.nn.Module): The model to be evaluated.
    - loss_function (torch.nn.Module): The loss function to compute the validation loss.
    - cuda_batches_queue (Queue): The queue storing GPU-transferred data batches.
    - Per_Step_Epoch (int): The number of steps per validation epoch.
    - current_epoch (int): The current epoch number.

    Returns:
    - loss_mean (float): The mean validation loss for the epoch.
    """
    model.eval()

    loss_values = []

    pbar = tqdm(total=Per_Step_Epoch)
    for batch_idx in range(int(Per_Step_Epoch)):
        data_image, target = cuda_batches_queue.get(block=True)
        data_out = model(data_image)

        loss = loss_function(data_out, target.view(-1, 1))
        loss_values.append(loss.item())
        pbar.update(1)
        pbar.set_postfix({'loss': loss.item()})

    pbar.close()

    loss_mean = np.mean(loss_values)
    return loss_mean


def train_model(model: torch.nn.Module,
                train_dataset,
                val_dataset,
                max_epochs=480):
    """
    Function to train a model on the given train and validation datasets.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - train_dataset: The training dataset.
    - val_dataset: The validation dataset.
    - max_epochs (int): The maximum number of training epochs (default: 480).

    Returns:
    None
    """
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=32, T_mult=2, eta_min=1e-8)

    # Data preprocessing threads starting
    batches_queue_length = 8
    preprocess_workers = 8

    train_batches_queue = Queue(maxsize=batches_queue_length)
    train_cuda_batches_queue = Queue(maxsize=4)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)

    for _ in range(preprocess_workers):
        thr = Thread(target=threaded_batches_feeder, args=(train_thread_killer, train_batches_queue, train_dataset))
        thr.start()

    train_cuda_transfers_thread_killer = thread_killer()
    train_cuda_transfers_thread_killer.set_tokill(False)
    train_cudathread = Thread(target=threaded_cuda_batches,
                             args=(train_cuda_transfers_thread_killer, train_cuda_batches_queue, train_batches_queue))
    train_cudathread.start()

    test_batches_queue = Queue(maxsize=batches_queue_length)
    test_cuda_batches_queue = Queue(maxsize=4)
    test_thread_killer = thread_killer()
    test_thread_killer.set_tokill(False)

    for _ in range(preprocess_workers):
        thr = Thread(target=threaded_batches_feeder, args=(test_thread_killer, test_batches_queue, val_dataset))
        thr.start()

    test_cuda_transfers_thread_killer = thread_killer()
    test_cuda_transfers_thread_killer.set_tokill(False)
    test_cudathread = Thread(target=threaded_cuda_batches,
                             args=(test_cuda_transfers_thread_killer, test_cuda_batches_queue,
                                   test_batches_queue))
    test_cudathread.start()

    Steps_Per_Epoch_Train = 64
    Steps_Per_Epoch_Test = len(val_dataset) // batch_size + 1

    best_val_loss = None
    best_epoch = None

    for epoch in range(max_epochs):
        print(f'Epoch {epoch} / {max_epochs}')
        train_loss = train_single_epoch(model, optimizer, loss_function, train_cuda_batches_queue,
                                        Steps_Per_Epoch_Train, current_epoch=epoch)
        val_loss = validate_single_epoch(model, loss_function, test_cuda_batches_queue, Steps_Per_Epoch_Test,
                                         current_epoch=epoch)

        wandb.log({"train_loss": train_loss, "epoch": epoch})
        wandb.log({"val_loss": val_loss, "epoch": epoch})
        print(f'Validation loss: {val_loss}')

        lr_scheduler.step()

        curr_run_dir = os.path.join(checkpoints_basepath, curr_run)
        os.makedirs(curr_run_dir, exist_ok=True)

        if best_val_loss is None or best_val_loss > val_loss:
            print(f'Best model yet, saving')
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.module.state_dict(), os.path.join(curr_run_dir, 'best_model.pt'))

        torch.save(model.module.state_dict(), os.path.join(curr_run_dir, 'model_ep%04d.pt' % epoch))

    # Stopping datapreprocessing threads
    test_thread_killer.set_tokill(True)
    train_thread_killer.set_tokill(True)
    test_cuda_transfers_thread_killer.set_tokill(True)
    train_cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread shutdown
            test_batches_queue.get(block=True, timeout=1)
            test_cuda_batches_queue.get(block=True, timeout=1)
            train_batches_queue.get(block=True, timeout=1)
            train_cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass



model = resnext50_32x4d()

for param in model.parameters():
    param.requires_grad = True

model = nn.DataParallel(model)
model = model.to(device1)

wandb.watch(model, log="all")

train_model(model,
            train_dataset=dataset_train,
            val_dataset=dataset_test,
            max_epochs=200)

wandb.finish()


