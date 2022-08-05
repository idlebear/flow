# PyTorch DataLoader for Autonomous vehicle motion prediction

from genericpath import samefile
import torch
import torch.nn as nn

# visualization and debugging
import wandb

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import time

from backbone_torch import BackBone
from lidar_dataset import LidarDataSet, ToTensor
from flow import FlowGrid
from focal_loss import FocalLoss

# Weights for the different loss functions to preserve some sort
# of priority for the occupancy over the motion vectors
KV_WEIGHT_FACTOR = 0.4
STATIC_WEIGHT_FACTOR = 1.3
DYNAMIC_WEIGHT_FACTOR = 0.6


def score_motion_vectors(gt_motion, predicted_motion, T=1, K=1, device='cuda'):
    # given the ground truth and a stack of motion vectors (should be
    # of the shape (N, K*2*T, H, W)), compare each of K pairs of vectors
    # for each of T time steps.
    #
    # At each time step, the result should be a Kx1 1 hot vector
    # with the closest vector to ground truth being the proposed
    # 'true' value
    #
    # The return shape is (N, K*T, H, W)
    output_size = predicted_motion.shape
    label_output = torch.zeros((output_size[0], T, output_size[2], output_size[3]), dtype=torch.long).to(device)
    one_hot_output = torch.zeros((output_size[0], T, K, output_size[2], output_size[3])).to(device)

    # revise the view of the predicted motion so that it is in the same format
    # as the gt:
    #
    #      (N, T, Dim, X, Y)
    #
    # where Dim is 2, the scale of the motion vectors, only we'll have T*K
    # in our view of motion so we can reshape as
    #
    #       (N, T, K, Dim, X, Y)
    #
    predicted_motion = predicted_motion.view((output_size[0], T, K, 2, output_size[2], output_size[3]))

    # for each batch entry
    for batch_index in range(output_size[0]):

        for t in range(T):

            accumulated_norm = None
            for k in range(K):
                pred_norm = torch.linalg.vector_norm(predicted_motion[batch_index, t, k, :, :, :] -
                                                     gt_motion[batch_index, t, :, :, :], ord=2, dim=0).unsqueeze(0)
                if accumulated_norm is None:
                    accumulated_norm = pred_norm
                else:
                    accumulated_norm = torch.cat([accumulated_norm, pred_norm], dim=0)

            # get the corresponding labels
            labels = torch.argmin(accumulated_norm, dim=0, keepdim=True)

            # TODO: apparently, one-hot vectors aren't used in CrossCategoricalEntropy, just the class
            #       labels (which are usually more memory efficient).  So, we should be able to just
            #       stack our minimum values - an array of values (0,1,...K-1) that identifies which
            #       is the most correct vector at any given time
            label_output[batch_index, t, :, :] = labels

            # TODO: however, we still need one hot vectors to calculate the loss function for the
            #       actual motion vectors
            one_hot_label = torch.nn.functional.one_hot(labels, num_classes=K)
            one_hot_label = torch.moveaxis(one_hot_label, -1, 1)
            one_hot_output[batch_index, t, :, :, :] = one_hot_label

    return label_output, one_hot_output


def calculate_score_loss(score_loss_fn, pred_score, target_score, K, T):

    # Loss is a compound function, with contributions coming from occupancy (as a binary cross
    # entropy with logits), motion scores (which of the vectors is the correct one to reduce),
    # and for the motions themselves
    loss = 0

    # reshape the inputs to make them compatible and a little more transparent
    target_score = target_score.unsqueeze(2)
    pred_shape = pred_score.shape
    pred_score = pred_score.view(-1, T, K, pred_shape[-2], pred_shape[-1])

    # for each time interval, calculate the motion score as catagorical cross entropy
    for t in range(T):
        # at each time step there is a one-hot vector that indicates the most correct estimate
        # of the motion -- however, this being pytorch, the cross entropy loss wants the one-hot
        # in the form of class value, a 1xHxW tensor for each time step
        loss += score_loss_fn(pred_score[:, t, :, :, :], target_score[:, t, 0, :, :])

    return loss


def calculate_motion_loss(motion_loss_fn, pred_motion, target_motion, motion_one_hot, K, T):
    #
    # To calculate the motion loss, we first need to find the motion loss with respect to all
    # vectors, then (presumably) zero out the ones we don't want/need
    #
    # Motion predictions are of form:  (N, C, H, W)
    #
    # where C is K * T * 2: K pairs for each time step.  We generated a one_hot vector, that we
    # need to convert into a two_hot version since we have two layers for each K (x and y dims)
    #
    # the motion one hot is of form: (N, 1, H, W, K)
    loss = 0

    # reshape the inputs to make them compatible and a little more transparent
    motion_one_hot = motion_one_hot.unsqueeze(3)
    pred_shape = pred_motion.shape
    pred_motion = pred_motion.view(-1, T, K, 2, pred_shape[-2], pred_shape[-1])
    target_motion = target_motion.unsqueeze(2)

    for t in range(T):
        # calculate the loss for each pair of (x,y) vector descriptors
        for k in range(K):
            loss += motion_loss_fn(pred_motion[:, t, k, :, :, :]*motion_one_hot[:, t, k, :, :, :],
                                   target_motion[:, t, 0, :, :, :]*motion_one_hot[:, t, k, :, :, :])

    return loss


def train_loop(dataloader, model, optimizer, static_pred_loss_fn, dyn_pred_loss_fn, score_loss_fn,
               motion_loss_fn, device='cuda', alpha=1.0, gamma=2.0):

    size = len(dataloader.dataset)
    for batch, sample in enumerate(dataloader):

        X = sample['data'].to(device)
        y_static_occ = sample['static'].to(device)
        y_dynamic_occ = sample['dynamic'].to(device)
        y_motion = sample['motion'].to(device)

        # forward pass
        pred_static_occ, pred_dyn_occ, pred_score, pred_motion = model(X)

        score_labels, score_one_hot = score_motion_vectors(y_motion, pred_motion, T=model.T, K=model.K, device=device)

        # static occupancy loss
        loss = static_pred_loss_fn(pred_static_occ.squeeze(), y_static_occ[:, 0, :, :].squeeze()) * STATIC_WEIGHT_FACTOR
        # loss = 0

        # dynamic occupancy loss
        loss += dyn_pred_loss_fn(pred_dyn_occ.squeeze(), y_dynamic_occ[:, 0, :, :].squeeze()) * DYNAMIC_WEIGHT_FACTOR

        # calculate the score loss
        loss += calculate_score_loss(score_loss_fn, pred_score, score_labels, T=model.T, K=model.K) * KV_WEIGHT_FACTOR

        # finally calculate the motion loss for just the single position/matching vector(s) to the k values identified by the one-hot -- multiplying the
        # values by the one-hot should zero the loss values corresponding to the non-relevent k values
        loss += calculate_motion_loss(motion_loss_fn, pred_motion, y_motion, score_one_hot, T=model.T, K=model.K) * KV_WEIGHT_FACTOR

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # visualization
        wandb.log({"train epoch": batch, "train loss": loss})

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, static_pred_loss_fn, dyn_pred_loss_fn, score_loss_fn, motion_loss_fn, device='cuda'):
    size = len(dataloader.dataset)
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            X = sample['data'].to(device)
            y_static_occ = sample['static'].to(device)
            y_dynamic_occ = sample['dynamic'].to(device)
            y_motion = sample['motion'].to(device)

            # forward pass
            pred_static_occ, pred_dyn_occ, pred_score, pred_motion = model(X)

            score_labels, score_one_hot = score_motion_vectors(y_motion, pred_motion, T=model.T, K=model.K)

            # static occupancy loss
            test_loss = static_pred_loss_fn(pred_static_occ.squeeze(), y_static_occ[:, 0, :, :].squeeze()) * STATIC_WEIGHT_FACTOR
            # test_loss = 0

            # dynamic occupancy loss
            # TODO: For now, we're using the same loss function as the static occupancy
            test_loss += dyn_pred_loss_fn(pred_dyn_occ.squeeze(), y_dynamic_occ[:, 0, :, :].squeeze()) * DYNAMIC_WEIGHT_FACTOR

            # # # calculate the score loss
            KV_loss = calculate_score_loss(score_loss_fn, pred_score, score_labels, T=model.T, K=model.K)

            # # # finally calculate the motion loss for just the single position/matching vector(s) to the k values identified by the one-hot -- multiplying the
            # # # values by the one-hot should zero the loss values corresponding to the non-relevent k values
            KV_loss += calculate_motion_loss(motion_loss_fn, pred_motion, y_motion, score_one_hot, T=model.T, K=model.K)

            # # apply a weighting to decrease the impact of the score/motion on the occupancy
            test_loss += KV_loss * KV_WEIGHT_FACTOR

            wandb.log({"test epoch": batch, "test loss": test_loss})

            if batch % 100 == 0:

                T = model.T
                K = model.K

                pred_static_occ, pred_dyn_occ, pred_score, pred_motion = model(X)

                # move everything to numpy for display
                pred_static_occ = pred_static_occ.cpu().detach().numpy()
                pred_dyn_occ = pred_dyn_occ.cpu().detach().numpy()
                y_dynamic_occ = y_dynamic_occ.cpu().detach().numpy()
                y_static_occ = y_static_occ.cpu().detach().numpy()
                y_motion = y_motion.cpu().detach().numpy()

                # score data is ( N, T, K, 1, H, W )
                pred_shape = pred_score.shape
                pred_score = pred_score.view(-1, T, K, 1, pred_shape[-2], pred_shape[-1])
                for t in range(T):
                    pred_score[:, t, :, :, :, :] = nn.Softmax(dim=1)(pred_score[:, t, :, :, :, :])
                pred_score = pred_score.cpu().detach().numpy()

                # motion data is (N, T, K, 2, H, W )
                pred_shape = pred_motion.shape
                pred_motion = pred_motion.view(-1, T, K, 2, pred_shape[-2], pred_shape[-1])
                pred_motion = pred_motion.cpu().detach().numpy()

                lidar_data = sample['data'].cpu().detach().numpy()
                lidar_img = lidar_data[0, 0, :, :]
                for j in range(25):
                    for k in range(10):
                        lidar_img = np.logical_or(lidar_img, lidar_data[0, j*10 + k, :, :])

                # proportion the vectors by their score contribution
                pred_xmo = np.zeros((10, 200, 200))
                pred_ymo = np.zeros((10, 200, 200))
                for t in range(model.T):
                    for k in range(model.K):
                        pred_xmo[t, :, :] += pred_score[0, t, k, 0, :, :] * pred_motion[0, t, k, 0, :, :]
                        pred_ymo[t, :, :] += pred_score[0, t, k, 0, :, :] * pred_motion[0, t, k, 1, :, :]

                gt_flow_grid = FlowGrid(y_dynamic_occ[0, 0, :, :], None, np.expand_dims(y_motion[0, :, :, :, :], axis=1), scale=0.4)
                flow_grid = FlowGrid(pred_dyn_occ[0, 0, :, :], pred_score[0, :, :, :, :, :], pred_motion[0, :, :, :, :, :], scale=0.4)
                gt_flow = gt_flow_grid.flow(mode='bilinear', dt=0.5)
                flow = flow_grid.flow(mode='bilinear', dt=0.5)

                fig, ax = plt.subplots(8, T+1, num=1, figsize=(25, 15))

                row_labels = ['LIDAR (->T=0)', 'GT STATIC', 'X motion', 'Y motion', 'Pred X motion', 'Pred Y motion', 'GT Flow', 'Pred Flow']
                for _ax, label in zip(ax[:, 0], row_labels[:]):
                    _ax.set_ylabel(label, size='x-large')

                col_labels = ['T = 0', 'T = 0.5', 'T = 1.0', 'T = 1.5', 'T = 2.0', 'T = 2.5', 'T = 3.0', 'T = 3.5', 'T = 4.0', 'T = 4.5']  # , 'T = 5.0', ]
                for _ax, label in zip(ax[0, :], col_labels):
                    _ax.set_title(label, size='x-large')

                ax[0, 0].imshow(lidar_img)
                ax[0, 1].imshow(pred_static_occ[0, 0, :, :])

                ax[6, 0].imshow(y_dynamic_occ[0, 0, :, :])
                ax[7, 0].imshow(pred_dyn_occ[0, 0, :, :])

                gt_img = np.squeeze(y_static_occ[0, :, :])
                ax[1, 0].imshow(gt_img)

                for j in range(T):

                    gt_xmo_img = np.squeeze(y_motion[0, j, 0, :, :])
                    gt_xmo_img -= np.min(gt_xmo_img)
                    gt_xmo_img = (gt_xmo_img / np.max(gt_xmo_img))
                    gt_ymo_img = np.squeeze(y_motion[0, j, 1, :, :])
                    gt_ymo_img -= np.min(gt_ymo_img)
                    gt_ymo_img = (gt_ymo_img / np.max(gt_ymo_img))

                    pred_xmo_img = np.squeeze(pred_xmo[j, :, :])
                    pred_xmo_img -= np.min(pred_xmo_img)
                    pred_xmo_img = (pred_xmo_img / np.max(pred_xmo_img))
                    pred_ymo_img = np.squeeze(pred_ymo[j, :, :])
                    pred_ymo_img -= np.min(pred_ymo_img)
                    pred_ymo_img = (pred_ymo_img / np.max(pred_ymo_img))

                    ax[2, j].imshow(gt_xmo_img)
                    ax[3, j].imshow(gt_ymo_img)
                    ax[4, j].imshow(pred_xmo_img)
                    ax[5, j].imshow(pred_ymo_img)
                    ax[6, j+1].imshow(gt_flow[j, :, :])
                    ax[7, j+1].imshow(flow[j, :, :])

                for k in range(0, 7):
                    for j in range(T+1):
                        ax[k, j].set_xticklabels([])
                        ax[k, j].set_yticklabels([])

                plt.draw()
                plt.pause(0.1)

    test_loss /= size

    print('Current Test Loss: {}'.format(test_loss))
    return test_loss


if __name__ == '__main__':

    wandb.init(project='lidar-prediction')

    path = 'data/lidar_static_dynamic'

    # setting device on GPU if available, else CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # instantiate the model and load it with the current best
    model = BackBone(3, 10).float().to(device)
    model_parameters_file = './full_occupation_model'

    # add a transform to convert the data from image format to friendly channel first format
    transform = transforms.Compose([ToTensor()])
    ds = LidarDataSet(path, data_prefix='town4-lidar-', static_target='static-occ', dynamic_target='dynamic-occ', transform=transform)
    print('Length: {}'.format(len(ds)))
    print('Shape: {}'.format(ds[2]['data'].shape))

    test_len = len(ds) // 10                    # 10% test data
    train_len = len(ds) - test_len              # and the remainder for training

    # hyperparameters
    learning_rate = 1e-4
    batch_size = 8
    epochs = 50

    wandb.config = {'learning_rate': learning_rate, 'epochs': epochs, 'batch_size': batch_size}
    wandb.watch(model)

    datasets = torch.utils.data.random_split(ds, (train_len, test_len), generator=torch.Generator().manual_seed(5))
    train_dataloader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(datasets[1], batch_size=batch_size, shuffle=True)

    for idx, sample in enumerate(train_dataloader):
        print("Shape of data [N, C, H, W]: ", sample['data'].shape)
        print("Shape of static occ label: ", sample['static'].shape, sample['static'].dtype)
        print("Shape of motion label: ", sample['motion'].shape, sample['motion'].dtype)
        break

    # pred_loss_fn = nn.BCELoss()  # nn.BCEWithLogitsLoss()
    static_pred_loss_fn = FocalLoss.LossFn(alpha=0.9, gamma=2, device=device)
    dyn_pred_loss_fn = FocalLoss.LossFn(alpha=0.2, gamma=2, device=device)
    score_loss_fn = nn.CrossEntropyLoss()
    motion_loss_fn = nn.HuberLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if isfile(model_parameters_file):
        model.load_state_dict(torch.load(model_parameters_file))
        min_loss = test_loop(test_dataloader, model, static_pred_loss_fn=static_pred_loss_fn, dyn_pred_loss_fn=dyn_pred_loss_fn,
                             score_loss_fn=score_loss_fn, motion_loss_fn=motion_loss_fn, device=device)
    else:
        min_loss = np.inf

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model=model, optimizer=optimizer, static_pred_loss_fn=static_pred_loss_fn,
                   dyn_pred_loss_fn=dyn_pred_loss_fn, score_loss_fn=score_loss_fn, motion_loss_fn=motion_loss_fn,
                   device=device, alpha=0.75, gamma=2)

        loss = test_loop(test_dataloader, model, static_pred_loss_fn=static_pred_loss_fn, dyn_pred_loss_fn=dyn_pred_loss_fn,
                         score_loss_fn=score_loss_fn, motion_loss_fn=motion_loss_fn, device=device)
        if loss < min_loss:
            torch.save(model.state_dict(), model_parameters_file)
            min_loss = loss
