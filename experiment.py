import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from datetime import datetime
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from semDataset import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models import *
import seaborn as sns
from pynvml import *
from loss import weighted_l1_loss
nvmlInit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def abs_err(x, y):
    return np.abs(x-y)


def reg_err(x, y):
    return x-y


class Experiment(object):
    # initialize the model
    def __init__(self, name="default", root_dir="./", stats_dir="experiments"):
        try:
            config_data_path = os.path.join(root_dir, f"{name}.json")
            if os.path.isfile(config_data_path):
                with open(config_data_path) as json_file:
                    config_data = json.load(json_file)
            else:
                raise Exception("Not valid path!")
        except Exception as e:
            print(e)
            print("Failed to open config for experiment: " + f"{name}.json")

        # Initialize experiment folder directory
        self.__name = config_data["experiment_name"]
        print(f"Loading {self.__name}")
        self.__expt_dir = os.path.join(root_dir, stats_dir, self.__name)
        os.makedirs(self.__expt_dir, exist_ok=True)
        self.__train_log_path = os.path.join(
            self.__expt_dir, config_data['output']["train_file_suffix"])
        self.__plot_folder_path = os.path.join(
            self.__expt_dir, config_data['output']["plot_folder_suffix"])
        os.makedirs(self.__plot_folder_path, exist_ok=True)
        self.__best_model_path = os.path.join(
            self.__expt_dir, config_data['output']["best_model_path"])
        self.dpi = config_data['output']['dpi']

        # Initialize dataset, for this case there should only be one dataset
        if config_data['dataset']["sem_dataset"] == "sem_dataset":
            dataset = semDataset
        else:
            raise Exception("Dataset not supported!")

        # Transforms go here
        # TODO: Modularize rescaling factors

        # Configurable transforms
        self.rotate = config_data['transforms']['rotate']
        self.resize = config_data['transforms']['resize']
        self.crop = config_data['transforms']['crop']
        self.normalize = config_data['transforms']['normalize']
        transform = transforms.Compose(
            [transforms.RandomRotation(degrees=self.rotate),
             transforms.Resize(self.resize),
             #     transforms.Grayscale(3),
             transforms.CenterCrop(self.crop),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(*self.normalize)
             ]
        )
        eval_transform = transforms.Compose(
            [transforms.Resize(self.resize),
             #     transforms.Grayscale(3),
             transforms.CenterCrop(self.crop),
             transforms.ToTensor(),
             transforms.Normalize(*self.normalize)
             ]
        )

        # Generate the train/val/test split
        train_idx, test_idx = train_test_split(list(range(len(dataset(filepath=config_data['dataset']['filepath'])))),
                                               test_size=config_data['dataset']["test_split"],
                                               random_state=config_data['dataset']["random_state"])
        train_idx, val_idx = train_test_split(train_idx,
                                              test_size=config_data['dataset']["val_split"],
                                              random_state=config_data['dataset']["random_state"])
        self.train_data = torch.utils.data.Subset(
            dataset(filepath=config_data['dataset']['filepath'], transform=transform, lds=config_data['hparams']['lds'], lds_ks=config_data['hparams']['lds_ks'], lds_sigma=config_data['hparams']['lds_sigma'], bf=config_data['hparams']['bf']), train_idx)
        self.train_eval_data = torch.utils.data.Subset(
            dataset(filepath=config_data['dataset']['filepath'], transform=eval_transform, lds=config_data['hparams']['lds'], lds_ks=config_data['hparams']['lds_ks'], lds_sigma=config_data['hparams']['lds_sigma'], bf=config_data['hparams']['bf']), train_idx)
        self.val_data = torch.utils.data.Subset(
            dataset(filepath=config_data['dataset']['filepath'], transform=eval_transform, lds=config_data['hparams']['lds'], lds_ks=config_data['hparams']['lds_ks'], lds_sigma=config_data['hparams']['lds_sigma'], bf=config_data['hparams']['bf']), val_idx)
        self.test_data = torch.utils.data.Subset(
            dataset(filepath=config_data['dataset']['filepath'], transform=eval_transform, lds=config_data['hparams']['lds'], lds_ks=config_data['hparams']['lds_ks'], lds_sigma=config_data['hparams']['lds_sigma'], bf=config_data['hparams']['bf']), test_idx)
        self.all_data = dataset(filepath=config_data['dataset']['filepath'],
                                transform=eval_transform, lds=config_data['hparams']['lds'], lds_ks=config_data['hparams']['lds_ks'], lds_sigma=config_data['hparams']['lds_sigma'], bf=config_data['hparams']['bf'])

        self.batch_size = config_data['hparams']['batch_size']
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                       shuffle=True, num_workers=0, persistent_workers=False)
        self.train_eval_loader = DataLoader(self.train_eval_data, batch_size=self.batch_size,
                                            shuffle=False, num_workers=0, persistent_workers=False)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size,
                                     shuffle=False, num_workers=0, persistent_workers=False)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                      shuffle=False, num_workers=0, persistent_workers=False)

        # Initialize model, optimizer and scheduler

        # Intialize model
        pretrained = config_data["model"]['pretrained_model']
        # If you want to add a potential model, modify this
        if config_data["model"]["model_name"] == "resnext_large":
            self.model = SEMPro_resNext(fc_size=config_data["model"]['fc_size'],
                                        large=True, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "resnext_small":
            self.model = SEMPro_resNext(fc_size=config_data["model"]['fc_size'],
                                        large=False, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "densenet121":
            self.model = SEMPro_denseNet(fc_size=config_data["model"]['fc_size'],
                                         size=0, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "densenet169":
            self.model = SEMPro_denseNet(fc_size=config_data["model"]['fc_size'],
                                         size=1, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "densenet201":
            self.model = SEMPro_denseNet(fc_size=config_data["model"]['fc_size'],
                                         size=2, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "densenet161":
            self.model = SEMPro_denseNet(fc_size=config_data["model"]['fc_size'],
                                         size=3, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "convnext_tiny":
            self.model = SEMPro_ConvNext(fc_size=config_data["model"]['fc_size'],
                                         size=0, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "convnext_small":
            self.model = SEMPro_ConvNext(fc_size=config_data["model"]['fc_size'],
                                         size=1, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "convnext_base":
            self.model = SEMPro_ConvNext(fc_size=config_data["model"]['fc_size'],
                                         size=2, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "convnext_large":
            self.model = SEMPro_ConvNext(fc_size=config_data["model"]['fc_size'],
                                         size=3, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "alexnet":
            self.model = SEMPro_AlexNet()
        elif config_data["model"]["model_name"] == "vgg_11":
            self.model = SEMPro_VGG(fc_size=config_data["model"]['fc_size'],
                                    size=0, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "vgg_13":
            self.model = SEMPro_VGG(fc_size=config_data["model"]['fc_size'],
                                    size=1, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "vgg_16":
            self.model = SEMPro_VGG(fc_size=config_data["model"]['fc_size'],
                                    size=2, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "vgg_19":
            self.model = SEMPro_VGG(fc_size=config_data["model"]['fc_size'],
                                    size=3, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "resnet18":
            self.model = SEMPro_ResNet(fc_size=config_data["model"]['fc_size'],
                                       size=0, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "resnet34":
            self.model = SEMPro_ResNet(fc_size=config_data["model"]['fc_size'],
                                       size=1, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "resnet50":
            self.model = SEMPro_ResNet(fc_size=config_data["model"]['fc_size'],
                                       size=2, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "resnet101":
            self.model = SEMPro_ResNet(fc_size=config_data["model"]['fc_size'],
                                       size=3, pretrained=pretrained)
        elif config_data["model"]["model_name"] == "googlenet":
            self.model = SEMPro_GoogLeNet()
        elif config_data["model"]["model_name"] == "baseplate":
            self.model = torch.load("experiments/baseplate/best.pt")
        else:
            raise Exception("Invalid model specified!")
        self.model.to(device)

        # Initialize optimizer
        # If you want to change the optimizer, modify this
        if config_data['hparams']['optimizer'] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=config_data['hparams']["lr"],
                                               weight_decay=config_data['hparams']["weight_decay"])
        elif config_data['hparams']['optimizer'] == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=config_data['hparams']["lr"],
                                              weight_decay=config_data['hparams']["weight_decay"])
        else:
            raise Exception("Invalid optimizer specified")

        # Intialize criterion
        # If you want to change the criterion, modify this
        if config_data['model']['criterion'] == 'L1':
            self.criterion = nn.L1Loss()
        elif config_data['model']['criterion'] == 'L2':
            self.criterion = nn.MSELoss()
        else:
            raise Exception("Invalid criterion (loss function) specified!")
        self.criterion.to(device)

        # Initialize scheduler
        # If you want to change the scheduler, modify this
        if config_data['hparams']['LR_scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config_data['hparams']['steps_per_decay'],
                                                             gamma=config_data['hparams']['LR_gamma'])
        elif config_data['hparams']['LR_scheduler'] == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer)
        else:
            self.scheduler = None
        self.epochs = config_data['epochs']

        # Initialize logging data here
        self.train_mae_loss_list = []
        self.train_mse_loss_list = []
        self.val_mae_loss_list = []
        self.val_mse_loss_list = []
        print("Initialized experiment")

    # train the experiment
    def train(self):
        '''
        Training and validation loop for the model
        '''
        best_mae = 1000000000
        for epoch in range(self.epochs):
            print(f"Now running epoch {epoch}")
            # Iterate through training dataset
            train_mae_loss = 0
            train_mse_loss = 0
            val_mae_loss = 0
            val_mse_loss = 0
            self.model.train()
            for image, label, weight, path in tqdm(self.train_loader, desc="Training data: "):
                # Send variables to device
                image = image.to(device)
                label = label.to(device)
                weight = weight.to(device)
                # Zero gradients
                self.optimizer.zero_grad()
                # Predict values
                pred = self.model(image)
                if type(pred).__name__ == "GoogLeNetOutputs":
                    # Compute loss
                    # loss = self.criterion(pred.squeeze(), label, weight)
                    loss = weighted_l1_loss(
                        pred.logits.squeeze(), label, weight)
                    # Backpropagate
                    loss.backward(retain_graph=True)
                    # Compute loss
                    # loss = self.criterion(pred.squeeze(), label, weight)
                    loss = weighted_l1_loss(
                        pred.aux_logits2.squeeze(), label, weight)
                    # Backpropagate
                    loss.backward(retain_graph=True)
                    # Compute loss
                    # loss = self.criterion(pred.squeeze(), label, weight)
                    loss = weighted_l1_loss(
                        pred.aux_logits1.squeeze(), label, weight)
                    # Backpropagate
                    loss.backward()
                    # Update parameters
                    self.optimizer.step()
                    # Move variables to cpu
                    pred = pred.logits.cpu().detach()
                    label = label.cpu().detach()
                    # Compute losses
                    train_mae_loss += mean_absolute_error(
                        pred, label)*image.shape[0]
                    train_mse_loss += mean_squared_error(pred,
                                                         label)*image.shape[0]
                else:
                    # Compute loss
                    # loss = self.criterion(pred.squeeze(), label, weight)
                    loss = weighted_l1_loss(pred.squeeze(), label, weight)
                    # Backpropagate
                    loss.backward()
                    # Update parameters
                    self.optimizer.step()
                    # Move variables to cpu
                    pred = pred.cpu().detach()
                    label = label.cpu().detach()
                    # Compute losses
                    train_mae_loss += mean_absolute_error(
                        pred, label)*image.shape[0]
                    train_mse_loss += mean_squared_error(pred,
                                                         label)*image.shape[0]
            train_mae_loss /= len(self.train_data)
            train_mse_loss /= len(self.train_data)
            self.model.eval()
            # Run validation dataset
            for image, label, weight, path in tqdm(self.val_loader, desc="Validation data: "):
                # Send variables to device
                image = image.to(device)
                label = label.to(device)
                # Run model
                pred = self.model(image)
                # Move variables to cpu
                pred = pred.cpu().detach()
                label = label.cpu().detach()
                # Compute loss
                val_mae_loss += mean_absolute_error(pred, label)*image.shape[0]
                val_mse_loss += mean_squared_error(pred, label)*image.shape[0]
            val_mae_loss /= len(self.val_data)
            val_mse_loss /= len(self.val_data)
            print(
                f"Training losses: MAE = {train_mae_loss}, MSE = {train_mse_loss}")
            print(
                f"Validation losses: MAE = {val_mae_loss}, MSE = {val_mse_loss}")
            self.train_mae_loss_list.append(train_mae_loss)
            self.train_mse_loss_list.append(train_mse_loss)
            self.val_mae_loss_list.append(val_mae_loss)
            self.val_mse_loss_list.append(val_mse_loss)
            if self.scheduler:
                self.scheduler.step(val_mae_loss)
            # Checkpoint best model
            if val_mae_loss < best_mae:
                best_mae = val_mae_loss
                torch.save(self.model, self.__best_model_path)  # TODO
        log_data = {"Train MAE": self.train_mae_loss_list,
                    "Train MSE": self.train_mse_loss_list,
                    "Val MAE": self.val_mae_loss_list,
                    "Val MSE": self.val_mse_loss_list}
        train_df = pd.DataFrame(log_data)
        train_df.to_csv(self.__train_log_path)

    # test the experiment

    def test(self):
        model = self.get_best_model()
        model.eval()
        # Run testing loop
        test_mae_loss = 0
        test_mse_loss = 0
        for image, label, weight, path in tqdm(self.test_loader, desc="Test data: "):
            # Send variables to device
            image = image.to(device)
            label = label.to(device)
            # Run model
            pred = model(image)
            # Move variables to cpu
            pred = pred.cpu().detach()
            label = label.cpu().detach()
            # Compute loss
            test_mae_loss += mean_absolute_error(pred, label)*image.shape[0]
            test_mse_loss += mean_squared_error(pred, label)*image.shape[0]
        test_mae_loss /= len(self.test_data)
        test_mse_loss /= len(self.test_data)
        print(f"Test loss: MAE = {test_mae_loss}, MSE = {test_mse_loss}")
        return test_mae_loss, test_mse_loss

    # for additional functionality
    def get_best_model(self):
        return torch.load(self.__best_model_path)

    def set_best_model(self, path):
        self.__best_model_path = path

    def analyze_training(self):
        '''
        This function analyzes the training loop and data distribution of the model,
        and saves the plots of the loss curves.
        '''
        # plot log10 MAE loss
        plt.figure(1)
        plt.plot(range(1, self.epochs+1), self.train_mae_loss_list,
                 '-r', label="Training data")
        plt.plot(range(1, self.epochs+1), self.val_mae_loss_list,
                 '-g', label="Validation data")
        plt.gca().set_xlabel("Epochs")
        plt.gca().set_ylabel("log10 Mean Absolute Error")
        plt.gca().set_title("log10 MAE Loss")
        plt.legend()
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "log10mae_loss.png"), dpi=self.dpi)
        plt.clf()
        # plot log10 MSE loss
        plt.figure(2)
        plt.plot(range(1, self.epochs+1), self.train_mse_loss_list,
                 '-r', label="Training data")
        plt.plot(range(1, self.epochs+1), self.val_mse_loss_list,
                 '-g', label="Validation data")
        plt.gca().set_xlabel("Epochs")
        plt.gca().set_ylabel("log10 Mean Squared Error")
        plt.gca().set_title("log10 MSE Loss")
        plt.legend()
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "log10mse_loss.png"), dpi=self.dpi)
        plt.clf()

    def analyze_error_dist(self):
        '''
        This function analyzes the error distribution of the model,
        and saves the plots of the error distribution.
        '''
        print("Analyzing distribution...")
        model = self.get_best_model()
        # Generate itemwise training errors using model
        model.eval()
        train_mae_loss = []
        train_reg_loss = []
        train_pred = []
        train_label = []
        for image, label, weight, path in tqdm(self.train_eval_loader, desc="Train data: "):
            # Send variables to device
            image = image.to(device)
            label = label.to(device)
            # Run model
            pred = model(image)
            # Move variables to cpu
            pred = pred.cpu().squeeze().detach()
            label = label.cpu().squeeze().detach()
        #     print(pred.shape)
        #     print(abs_err(pred,label))
            # Compute loss
            train_mae_loss = np.concatenate(
                (train_mae_loss, abs_err(pred, label)))
            train_reg_loss = np.concatenate(
                (train_reg_loss, reg_err(pred, label)))
            train_pred = np.concatenate((train_pred, pred))
            train_label = np.concatenate((train_label, label))
        # Get training data error distribution
        plt.figure(1)
        sns.histplot(train_reg_loss)  # , kde=True)
        plt.gca().set_title("Training data log10 error distribution")
        plt.gca().set_xlabel("log10 error")
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "train_err_dist.png"), dpi=self.dpi)
        plt.clf()
        # Get training data abs error distrubtion
        plt.figure(2)
        # Plot absolute error (log) distribution for model
        sns.histplot(train_mae_loss)  # , kde=True)
        plt.gca().set_title("Training data log10 absolute error distribution")
        plt.gca().set_xlabel("log10 absolute error")
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "train_abserr_dist.png"), dpi=self.dpi)
        plt.clf()

        plt.figure(num=3, dpi=100)
        plt.scatter(train_label, train_pred, s=2)
        plt.gca().set_box_aspect(1)
        plt.gca().set_xlim(0, 10)
        plt.gca().set_ylim(0, 10)
        plt.gca().set_xlabel("Actual")
        plt.gca().set_ylabel("Predicted")
        plt.gca().set_title("Training data")
        xpoints = ypoints = plt.gca().get_xlim()
        plt.gca().plot(xpoints, ypoints, linestyle='-',
                       color='k', lw=1, scalex=False, scaley=False)
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "train_pred.png"), dpi=self.dpi)
        plt.clf()

        # Generate itemwise validation errors using model
        model.eval()
        val_mae_loss = []
        val_reg_loss = []
        val_pred = []
        val_label = []

        for image, label, weight, path in tqdm(self.val_loader, desc="Validation data: "):
            # Send variables to device
            image = image.to(device)
            label = label.to(device)
            # Run model
            pred = model(image)
            # Move variables to cpu
            pred = pred.cpu().squeeze().detach()
            label = label.cpu().squeeze().detach()
        #     print(pred.shape)
        #     print(abs_err(pred,label))
            # Compute loss
            val_mae_loss = np.concatenate((val_mae_loss, abs_err(pred, label)))
            val_reg_loss = np.concatenate((val_reg_loss, reg_err(pred, label)))
            val_pred = np.concatenate((val_pred, pred))
            val_label = np.concatenate((val_label, label))

        # get validation data error distribution
        plt.figure(4)
        sns.histplot(val_reg_loss)
        plt.gca().set_title("Validation data log10 error distribution")
        plt.gca().set_xlabel("log10 error")
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "val_err_dist.png"), dpi=self.dpi)
        plt.clf()
        # get validation data abs error distribution
        plt.figure(5)
        sns.histplot(val_mae_loss)
        plt.gca().set_title("Validation data log10 absolute error distribution")
        plt.gca().set_xlabel("log10 absolute error")
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "val_abserr_dist.png"), dpi=self.dpi)
        plt.clf()

        plt.figure(num=6, dpi=100)
        plt.scatter(val_label, val_pred, s=2)
        plt.gca().set_box_aspect(1)
        plt.gca().set_xlim(0, 10)
        plt.gca().set_ylim(0, 10)
        plt.gca().set_xlabel("Actual")
        plt.gca().set_ylabel("Predicted")
        plt.gca().set_title("Validation data")
        xpoints = ypoints = plt.gca().get_xlim()
        plt.gca().plot(xpoints, ypoints, linestyle='-',
                       color='k', lw=1, scalex=False, scaley=False)
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "valid_pred.png"), dpi=self.dpi)
        plt.clf()

        # Evaluate test distribution
        model.eval()
        test_mae_loss = []
        test_reg_loss = []
        test_pred = []
        test_label = []

        for image, label, weight, path in tqdm(self.test_loader, desc="Test data: "):
            # Send variables to device
            image = image.to(device)
            label = label.to(device)
            # Run model
            pred = model(image)
            # Move variables to cpu
            pred = pred.cpu().squeeze().detach()
            label = label.cpu().squeeze().detach()
        #     print(pred.shape)
        #     print(abs_err(pred,label))
            # Compute loss
            test_mae_loss = np.concatenate(
                (test_mae_loss, abs_err(pred, label)))
            test_reg_loss = np.concatenate(
                (test_reg_loss, reg_err(pred, label)))
            test_pred = np.concatenate((test_pred, pred))
            test_label = np.concatenate((test_label, label))
        # get test data error distribution
        plt.figure(7)
        sns.histplot(test_reg_loss)
        plt.gca().set_title("Test data log10 error distribution")
        plt.gca().set_xlabel("log10 error")
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "test_err_dist.png"), dpi=self.dpi)
        plt.clf()
        # get test data abs error distribution
        plt.figure(8)
        sns.histplot(test_mae_loss)
        plt.gca().set_title("Test data log10 absolute error distribution")
        plt.gca().set_xlabel("log10 absolute error")
        plt.savefig(os.path.join(self.__plot_folder_path,
                    "test_abserr_dist.png"), dpi=self.dpi)
        plt.clf()

        plt.figure(num=9, dpi=100)
        plt.scatter(test_label, test_pred, s=2)
        plt.gca().set_box_aspect(1)
        plt.gca().set_xlim(0, 10)
        plt.gca().set_ylim(0, 10)
        plt.gca().set_xlabel("Actual")
        plt.gca().set_ylabel("Predicted")
        plt.gca().set_title("Test data")
        xpoints = ypoints = plt.gca().get_xlim()
        plt.gca().plot(xpoints, ypoints, linestyle='-',
                       color='k', lw=1, scalex=False, scaley=False)
        plt.savefig(os.path.join(self.__plot_folder_path,
                                 "test_pred.png"), dpi=self.dpi)
        plt.clf()

        err = {"train_mae_loss": train_mae_loss.tolist(),
               "val_mae_loss": val_mae_loss.tolist(),
               "test_mae_loss": test_mae_loss.tolist()}
        with open(os.path.join(self.__plot_folder_path, "loss.json"), "w") as f_loss:
            json.dump(err, f_loss)

    def getExample(self, partition=0, idx=0):
        if partition == 0:  # Training
            if idx >= len(self.train_eval_data):
                print("Index out of bounds for Training dataset")
                raise Exception("Out of Bounds")
            return self.train_eval_data[idx]
        elif partition == 1:  # Validation
            if idx >= len(self.val_data):
                print("Index out of bounds for Training dataset")
                raise Exception("Out of Bounds")
            return self.val_data[idx]
        elif partition == 2:  # Test
            if idx >= len(self.test_data):
                print("Index out of bounds for Training dataset")
                raise Exception("Out of Bounds")
            return self.test_data[idx]
        elif partition == 3:
            if idx >= len(self.all_data):
                print("Index out of bounds for all dataset")
                raise Exception("Out of Bounds")
            return self.all_data[idx]
        print("Incorrect partition selected, 0 for training, 1 for validation, 2 for test.")
        raise Exception("Incorrect dataset chosen")

    # Test

    def clear_cache(self):
        torch.cuda.empty_cache()
