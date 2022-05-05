from __future__ import division, print_function

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.utils import shuffle
from torch.distributions.beta import Beta
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from src.ageclocks.blood.deepctrl.utils import (get_perturbed_input,
                                                verification, custom_slope_loss)


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=4):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(
        self,
        output_dim,
        rule_encoder,
        data_encoder,
        hidden_dim=4,
        n_layers=2,
        merge="cat",
        skip=False,
        input_type="state",
    ):
        super(Net, self).__init__()
        self.skip = skip
        self.input_type = input_type
        self.rule_encoder = rule_encoder  # Pass object of class Encoder
        self.data_encoder = data_encoder  # Pass object of class Encoder
        self.n_layers = n_layers
        assert self.rule_encoder.input_dim == self.data_encoder.input_dim
        assert self.rule_encoder.output_dim == self.data_encoder.output_dim
        self.merge = merge
        if merge == "cat":
            self.input_dim_decision_block = self.rule_encoder.output_dim * 2
        elif merge == "add":
            self.input_dim_decision_block = self.rule_encoder.output_dim

        self.net = []
        for i in range(n_layers):
            if i == 0:
                in_dim = self.input_dim_decision_block
            else:
                in_dim = hidden_dim

            if i == n_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim

            self.net.append(nn.Linear(in_dim, out_dim))
            if i != n_layers - 1:
                self.net.append(nn.ReLU())

        self.net = nn.Sequential(*self.net)

    def get_z(self, x, alpha=0.0):
        rule_z = self.rule_encoder(x)
        data_z = self.data_encoder(x)

        if self.merge == "add":
            z = alpha * rule_z + (1 - alpha) * data_z
        elif self.merge == "cat":
            z = torch.cat((alpha * rule_z, (1 - alpha) * data_z), dim=-1)
        elif self.merge == "equal_cat":
            z = torch.cat((rule_z, data_z), dim=-1)

        return z

    def forward(self, x, alpha=0.0):
        # merge: cat or add

        rule_z = self.rule_encoder(x)
        data_z = self.data_encoder(x)

        if self.merge == "add":
            z = alpha * rule_z + (1 - alpha) * data_z
        elif self.merge == "cat":
            z = torch.cat((alpha * rule_z, (1 - alpha) * data_z), dim=-1)
        elif self.merge == "equal_cat":
            z = torch.cat((rule_z, data_z), dim=-1)

        if self.skip:
            if self.input_type == "seq":
                return self.net(z) + x[:, -1, :]
            else:
                return self.net(z) + x  # predict delta values
        else:
            return self.net(z)  # predict absolute values


class DeepCTRLModel:
    def __init__(
        self,
        input_dim=5,
        merge="add",
        hidden_enc_dim=16,
        output_enc_dim=8,
        hidden_db_dim=8,
        output_db_dim=1,
        n_layers_db=2,
        alpha_training_distribution=Beta(0.1, 0.1),
        lr=0.001,
        pert_coeff=0.1,
        scale=1.0,  # To favorize task loss regarding of alpha,
        device="cpu:0",
        seed=42,
        rule_losses=[""],
        rule_indices=[2],
        bool_perturbed=[True],
        epochs=5,
    ):
        super(DeepCTRLModel, self).__init__()
        self.input_dim = input_dim
        self.merge = merge
        self.hidden_enc_dim = hidden_enc_dim
        self.output_enc_dim = output_enc_dim
        self.hidden_db_dim = hidden_db_dim
        self.output_db_dim = output_db_dim
        self.n_layers_db = n_layers_db
        self.rule_encoder = Encoder(
            self.input_dim, self.output_enc_dim, self.hidden_enc_dim
        )
        self.data_encoder = Encoder(
            self.input_dim, self.output_enc_dim, self.hidden_enc_dim
        )
        self.alpha_training_distribution = alpha_training_distribution
        self.lr = lr
        self.epochs = epochs
        self.pert_coeff = pert_coeff
        self.scale = scale
        self.device = device

        self.rule_losses = rule_losses
        self.rule_indices = rule_indices
        self.bool_perturbed = bool_perturbed

        assert len(rule_indices) == len(rule_indices) == len(bool_perturbed)

        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = Net(
            self.output_db_dim,
            self.rule_encoder,
            self.data_encoder,
            self.hidden_db_dim,
            self.n_layers_db,
            self.merge,
        ).to(self.device)

        print(summary(self.model, input_size=(self.input_dim,)))

    def _load_dataset(self, X, Y, validation_split=0.2, batch_size=32):
        # We assume the user has already split the dataset
        # into train/test and therefore we will only use validation split.
        X = np.array(X)
        X_np = X.copy()
        train_X, valid_X, train_y, valid_y = train_test_split(
            X_np, Y.to_numpy(), test_size=validation_split, random_state=self.seed)

        train_X, train_y = torch.tensor(
            train_X, dtype=torch.float32, device=self.device
        ), torch.tensor(train_y, dtype=torch.float32, device=self.device)
        valid_X, valid_y = torch.tensor(
            valid_X, dtype=torch.float32, device=self.device
        ), torch.tensor(valid_y, dtype=torch.float32, device=self.device)

        train_loader = DataLoader(
            TensorDataset(
                train_X,
                train_y),
            batch_size=batch_size,
            shuffle=True)
        valid_loader = DataLoader(
            TensorDataset(valid_X, valid_y), batch_size=valid_X.shape[0]
        )
        return train_loader, valid_loader

    def fit(self, X, Y, validation_split=0.2, batch_size=32):

        train_loader, valid_loader = self._load_dataset(
            X, Y, validation_split=validation_split, batch_size=batch_size
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        loss_task_func = nn.MSELoss()
        loss_rule_func = self.rule_losses[0]

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch_train_x, batch_train_y in train_loader:

                batch_train_y = batch_train_y.unsqueeze(-1)

                optimizer.zero_grad()

                alpha = self.alpha_training_distribution.sample().item()
                # alpha=1.0

                output = self.model(batch_train_x, alpha=alpha)

                loss_task = loss_task_func(output, batch_train_y)

                pert_batch_train_x = batch_train_x.detach().clone()
                pert_batch_train_x[:, self.rule_indices[0]] = get_perturbed_input(
                    pert_batch_train_x[:, self.rule_indices[0]], self.pert_coeff)
                pert_output = self.model(pert_batch_train_x, alpha=alpha)

                loss_rule = custom_slope_loss(
                    batch_train_x, pert_batch_train_x, output, pert_output
                )

                loss = alpha * loss_rule + self.scale * (1 - alpha) * loss_task

                loss.backward()
                optimizer.step()

        with torch.no_grad():
            for val_x, val_y in valid_loader:
                val_y = val_y.unsqueeze(-1)

                output = self.model(val_x, alpha=alpha)
                val_loss_task = loss_task_func(output, val_y).item()

                # perturbed input and its output
                pert_val_x = val_x.detach().clone()
                pert_val_x[:, self.rule_indices[0]] = get_perturbed_input(
                    pert_val_x[:, self.rule_indices[0]], self.pert_coeff
                )
                pert_output = self.model(
                    pert_val_x, alpha=alpha
                )  # \hat{y}_{p}    predicted sales from perturbed input

                # val_loss_rule = loss_rule_func(output, pert_output).item()
                val_loss_rule = custom_slope_loss(
                    val_x, pert_val_x, output, pert_output
                )

                val_loss = val_loss_task
                print("Task: ", val_loss_task, "\nRule: ", val_loss_rule)

    def predict(self, sample, alpha=0.0):
        # Transform sample into tensor
        sample_tensor = torch.tensor(
            np.array(sample), dtype=torch.float32, device=self.device
        )
        return (self.model(sample_tensor, alpha=alpha)).detach().numpy()


if __name__ == "__main__":

    df = pd.read_csv(
        "/home/filip/IT/Longevize/machine-learning/src/ageclocks/blood/NHANES/data/nhanesModified/nhanes_concatenated_olderThan17.csv"
    )

    print(df.columns)

    features = [
        "albumin",
        "hbA1c%",
        "cholesterol",
        "SHBG",
        "urea",
        "apolipoproteinB",
        "gender_male",
        "creatinine",
    ]

    larsen_vals = [54.0, 5.7, 2.85, 13.6, 4.5, 1.01, 1.0, 101]

    df = df.dropna(how="any", subset=features)

    model = DeepCTRLModel(
        scale=2,
        epochs=100,
        input_dim=len(features),
        alpha_training_distribution=Beta(0.62, 0.16),
    )

    X_raw = df[features]
    Y = df["age"]

    train_X, test_X, train_y, test_y = train_test_split(
        X_raw, Y, test_size=0.05)

    model.fit(X=train_X, Y=train_y)

    torch.save(model, "/home/filip/IT/Longevize/machine-learning/src/ageclocks/blood/deepctrl/saved_models/model1")

    # print(f"{len(test_y)} are in test group")
    # person_idx = 125

    # print("Parameters: ", test_X.iloc[person_idx])
    # print("Actual age: ", test_y.iloc[person_idx])

    for alpha10 in range(0, 11, 1):
        alpha = alpha10 / 10
        print(
            f"Prediction using alpha of {alpha}:",
            model.predict(larsen_vals, alpha=alpha),
        )

    # preds = model.predict(test_X, alpha=1.0)

    from sklearn.metrics import mean_absolute_error

    # print(mean_absolute_error(preds, test_y))

    print(model.predict(larsen_vals))

    for add_chol in range(1, 10, 1):
        # current_vals = test_X.iloc[person_idx]
        current_vals = larsen_vals
        # current_vals[features.index("cholesterol")] += add_chol / 10
        current_vals[2] = current_vals[2] + 0.1
        print(current_vals[2], model.predict(current_vals, alpha=1.0))
        print("-----")
