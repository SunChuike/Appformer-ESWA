from data.data_loader import Dataset_Tsinghua
from exp.exp_basic import Exp_Basic
from models.model import Informer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from utils.tools import EarlyStopping, adjust_learning_rate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import warnings
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        self.app_out = args.app_out
        self.k = 5
        self.alpha = 0.1
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _build_model(self):
        model_dict = {
            'informer': Informer,
        }
        if self.args.model == 'informer' :
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.app_dim,
                self.args.enc_in,
                self.args.dec_in,
                self.args.app_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, user=None):
        args = self.args

        data_dict = {
            'TSapp': Dataset_Tsinghua
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            flag = 'test_pri'
            args.data_path = 'auxiliary/' + user + '/25'
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        if user != None:
            print('Analysing user_id', user, ' type:', flag, len(data_set))
        else:
            print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def txt2dict(self, path):
        dic = {}
        f = open(path, 'r')
        samples = f.readlines()
        for line in samples:
            sample = line.split('\t')
            dic[sample[0]] = sample[1].split('\n')[0]
        f.close()
        return dic

    def aux_model(self, aux_data):

        model = MultinomialNB()
        label = aux_data['target']
        features = aux_data.drop(['target'], axis=1)
        model.fit(features, label)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def calculate_loss(self, pred, true):
        a = pred.contiguous()
        b = true.contiguous().view(-1)
        loss = F.cross_entropy(pred.contiguous(), true.contiguous().view(-1), reduction='none')
        loss = torch.mean(loss)
        return loss

    def calculate_scores(self, predicted_scores, true_labels):
        predicted_labels = predicted_scores.cpu().numpy()
        true_labels = true_labels.cpu().numpy()

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='macro'
        )

        return precision, recall, f1_score

    def MRR(self, predicted_scores, true_labels):
        batch_size = predicted_scores.size(0)
        _, ranked_positions = torch.sort(predicted_scores, dim=1, descending=True)
        position_of_true_labels = (ranked_positions == true_labels).nonzero()[:, 1] + 1
        reciprocal_ranks = 1.0 / position_of_true_labels.float()
        mrr = reciprocal_ranks.sum() / batch_size

        return mrr

    def _process_one_batch(self, dataset_object, app_seq, time_seq, location_vectors, user):  # 每个一个batch的训练、验证或者测试
        app_seq = app_seq.long().to(self.device)
        time_seq = time_seq.to(self.device)
        location_vectors = location_vectors.float().to(self.device)
        user = user.to(self.device)

        outputs_app = self.model(app_seq, time_seq, location_vectors, user)

        if self.args.inverse:
            outputs_app = dataset_object.inverse_transform(outputs_app)
        f_dim = -1 if self.args.features == 'MS' else 0
        app_seq = app_seq[:, -self.args.pred_len:, f_dim:].to(self.device)
        app_seq = torch.squeeze(app_seq, dim=2)

        return outputs_app, app_seq

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (app_seq, time_seq, location_vectors, user) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred_app, true = self._process_one_batch(train_data, app_seq, time_seq, location_vectors, user)

                app_loss = self.calculate_loss(pred_app, true)
                train_loss.append(app_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | app_loss: {2:.7f}"
                          .format(i + 1, epoch + 1, app_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:

                    scaler.scale(app_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    app_loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss, vali_acc_1, vali_acc_3, vali_acc_5, precisions, recalls, f1 = self.vali(vali_data, vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            print('Vali Acc_1: ', vali_acc_1, 'Vali Acc_3: ', vali_acc_3, 'Vali Acc_5: ', vali_acc_5)
            print('precision: ', precisions, 'recall: ', recalls, 'f1: ', f1)
            print('--------------------------------------------------------------------------')

            vali_acc_1 = -np.float32(vali_acc_1.item())
            early_stopping(vali_acc_1, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        accuracys_1 = []
        accuracys_3 = []
        accuracys_5 = []
        precision_batch = []
        recall_batch = []
        f1_batch = []
        mrrs = []
        for i, (app_seq, time_seq, location_vectors, user) in enumerate(vali_loader):
            pred_app, true = self._process_one_batch(vali_data, app_seq, time_seq, location_vectors, user)

            app_loss = self.calculate_loss(pred_app, true)
            app_loss = app_loss.detach().cpu()
            total_loss.append(app_loss)

            app_loss = app_loss.detach().cpu()
            total_loss.append(app_loss)

            pred = torch.topk(pred_app, dim=1, k=self.k).indices
            accuracy_1 = torch.eq(pred[:, 0:1], true).squeeze(-1)
            accuracy_3 = torch.eq(pred[:, 0:3], true)
            accuracy_3 = torch.sum(accuracy_3, dim=-1)
            accuracy_5 = torch.eq(pred[:, 0:5], true)
            accuracy_5 = torch.sum(accuracy_5, dim=-1)

            batch = len(accuracy_1)
            accuracy_1 = torch.sum(accuracy_1, dim=0).float() / batch
            accuracy_3 = torch.sum(accuracy_3, dim=0).float() / batch
            accuracy_5 = torch.sum(accuracy_5, dim=0).float() / batch

            accuracy_1 = accuracy_1.detach().cpu()
            accuracy_3 = accuracy_3.detach().cpu()
            accuracy_5 = accuracy_5.detach().cpu()
            accuracys_1.append(accuracy_1.numpy())
            accuracys_3.append(accuracy_3.numpy())
            accuracys_5.append(accuracy_5.numpy())

            _, predicted_labels = torch.max(pred_app, dim=1)
            precision, recall, f1_score = self.calculate_scores(predicted_labels, true)

            precision = precision.astype(np.float32)
            precision_batch.append(precision)

            recall = recall.astype(np.float32)
            recall_batch.append(recall)

            f1_score = f1_score.astype(np.float32)
            f1_batch.append(f1_score)

            mrr = self.MRR(pred_app, true)
            mrr = mrr.detach().cpu().numpy()
            mrrs.append(mrr)

        precisions = np.mean(precision_batch)
        recalls = np.mean(recall_batch)
        f1 = np.mean(f1_batch)

        total_accuracy_1 = np.average(accuracys_1, axis=0)
        total_accuracy_3 = np.average(accuracys_3, axis=0)
        total_accuracy_5 = np.average(accuracys_5, axis=0)
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss, total_accuracy_1, total_accuracy_3, total_accuracy_5, precisions, recalls, f1  # 返回整个val数据集的loss以及Top-1/5的准确率

    def test(self, setting):

        vali_data, vali_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=device))
        self.model.eval()

        total_loss = []
        accuracys_1 = []
        accuracys_3 = []
        accuracys_5 = []
        precision_batch = []
        recall_batch = []
        f1_batch = []
        mrrs = []
        for i, (app_seq, time_seq, location_vectors, user) in enumerate(vali_loader):
            pred_app, true = self._process_one_batch(vali_data, app_seq, time_seq, location_vectors, user)

            app_loss = self.calculate_loss(pred_app, true)
            app_loss = app_loss.detach().cpu()
            total_loss.append(app_loss)

            pred = torch.topk(pred_app, dim=1, k=self.k).indices

            Top1 = pred[:, 0:1]
            Top1_cpu = Top1.cpu()
            numpy_array = Top1_cpu.numpy()
            with open('data/Tsinghua_new/time_division/division/Top1.txt', 'a') as f:
                np.savetxt(f, numpy_array, delimiter=' ', fmt='%d')

            accuracy_1 = torch.eq(pred[:, 0:1], true).squeeze(-1)
            accuracy_3 = torch.eq(pred[:, 0:3], true)
            accuracy_3 = torch.sum(accuracy_3, dim=-1)
            accuracy_5 = torch.eq(pred[:, 0:5], true)
            accuracy_5 = torch.sum(accuracy_5, dim=-1)

            batch = len(accuracy_1)
            accuracy_1 = torch.sum(accuracy_1, dim=0).float() / batch
            accuracy_3 = torch.sum(accuracy_3, dim=0).float() / batch
            accuracy_5 = torch.sum(accuracy_5, dim=0).float() / batch

            accuracy_1 = accuracy_1.detach().cpu()
            accuracy_3 = accuracy_3.detach().cpu()
            accuracy_5 = accuracy_5.detach().cpu()
            accuracys_1.append(accuracy_1.numpy())
            accuracys_3.append(accuracy_3.numpy())
            accuracys_5.append(accuracy_5.numpy())

            _, predicted_labels = torch.max(pred_app, dim=1)
            precision, recall, f1_score = self.calculate_scores(predicted_labels, true)

            precision = precision.astype(np.float32)
            precision_batch.append(precision)

            recall = recall.astype(np.float32)
            recall_batch.append(recall)

            f1_score = f1_score.astype(np.float32)
            f1_batch.append(f1_score)

            mrr = self.MRR(pred_app, true)
            mrr = mrr.detach().cpu().numpy()
            mrrs.append(mrr)

        total_mrr = np.average(mrrs, axis=0)
        precisions = np.mean(precision_batch)
        recalls = np.mean(recall_batch)
        f1 = np.mean(f1_batch)

        total_accuracy_1 = np.average(accuracys_1, axis=0)
        total_accuracy_3 = np.average(accuracys_3, axis=0)
        total_accuracy_5 = np.average(accuracys_5, axis=0)
        total_loss = np.average(total_loss)

        print('Acc_1: ', total_accuracy_1, 'Acc_3: ', total_accuracy_3, 'Acc_5: ', total_accuracy_5)
        print('--------------------------------------------------------------------------')
        print('total_loss: ', total_loss, 'mrr: ', total_mrr, 'precision: ', precisions, 'recall: ', recalls, 'best_f1: ', f1)

        return total_loss, total_accuracy_1, total_accuracy_3, total_accuracy_5, total_mrr, precisions, recalls, f1
