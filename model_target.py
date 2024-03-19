import torch
import torch.nn as nn
from models import f_domain,f_common,Discriminator,Export_Gap


class TinySleepNet(nn.Module):
    def __init__(self, config):
        super(TinySleepNet, self).__init__()
        self.sharedNet = f_common()

        self.sonnet0 = f_domain()
        self.sonnet1 = f_domain()
        self.sonnet_mix = f_domain()

        self.rnn_son0 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.rnn_son1 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.rnn_sonmix = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)

        self.fc_son0 = nn.Linear(128, 5)
        self.fc_son1 = nn.Linear(128, 5)
        self.fc_sonmix = nn.Linear(128, 5)

        self.discriminator0 = Discriminator()
        self.discriminator1 = Discriminator()
        self.export0 = Export_Gap()
        self.export1 = Export_Gap()

        self.cls_loss = nn.CrossEntropyLoss(reduce=False)

        self.optimizer = torch.optim.Adam(list(self.sharedNet.parameters())+list(self.sonnet0.parameters())+list(self.sonnet1.parameters())\
                                          +list(self.sonnet_mix.parameters())+list(self.rnn_son0.parameters())+list(self.rnn_son1.parameters())\
                                          +list(self.rnn_sonmix.parameters())+list(self.fc_son0.parameters())+list(self.fc_son1.parameters())\
                                          +list(self.fc_sonmix.parameters())+list(self.export0.parameters())+list(self.export1.parameters()), lr=config["learning_rate"])

        self.optimizer_dis0 = torch.optim.Adam(self.discriminator0.parameters(), lr=config["learning_rate"])
        self.optimizer_dis1 = torch.optim.Adam(self.discriminator1.parameters(), lr=config["learning_rate"])


    def getcls_loss(self,pred,label,sw):
        # weight by sample
        cls_loss = self.cls_loss(pred,label)
        loss = torch.mul(cls_loss, sw).cuda()

        # Weight by class
        one_hot = torch.zeros(len(label), 5).cuda().scatter_(1, label.unsqueeze(dim=1), 1)

        sample_weight = torch.mm(one_hot, torch.Tensor([1, 1.5, 1, 1, 1]).unsqueeze(dim=1).cuda()).view(
                -1)
        loss = torch.mul(loss, sample_weight).sum().cuda() / sw.sum().cuda()

        return loss

    def getdisloss(self,pred,label,sw):
        cls_loss = self.cls_loss(pred, label)
        loss = torch.mul(cls_loss, sw).cuda()
        loss = loss.sum().cuda() / sw.sum().cuda()

        return loss

    def train_all(self):
        self.sharedNet.train()
        self.sonnet0.train()
        self.sonnet1.train()
        self.sonnet_mix.train()
        self.rnn_son0.train()
        self.rnn_son1.train()
        self.rnn_sonmix.train()
        self.fc_son0.train()
        self.fc_son1.train()
        self.fc_sonmix.train()
        self.discriminator0.train()
        self.discriminator1.train()
        self.export0.train()
        self.export1.train()

    def eval_all(self):
        self.sharedNet.eval()
        self.sonnet0.eval()
        self.sonnet1.eval()
        self.sonnet_mix.eval()
        self.rnn_son0.eval()
        self.rnn_son1.eval()
        self.rnn_sonmix.eval()
        self.fc_son0.eval()
        self.fc_son1.eval()
        self.fc_sonmix.eval()
        self.discriminator0.eval()
        self.discriminator1.eval()
        self.export0.eval()
        self.export1.eval()


    def train_da(self, data_src, data_tgt, label_src, mark, state_s, state_t,sw,tw):#,total_loss4_0,total_loss4_1,total_loss0,total_loss1

        self.train_all()

        data_src = self.sharedNet(data_src)
        data_tgt = self.sharedNet(data_tgt)

        data = self.sonnet_mix(data_src)
        dim = data.shape[1]
        data = data.view(-1, 20, dim)
        pred_src_son4, state_s[mark][1] = self.rnn_sonmix(data, state_s[mark][1])
        pred_src_son4_fea = pred_src_son4.reshape(-1, 128)
        pred_src_son4 = self.fc_sonmix(pred_src_son4_fea)
        cls_loss4 = self.getcls_loss(pred_src_son4, label_src, sw)

        if mark == 0:

            for param in self.discriminator0.parameters():
                param.requires_grad = True
            data_src = self.sonnet0(data_src)
            dim = data_src.shape[1]
            data_src = data_src.view(-1, 20, dim)
            pred_src, state_s[mark][0] = self.rnn_son0(data_src, state_s[mark][0])
            pred_src_fea = pred_src.reshape(-1, 128)


            data_tgt_son0 = self.sonnet0(data_tgt)  # 用这个做lmmd loss
            dim = data_tgt_son0.shape[1]
            data_tgt_rnn0 = data_tgt_son0.view(-1, 20, dim)
            pred_tgt_son0, state_t[mark][0] = self.rnn_son0(data_tgt_rnn0, state_t[mark][0])
            pred_tgt_son0 = pred_tgt_son0.reshape(-1, 128)

            pred_src_new = pred_src_fea.detach()
            pred_tgt_new = pred_tgt_son0.detach()

            # predict the domain label by the discirminator network
            pred_dis0 = self.discriminator0(pred_src_new)
            pred_dis1 = self.discriminator0(pred_tgt_new)

            # prepare real labels for the training the discriminator
            disc_src0_labels = torch.ones(pred_src_fea.size(0)).long().cuda()
            disc_src1_labels = torch.zeros(pred_tgt_son0.size(0)).long().cuda()

            # Discriminator Loss
            loss_disc0 = self.getdisloss(pred=pred_dis0, label=disc_src0_labels,sw=sw)
            loss_disc1 = self.getdisloss(pred=pred_dis1, label=disc_src1_labels,sw=tw)
            loss_disc = loss_disc1 + loss_disc0

            self.optimizer_dis0.zero_grad()
            loss_disc.backward()
            self.optimizer_dis0.step()

            for param in self.discriminator0.parameters():
                param.requires_grad = False

            pred_dis0_0 = self.discriminator0(pred_src_fea)
            pred_dis1_0 = self.discriminator0(pred_tgt_son0)

            # prepare false labels for the training the discriminator
            disc_src_labels_false = torch.zeros(pred_src_fea.size(0)).long().cuda()
            disc_tgt_labels_false = torch.ones(pred_tgt_son0.size(0)).long().cuda()

            # Discriminator Loss

            loss_disc0_false = self.getdisloss(pred=pred_dis0_0, label=disc_src_labels_false,sw=sw)
            loss_disc1_false = self.getdisloss(pred=pred_dis1_0, label=disc_tgt_labels_false,sw=tw)
            loss_disc_false = loss_disc1_false + loss_disc0_false

            pred_src = self.fc_son0(pred_src_fea)
            cls_loss = self.getcls_loss(pred_src, label_src, sw)

            # 专家loss
            pred_weight = self.export0(pred_src_son4_fea,pred_src_fea)
            pred_weight0, pred_weight1 = torch.split(pred_weight, (1, 1), dim=1)
            pred_all = pred_weight0 * pred_src_son4  + pred_weight1 * pred_src
            cls_loss_expert = self.getcls_loss(pred_all, label_src, sw)


            for j in range(2):
                for z in range(2):
                    state_s[j][z] = (state_s[j][z][0].detach(), state_s[j][z][1].detach())  # 不更新state_s和state_t
                    state_t[j][z] = (state_t[j][z][0].detach(), state_t[j][z][1].detach())


            cnn_weights = [parm for name, parm in self.named_parameters() if 'conv' in name]
            reg_loss = 0
            for p in cnn_weights:
                reg_loss += torch.sum(p ** 2) / 2
            reg_loss = 2*1e-3 * reg_loss

            total_loss = cls_loss + cls_loss4 + loss_disc_false + reg_loss + cls_loss_expert


            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.sharedNet.parameters()) + list(self.sonnet0.parameters()) + list(self.sonnet1.parameters()) \
                + list(self.sonnet_mix.parameters()) + list(self.rnn_son0.parameters()) + list(
                    self.rnn_son1.parameters()) \
                + list(self.rnn_sonmix.parameters()) + list(self.fc_son0.parameters()) + list(self.fc_son1.parameters()) \
                + list(self.fc_sonmix.parameters()) + list(self.export0.parameters()) + list(self.export1.parameters()),
                max_norm=1.00, norm_type=2)
            self.optimizer.step()


            return cls_loss4, cls_loss,loss_disc,cls_loss_expert, state_s, state_t

        elif mark == 1:

            for param in self.discriminator1.parameters():
                param.requires_grad = True
            data_src = self.sonnet1(data_src)
            dim = data_src.shape[1]
            data_src = data_src.view(-1, 20, dim)
            pred_src, state_s[mark][0] = self.rnn_son1(data_src, state_s[mark][0])
            pred_src_fea = pred_src.reshape(-1, 128)


            data_tgt_son1 = self.sonnet1(data_tgt)  # 用这个做lmmd loss
            dim = data_tgt_son1.shape[1]
            data_tgt_rnn1 = data_tgt_son1.view(-1, 20, dim)
            pred_tgt_son1, state_t[mark][0] = self.rnn_son1(data_tgt_rnn1, state_t[mark][0])
            pred_tgt_son1 = pred_tgt_son1.reshape(-1, 128)


            pred_src_new = pred_src_fea.detach()
            pred_tgt_new = pred_tgt_son1.detach()
            # predict the domain label by the discirminator network
            pred_dis0 = self.discriminator1(pred_src_new)
            pred_dis1 = self.discriminator1(pred_tgt_new)

            # prepare real labels for the training the discriminator
            disc_src0_labels = torch.ones(pred_src_fea.size(0)).long().cuda()
            disc_src1_labels = torch.zeros(pred_tgt_son1.size(0)).long().cuda()

            # Discriminator Loss
            loss_disc0 = self.getdisloss(pred=pred_dis0, label=disc_src0_labels, sw=sw)
            loss_disc1 = self.getdisloss(pred=pred_dis1, label=disc_src1_labels, sw=tw)
            loss_disc = loss_disc1 + loss_disc0

            self.optimizer_dis1.zero_grad()
            loss_disc.backward()
            self.optimizer_dis1.step()

            for param in self.discriminator1.parameters():
                param.requires_grad = False

            pred_dis0_0 = self.discriminator1(pred_src_fea)
            pred_dis1_0 = self.discriminator1(pred_tgt_son1)

            # prepare false labels for the training the discriminator
            disc_src_labels_false = torch.zeros(pred_src_fea.size(0)).long().cuda()
            disc_tgt_labels_false = torch.ones(pred_tgt_son1.size(0)).long().cuda()

            # Discriminator Loss

            loss_disc0_false = self.getdisloss(pred=pred_dis0_0, label=disc_src_labels_false, sw=sw)
            loss_disc1_false = self.getdisloss(pred=pred_dis1_0, label=disc_tgt_labels_false, sw=tw)
            loss_disc_false = loss_disc1_false + loss_disc0_false

            pred_src = self.fc_son1(pred_src_fea)
            cls_loss = self.getcls_loss(pred_src, label_src, sw)

            pred_weight = self.export1(pred_src_son4_fea, pred_src_fea)
            pred_weight0, pred_weight1 = torch.split(pred_weight, (1, 1), dim=1)
            pred_all = pred_weight0 * pred_src_son4 + pred_weight1 * pred_src
            cls_loss_expert = self.getcls_loss(pred_all, label_src, sw)

            for j in range(2):
                for z in range(2):
                    state_s[j][z] = (state_s[j][z][0].detach(), state_s[j][z][1].detach())  # 不更新state_s和state_t
                    state_t[j][z] = (state_t[j][z][0].detach(), state_t[j][z][1].detach())


            cnn_weights = [parm for name, parm in self.named_parameters() if 'conv' in name]
            reg_loss = 0
            for p in cnn_weights:
                reg_loss += torch.sum(p ** 2) / 2
            reg_loss = 2*1e-3 * reg_loss

            total_loss = cls_loss4 + cls_loss + loss_disc_false + reg_loss + cls_loss_expert

            self.optimizer.zero_grad()
            # cls_tgtloss.backward()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.sharedNet.parameters()) + list(self.sonnet0.parameters()) + list(self.sonnet1.parameters()) \
                + list(self.sonnet_mix.parameters()) + list(self.rnn_son0.parameters()) + list(
                    self.rnn_son1.parameters()) \
                + list(self.rnn_sonmix.parameters()) + list(self.fc_son0.parameters()) + list(self.fc_son1.parameters()) \
                + list(self.fc_sonmix.parameters()) + list(self.export0.parameters()) + list(self.export1.parameters()),
                max_norm=1.00, norm_type=2)
            self.optimizer.step()

            return cls_loss4, cls_loss,loss_disc,cls_loss_expert, state_s, state_t

    def valid(self, data_tgt, state_t):

        self.eval_all()
        with torch.no_grad():
            data = self.sharedNet(data_tgt)

            fea_son0 = self.sonnet0(data)
            fea_son1 = self.sonnet1(data)
            fea_son4 = self.sonnet_mix(data)

            dim = fea_son0.shape[1]
            fea_son0 = fea_son0.view(-1, 20, dim)
            fea_son1 = fea_son1.view(-1, 20, dim)
            fea_son4 = fea_son4.view(-1, 20, dim)

            pred0, state_newt0 = self.rnn_son0(fea_son0, state_t[0])
            pred1, state_newt1 = self.rnn_son1(fea_son1, state_t[1])
            pred4, state_newt4 = self.rnn_sonmix(fea_son4, state_t[2])
            state_t[0] = state_newt0
            state_t[1] = state_newt1
            state_t[2] = state_newt4

            pred0_fea = pred0.reshape(-1, 128)
            pred1_fea = pred1.reshape(-1, 128)
            pred4_fea = pred4.reshape(-1, 128)

            pred0 = self.fc_son0(pred0_fea)
            pred1 = self.fc_son1(pred1_fea)
            pred4 = self.fc_sonmix(pred4_fea)

            pred_weight0 = self.export0(pred4_fea, pred0_fea)
            pred_weight0_0, pred_weight0_1 = torch.split(pred_weight0, (1, 1), dim=1)
            pred_export_0 = pred_weight0_0 * pred4 + pred_weight0_1 * pred0


            pred_weight1 = self.export1(pred4_fea, pred1_fea)
            pred_weight1_0, pred_weight1_1 = torch.split(pred_weight1, (1, 1), dim=1)

            pred_export_1 = pred_weight1_0 * pred4 + pred_weight1_1 * pred1
            weight1 = torch.cat((pred_weight0_1,pred_weight0_0,pred_weight1_1, pred_weight1_0),dim=1)


        return pred0, pred1, pred4, state_t,pred_export_0,pred_export_1,weight1