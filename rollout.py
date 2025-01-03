import copy
import numpy as np
import torch

class Rollout(object):

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    # 这里的x是将生成器生成的一个序列放进去，num是个超参数
    def get_reward(self, x, num, discriminator):

        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):

                additional_seq_len = 20 - l
                # 先行全都要，然后列只要0-l列，然后通过生成器生成补全后面的序列
                initial_seq = x[:, 0:l, :]
                initial_seq_tensor = initial_seq.cuda()
                initial_seq_tensor = initial_seq_tensor[:, :, 0]

                wind_speed_data = x[:, :, 1]
                wind_speed_data_tensor = wind_speed_data.cuda()
                wind_speed_data_tensor = wind_speed_data_tensor[:, :, None]

                # 通过copy的own_model，不去影响原始的模型，后面才去更新
                samples = self.own_model.sample_batch(batch_size, additional_seq_len, wind_speed_data_tensor, x=initial_seq_tensor)
                # 放入判别器中获得reward，这个过程就是mc search
                with torch.no_grad():
                    pred = discriminator(samples)
                pred = pred.cpu().data[:, 1].numpy()
                #pred = pred.cpu().data.numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l - 1] += pred

            # for the last token
            x_pre = x[:, :, 0]
            with torch.no_grad():
                pred = discriminator(x_pre)
            pred = pred.cpu().data[:, 1].numpy()
            #pred = pred.cpu().data.numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len - 1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num)
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
                # update_rate介于0-1之间的超参数，控制新旧参数值的融合比例，自有模型的参数将会是其当前值和原始模型中相应参数值的加权平均，旨在平衡保持学习到的知识（para.data）和引入原始模型的知识（dis[name]）之间的关系

                # 这种更新策略允许模型在学习新任务或数据时，仍然保留一部分之前学习到的知识。对于嵌入层参数的直接复制保证了词嵌入的稳定性，而对于其他参数的加权更新则提供了一种灵活的方式来控制新旧知识之间的平衡。这种方法在许多场景下都很有用，特别是在需要细粒度控制参数更新程度的迁移学习和增量学习场景中。
                #
