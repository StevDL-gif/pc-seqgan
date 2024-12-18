import torch
import torch.optim as optim
import torch.nn as nn
import sys
import helper
import generator
from math import ceil
from discriminator import Discriminator
from torch.utils.data import Dataset, DataLoader
import rollout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import TargetLSTM
import torch.nn.functional as F

# 参数定义
cuda = True
vocab_size = 5000
batch_size = 64
emb_dim = 32
hidden_dim = 32
seq_len = 20
start_token = 0
TRAIN_EPOCHS = 120
window_size = 20
stride = 10
real_samples_path = 'J00005.csv'
POS_NEG_SAMPLES = 10000
START_LETTER = 0
dis_embedding_dim = 64
# 对抗训练轮数
adv_train_epochs = 10000
num_samples = 100

# 判别器参数
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_dropout = 0.75
d_num_class = 2
rnn_hidden_dim = 64
# 是否为测试
test = True
l2_reg_lambda = 0.01
positive_file = "real_data.txt"
negative_file = "generator_sample.txt"
eval_file = "eval_file.txt"
load_model = False


def generator_samples(model, batch_size, real_sample, real_seq_length=5, gen_seq_length=20, eval_mode=False):
    all_samples = []
    additional_seq_len = gen_seq_length - real_seq_length  # 计算需要额外生成的序列长度

    for batch_index in range(0, len(real_sample), batch_size):
        # 确保索引不越界
        batch_real_samples = real_sample[batch_index:batch_index + batch_size]
        # 获得一个batch中的所有行的前5列
        initial_seq = batch_real_samples[:, :real_seq_length]
        initial_seq_tensor = initial_seq.cuda()
        initial_seq_tensor = initial_seq_tensor[:, :, 0]

        # 生成后续数据时，将风速数据作为条件传入
        # 这里是全部的风速数据，包含后续需要生成的序列
        wind_speed_data = batch_real_samples[:, :, 1]
        wind_speed_data_tensor = wind_speed_data.cuda()
        # 增加一个维度从（64，20）变为（64，20，1）
        wind_speed_data_tensor = wind_speed_data_tensor[:, :, None]  # 增加一个维度，与词嵌入维度匹配

        # 直接传递额外生成的序列长度
        generated_seq = model.sample_batch(batch_size, additional_seq_len, wind_speed_data_tensor, x=initial_seq_tensor,
                                           eval_mode=eval_mode)
        # 将初始序列和生成的序列拼接
        all_samples.extend(generated_seq.cpu().data.numpy().tolist())

    return all_samples


def train_discriminator(discriminator, dis_opt, real_data_samples, gen, eval_mode=False):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    neg_val = generator_samples(gen, batch_size, real_sample, eval_mode=eval_mode)
    # 实例化 Dataset
    real_data_samples = real_data_samples[:, :, 0]
    dataset = helper.CustomDataset(real_data_samples, neg_val)
    # 实例化 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1):
        total_loss = 0
        total_words = 0
        loss_fn = nn.CrossEntropyLoss()
        for i, (data, labels) in enumerate(dataloader):
            clean_data = data.clone()
            data = clean_data.long()
            labels = labels.long()
            data, labels = data.cuda(), labels.cuda()
            labels = labels.contiguous().view(-1)
            out = discriminator.forward(data)
            loss = loss_fn(out, labels)
            dis_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)  # 梯度裁剪
            dis_opt.step()

            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
        return total_loss / len(dataloader)


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C), dtype=torch.bool)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        # one_hot = one_hot.type(torch.ByteTensor)
        loss = torch.masked_select(prob, one_hot)  # 仅筛选出对应目标的概率值      #这里是prob的一个对所有类别的概率预测
        loss = loss * reward
        loss = -torch.sum(loss)
        return loss


def calculate_berier_score(real_data, generated_data):
    """
    计算 Berier 分数。

    参数:
    real_data -- 真实数据的一个批次，numpy 数组。
    generated_data -- 生成的数据，numpy 数组。

    返回:
    Berier 分数。
    """
    # 计算真实数据的期望值
    E_gt_Y = np.mean(real_data, axis=0)

    # 计算 Berier 分数
    S_Berier = np.mean((E_gt_Y - generated_data) ** 2)

    return S_Berier


def generator_mini_batch(model, real_data_samples, batch_size, real_seq_length=5, gen_seq_length=15, eval_mode=False):
    """
    Generate samples based on a randomly selected mini-batch of real data.
    """
    all_samples = []
    # 随机选择真实数据的一个mini-batch
    indices = torch.randperm(len(real_data_samples))[:batch_size]
    batch_real_samples = real_data_samples[indices].cuda()

    # 获取风速数据
    wind_speed = batch_real_samples[:, :20, 1]  # 假定第1维是风速数据

    # 从选定的mini-batch生成假数据
    initial_seq = batch_real_samples[:, :real_seq_length]
    initial_seq = initial_seq[:, :, 0]
    initial_seq_tensor = initial_seq.cuda()
    # 生成后续数据时，将风速数据作为条件传入
    wind_speed_data = batch_real_samples[:, :, 1]
    wind_speed_data_tensor = wind_speed_data.cuda()
    wind_speed_data_tensor = wind_speed_data_tensor[:, :, None]  # 增加一个维度，与词嵌入维度匹配

    generated_seq = model.sample_batch(batch_size, gen_seq_length, wind_speed_data_tensor, x=initial_seq_tensor,
                                       eval_mode=eval_mode)
    full_seq = generated_seq.cpu().data.numpy().tolist()
    all_samples.extend(full_seq)
    return all_samples, wind_speed


def train_discriminator_minibatch(discriminator, dis_opt, real_data_samples, gen, batch_size=batch_size,
                                  eval_mode=False):
    # 随机选择真实数据的一个mini-batch
    indices = torch.randperm(len(real_data_samples))[:batch_size]
    real_batch = real_data_samples[indices].cuda()
    # 生成对应的假数据
    fake_batch = generator_samples(gen, batch_size, real_batch, eval_mode=eval_mode)
    fake_batch = torch.tensor(fake_batch).cuda()
    # fake_batch = fake_batch[:, :, 0]
    dataset = helper.CustomDataset(real_batch[:, :, 0], fake_batch)
    # 实例化 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1):
        total_loss = 0
        total_words = 0
        # loss_fn = nn.NLLLoss(reduction='sum')
        loss_fn = nn.CrossEntropyLoss()
        for i, (data, labels) in enumerate(dataloader):
            clean_data = data.clone()
            data = clean_data.long()
            labels = labels.long()
            data, labels = data.cuda(), labels.cuda()
            labels = labels.contiguous().view(-1)
            out = discriminator.forward(data)
            loss = loss_fn(out, labels)
            if (i + 1) % 1 == 0:  # 每20次迭代打印一次
                print(f"Interation{i + 1}, Loss:{loss.item()}")
            dis_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)  # 梯度裁剪
            dis_opt.step()

            total_loss += loss.item()
            # total_words += data.size(0) * data.size(1)
        return total_loss / len(dataloader)


if __name__ == '__main__':

    real_sample = helper.prepare_data(real_samples_path, window_size, stride)
    print('Real_sample.shape is', real_sample.shape)

    # 将real_sample变为整数  由于emb的原因，最后结果再除100
    real_sample[:, :, 0] *= 100
    real_sample = real_sample.int()

    # 定义oracle网络、生成网络、判别器网络
    oracle = TargetLSTM.TargetLSTM(vocab_size, emb_dim, hidden_dim, seq_len, start_token)
    gen = generator.Generator(emb_dim, hidden_dim, vocab_size, seq_len, gpu=cuda)
    dis = Discriminator(seq_len, d_num_class, vocab_size, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)

    # 使用cuda加速
    if cuda:
        oracle = oracle.cuda()
        gen = gen.cuda()
        real_sample.cuda()
        dis = dis.cuda()

    print('Starting Generator MLE Training...')
    gen_optimizer = optim.RMSprop(gen.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 定义dataset和dataloader
    real_dataset = helper.genDataset(real_sample)
    dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    # TargetLSTM.train_target_lstm(oracle, dataloader, epochs=100)
    # TargetLSTM.save_model(oracle)
    TargetLSTM.load_model(oracle, "target_lstm_model.pth")
    generated_samples = TargetLSTM.generate_samples(oracle, 10000, 20, start_token=0)
    real_sample_test = real_sample[:, :, 0]
    real_statistics = TargetLSTM.calculate_statistics(real_sample_test)
    generated_statistics = TargetLSTM.calculate_statistics(generated_samples)
    print("Real Data Statistics:", real_statistics)
    print("Generated Data Statistics:", generated_statistics)
    # TargetLSTM.plot_statistics(real_statistics, generated_statistics)

    if load_model:
        rollout = rollout.Rollout(gen, 0.9)
        print('#####################################################')
        print('Start Adeversatial Training...\n')
        # 对抗训练过程的loss
        gen_gan_loss = GANLoss().cuda()
        # 对抗训练过程gen的优化器
        gen_gan_optm = optim.Adam(gen.parameters())
        # 生成器的loss
        gen_loss = nn.CrossEntropyLoss().cuda()
        # 判别器的loss
        # dis_loss = nn.NLLLoss(reduction='sum').cuda()
        dis_loss = nn.CrossEntropyLoss().cuda()
        # 判别器的优化器
        dis_optimizer = optim.Adam(dis.parameters())
        gen.load_state_dict(torch.load('generator_epoch_20.pth'))
        dis.load_state_dict(torch.load('discriminator_epoch_20.pth'))
        generated_data = [
            generator_samples(gen, batch_size, real_sample[300:300 + batch_size, :], eval_mode=True)[0] for _ in
            range(100)]
        scaled_generated_data = [[value / 100 for value in sample] for sample in generated_data]
        #     # 缩小 real_sample_batch 中第一个batch的所有值
        scaled_real_sample_batch = real_sample[300:300 + batch_size, :, 0]
        # import pandas as pd
        #
        # df = pd.DataFrame(scaled_generated_data)
        # output_file = 'generated_data_1600_100次.xlsx'
        # # 将 DataFrame 保存到 Excel 文件
        # # 如果需要，可以添加 index=False 来防止写入索引列
        # df.to_excel(output_file, index=False)
        # print("Data saved to Excel file successfully!")

        scaled_real_sample_batch = [value / 100 for value in scaled_real_sample_batch[0]]
        #     # 设置绘图
        # 绘图以比较生成数据和真实数据
        plt.figure(figsize=(9, 6))
        for sample in scaled_generated_data:
            plt.plot(sample, color='blue', alpha=0.2)  # 半透明蓝线表示生成数据
        plt.plot(scaled_real_sample_batch, color='red', linewidth=2, label='Real Data')  # 红线表示真实数据
        average_generated = np.mean(scaled_generated_data, axis=0)
        plt.plot(average_generated, color='yellow', linewidth=2, label='Average Generated Data')  # 黄线表示生成数据的平均值
        plt.legend()
        plt.title('Comparison of Generated and Real Data')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.ylim(0, 30)  # 设置y轴的范围为0到100
        plt.savefig("figure_epoch_test.png")  # 根据需要替换文件名
        plt.show()
        real_stats = {
            'mean': np.mean(scaled_real_sample_batch, axis=0),
            'std': np.std(scaled_real_sample_batch, axis=0),
            'min': np.min(scaled_real_sample_batch, axis=0),
            'max': np.max(scaled_real_sample_batch, axis=0)
        }

        generated_stats = {
            'mean': np.mean(scaled_generated_data, axis=0),
            'std': np.std(scaled_generated_data, axis=0),
            'min': np.min(scaled_generated_data, axis=0),
            'max': np.max(scaled_generated_data, axis=0)
        }

        print("Real Data Statistics:", real_stats)
        print("Generated Data Statistics:", generated_stats)
        #
        scaled_real_sample_batch_means = np.mean(scaled_generated_data, axis=0)
        Smse = np.mean((scaled_real_sample_batch_means - scaled_real_sample_batch) ** 2)
        mse = mean_squared_error(scaled_real_sample_batch, average_generated)
        mae = mean_absolute_error(scaled_real_sample_batch, average_generated)
        #
        print(f"Mean Squared Error: {Smse}")
        print(f"Mean Absolute Error: {mae}")
        #
        correlation = np.corrcoef(scaled_real_sample_batch, scaled_generated_data)[0, 1]
        print(f"Correlation coefficient between real and generated data: {correlation}")
        #
        berier_score = calculate_berier_score(np.array(scaled_real_sample_batch), np.array(average_generated))
        print(f"Berier Score: {berier_score}")



    else:
        # 对生成器预训练
        for epoch in range(10):
            total_loss = 0
            for gen_data, gen_labels in dataloader:
                gen_optimizer.zero_grad()
                gen_data, gen_labels = gen_data.long().cuda(), gen_labels.long().cuda()
                gen_labels = gen_labels[:, :, 0].contiguous().view(-1)
                pred = gen.forward(gen_data)
                loss = criterion(pred, gen_labels)
                loss.backward()
                gen_optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(dataloader)
            print(f'Average loss for epoch {epoch}: {average_loss}')

        # 预训练验证、定性评估
        gen.eval()
        with torch.no_grad():
            # 使用预训练好的生成器验证，生成一组数据
            generated_samples = generator_samples(gen, batch_size, real_sample[128:128 + batch_size, :], eval_mode=True)
            print(f'real samples is{real_sample[128:160, :]}, fake_smale is {generated_samples}')

        print('\nStarting Discriminator Training...')
        dis_optimizer = optim.Adam(dis.parameters(), lr=0.01)
        # pretrain Discriminator
        loss = train_discriminator(dis, dis_optimizer, real_sample, gen, eval_mode=True)
        print(f'pre_dis_loss is {loss}')
        # adversarial Training
        rollout = rollout.Rollout(gen, 0.9)

        print('#####################################################')
        print('Start Adeversatial Training...\n')
        # 对抗训练过程的loss
        gen_gan_loss = GANLoss().cuda()
        # 对抗训练过程gen的优化器
        gen_gan_optm = optim.Adam(gen.parameters(), lr=0.001)
        # 生成器的loss
        gen_loss = nn.CrossEntropyLoss().cuda()
        # 判别器的loss
        # dis_loss = nn.NLLLoss(reduction='sum').cuda()
        dis_loss = nn.CrossEntropyLoss().cuda()
        # 判别器的优化器
        dis_optimizer = optim.Adam(dis.parameters(), lr=0.01)

        for epoch in range(adv_train_epochs):
            print('\n--------\nEPOCH %d\n--------' % (epoch + 1))

            # 训练生成器（1步）
            for i in range(2):
                # 先用生成器生成一组数据
                # 使用mini_batch
                samples, wind_data = generator_mini_batch(gen, real_sample, batch_size)
                samples = torch.tensor(samples).cuda()

                samples_expanded = samples.unsqueeze(2)  # 或者使用 samples[:, :, None]
                wind_data_expanded = wind_data.unsqueeze(2)  # 或者使用 wind_data[:, :, None]

                # 然后，沿着最后一个维度（dim=2）将它们拼接起来
                combined_data = torch.cat([samples_expanded, wind_data_expanded], dim=2)

                zeros_start_token = torch.zeros(combined_data.size(0), 1, combined_data.size(2)).to(combined_data.device)
                inputs = torch.zeros_like(combined_data)
                inputs[:, 1:, :] = combined_data[:, :-1, :]
                target = (samples.data).contiguous().view((-1,))

                # 计算奖励
                reward = rollout.get_reward(combined_data, 16, dis)
                # reward = torch.Tensor(reward)
                rewards = torch.Tensor(reward).contiguous().view((-1,)).cuda()
                # rewards = torch.exp(reward).contiguous().view((-1,)).cuda()
                rewards = rewards.unsqueeze(1)
                prob = gen.forward(inputs)
                probabilities = torch.softmax(prob, dim=-1)
                loss = -torch.mean(torch.log(probabilities) * rewards)
                # loss = gen_gan_loss(prob, target, rewards)
                print(loss)
                print("Before update:", gen.lstm.weight_ih_l0.data[0][0])  # 举例访问权重
                gen_gan_optm.zero_grad()
                loss.backward()
                gen_gan_optm.step()
                print("Before update:", gen.lstm.weight_ih_l0.data[0][0])
            rollout.update_params()

            for _ in range(1):
                train_discriminator_minibatch(dis, dis_optimizer, real_sample, gen, batch_size, eval_mode=True)

            if (epoch + 1) % 20 == 0:
                torch.save(gen.state_dict(), f'generator_epoch_{epoch + 1}.pth')
                torch.save(dis.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')
            if epoch % 10 == 0:
                generated_data = [
                    generator_samples(gen, batch_size, real_sample[128:128 + batch_size, :], eval_mode=True)[1] for _ in
                    range(50)]
                scaled_generated_data = [[value / 100 for value in sample] for sample in generated_data]
                #     # 缩小 real_sample_batch 中第一个batch的所有值
                scaled_real_sample_batch = real_sample[128:128 + batch_size, :, 0]
                scaled_real_sample_batch = [value / 100 for value in scaled_real_sample_batch[1]]
                #     # 设置绘图
                # 绘图以比较生成数据和真实数据
                plt.figure(figsize=(9, 6))
                for sample in scaled_generated_data:
                    plt.plot(sample, color='blue', alpha=0.2)  # 半透明蓝线表示生成数据
                plt.plot(scaled_real_sample_batch, color='red', linewidth=2, label='Real Data')  # 红线表示真实数据
                average_generated = np.mean(scaled_generated_data, axis=0)
                plt.plot(average_generated, color='yellow', linewidth=2, label='Average Generated Data')  # 黄线表示生成数据的平均值
                plt.legend()
                plt.title('Comparison of Generated and Real Data')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.ylim(0, 50)  # 设置y轴的范围为0到100
                plt.savefig(f"figure_epoch_{epoch}.png")  # 根据需要替换文件名
                plt.show()
                real_stats = {
                    'mean': np.mean(scaled_real_sample_batch, axis=0),
                    'std': np.std(scaled_real_sample_batch, axis=0),
                    'min': np.min(scaled_real_sample_batch, axis=0),
                    'max': np.max(scaled_real_sample_batch, axis=0)
                }

                generated_stats = {
                    'mean': np.mean(scaled_generated_data, axis=0),
                    'std': np.std(scaled_generated_data, axis=0),
                    'min': np.min(scaled_generated_data, axis=0),
                    'max': np.max(scaled_generated_data, axis=0)
                }

                print("Real Data Statistics:", real_stats)
                print("Generated Data Statistics:", generated_stats)
            #
                scaled_real_sample_batch_means = np.mean(scaled_generated_data, axis=0)
                Smse = np.mean((scaled_real_sample_batch_means - scaled_real_sample_batch) ** 2)
                mse = mean_squared_error(scaled_real_sample_batch, average_generated)
                mae = mean_absolute_error(scaled_real_sample_batch, average_generated)
            #
                print(f"Mean Squared Error: {Smse}")
                print(f"Mean Absolute Error: {mae}")
            #
                correlation = np.corrcoef(scaled_real_sample_batch, scaled_generated_data)[0, 1]
                print(f"Correlation coefficient between real and generated data: {correlation}")
            #
                berier_score = calculate_berier_score(np.array(scaled_real_sample_batch), np.array(average_generated))
                print(f"Berier Score: {berier_score}")
            #
