import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import gamma, device, batch_size, sequence_length, burn_in_length, device

class DRQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(DRQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTMCell(input_size=num_inputs, hidden_size=32)
        self.fc = nn.Sequential(
            nn.Linear(32, hidden_size),
            nn.ReLU()
        )
        self.fc_adv = nn.Linear(hidden_size, num_outputs)
        self.fc_val = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None):
        # x [batch_size, num_inputs]
        batch_size = x.size()[0]
        if hidden is not None:
            hx, cx = self.lstm(x, hidden)
        else:
            hx, cx = self.lstm(x)

        out = self.fc(hx)
        adv = self.fc_adv(out)
        adv = adv.view(batch_size, self.num_outputs)
        val = self.fc_val(out)
        val = val.view(batch_size, 1)

        qvalue = val + (adv - adv.mean(dim=1, keepdim=True))

        return qvalue, (hx, cx) 


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        def slice_burn_in(item):
            return item[:, burn_in_length:, :]
        states = torch.stack(batch.state).view(batch_size, sequence_length, online_net.num_inputs).to(device)
        next_states = torch.stack(batch.next_state).view(batch_size, sequence_length, online_net.num_inputs).to(device)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long().to(device)
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1).to(device)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1).to(device)
        rnn_state = torch.stack(batch.rnn_state).view(batch_size, sequence_length, 2, -1).to(device)

        q_from_zero = []
        q_from_rnn_state = []
        next_q_from_zero = []
        next_q_from_zero_online = []

        
        states = states.transpose(0, 1)
        next_states = next_states.transpose(0, 1)
        # states [sequence_length, batch_size, num_inputs]
        rnn_state = rnn_state.transpose(0,1).transpose(1,2)
        # rnn_state [sequence_length, 2, batch_size, hidden]
        [hx, cx] = rnn_state[0].detach()
        new_rnn_state_list = []
        
        for idx in range(sequence_length):
            [ht, ct] = rnn_state[idx].detach()
            qvalue_from_rnn_state, _ = online_net(states[idx], (ht, ct))
            new_rnn_state_list.append(torch.stack([hx, cx]).to(device))
            qvalue_from_zero, (hx, cx) = online_net(states[idx], (hx, cx))
            

            q_from_rnn_state.append(qvalue_from_rnn_state)
            q_from_zero.append(qvalue_from_zero)
            
        [hx, cx] = rnn_state[1].detach()
        for idx in range(sequence_length):
            next_qvalue_from_zero, (hx, cx) = target_net(next_states[idx], (hx, cx))
            next_q_from_zero.append(next_qvalue_from_zero)

        [hx, cx] = rnn_state[1].detach()
        for idx in range(sequence_length):
            next_qvalue_from_zero, (hx, cx) = online_net(next_states[idx], (hx, cx))
            next_q_from_zero_online.append(next_qvalue_from_zero)
        

        q_from_zero = torch.stack(q_from_zero).to(device)
        q_from_rnn_state = torch.stack(q_from_rnn_state).to(device)
        next_q_from_zero = torch.stack(next_q_from_zero).to(device)
        next_q_from_zero_online = torch.stack(next_q_from_zero_online).to(device)

        q_from_zero = q_from_zero.transpose(0, 1)
        q_from_rnn_state = q_from_rnn_state.transpose(0, 1)
        next_q_from_zero = next_q_from_zero.transpose(0, 1)
        next_q_from_zero_online = next_q_from_zero_online.transpose(0, 1)

        q_discrepancy = (q_from_zero - q_from_rnn_state).pow(2) / q_from_zero.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        q_discrepancy = q_discrepancy.mean()
        
        pred = slice_burn_in(q_from_zero)
        next_pred = slice_burn_in(next_q_from_zero)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        next_pred_online = slice_burn_in(next_q_from_zero_online)
        
        pred = pred.gather(2, actions)
        _, next_pred_online_action = next_pred_online.max(2)
        
        target = rewards + masks * gamma * next_pred.gather(2, next_pred_online_action.unsqueeze(2))

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), q_discrepancy.item(), torch.stack(new_rnn_state_list).to(device)

    def get_action(self, state, hidden):
        state = state.unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden)
            
        _, action = torch.max(qvalue, 1)

        return action.cpu().numpy()[0], hidden
