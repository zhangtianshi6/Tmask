import utils

from transfer_modules import *


class ContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).
    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """

    def __init__(self, embedding_dim, input_dims, hidden_dim,
                 num_objects, hinge=1., sigma=0.5, encoder='large', heads=8, layers=6, trans_model='sub', black_lthr=0.4):
        super(ContrastiveSWM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma

        self.pos_loss = 0
        self.neg_loss = 0
        self.heads = heads
        self.layers = layers
        self.trans_model = trans_model
        self.black_lthr = black_lthr

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects)

        self.transition_model = TransitionGNN(
            inputd=embedding_dim,
            input_dim=embedding_dim,
            num_objects=num_objects,
            heads=self.heads,
            layers=self.layers,
            trans_model=self.trans_model)

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, states, action, next_states, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            sum_diff = 0
            batch_size = states.size(0)
            seq_num = states.size(1)
            num_obj = states.size(2)
            state = states.view(batch_size*seq_num, num_obj, -1)
            perm = np.random.permutation(batch_size*seq_num)
            next_state = state[perm]
            diff = state - next_state
            sum_diff = norm * diff.pow(2).sum(2).mean(1)
        else:
            sum_diff = 0
            pred_trans = self.transition_model(states, next_states)
            diff = states+pred_trans-next_states
            for i in range(diff.size(1)):
                sum_diff += norm * diff[:, i].pow(2).sum(2).mean(1)
            return sum_diff, states+pred_trans

        return sum_diff

    #def contrastive_loss(self, obs, action, next_obs, obs_neg, action_neg, next_obs_neg):
    def forward(self, obs, action, next_obs, obs_neg, action_neg, next_obs_neg):

        batch_size = obs.size(0)
        seq_num = obs.size(1)
        obj_c = obs.size(2)
        obj_h = obs.size(3)
        obj_w = obs.size(4)
        for i in range(seq_num-1):
            next_obs[:, i] = obs[:, i+1]
        obs = obs.view(batch_size*seq_num, obj_c, obj_h, obj_w)
        next_obs = next_obs.view(batch_size*seq_num, obj_c, obj_h, obj_w)
        objs = self.obj_extractor(obs)
        next_objs = self.obj_extractor(next_obs)

        state = self.obj_encoder(objs)
        next_state = self.obj_encoder(next_objs)

        # add  black_loss
        black_loss = objs.mean(3).mean(2)
        zeros = torch.zeros_like(black_loss)
        black_loss = torch.max(black_loss-0.3, zeros).mean(1).mean(0)
        obj_num = state.size(1)
        state = state.view(batch_size, seq_num, obj_num, -1)
        next_state = next_state.view(batch_size, seq_num, obj_num, -1)

        self.pos_loss, pred_trans = self.energy(state, action, next_state)

        neg_batch_size = obs_neg.size(0)
        obs_neg = obs_neg.view(neg_batch_size*seq_num, obj_c, obj_h, obj_w)
        next_obs_neg = next_obs_neg.view(
            neg_batch_size*seq_num, obj_c, obj_h, obj_w)
        objs_neg = self.obj_extractor(obs_neg)
        next_objs_neg = self.obj_extractor(next_obs_neg)
        state_neg = self.obj_encoder(objs_neg)
        next_state_neg = self.obj_encoder(next_objs_neg)
        state_neg = state_neg.view(neg_batch_size, seq_num, obj_num, -1)
        next_state_neg = next_state_neg.view(
            neg_batch_size, seq_num, obj_num, -1)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = self.energy(state_neg, action, next_state_neg, no_trans=True)
        zeros = torch.zeros_like(self.neg_loss)
        self.neg_loss = torch.max(zeros, self.hinge - self.neg_loss).mean()
        loss = self.pos_loss + self.neg_loss + black_loss
        return loss, self.pos_loss, self.neg_loss, black_loss, next_state, pred_trans

    def forward1(self, obs):
        return self.obj_encoder(self.obj_extractor(obs)), 


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""

    def __init__(self, inputd, input_dim, num_objects, act_fn='relu', heads=8, layers=2, trans_model='sub'):
        super(TransitionGNN, self).__init__()

        self.input_dim = inputd
        self.output_dim = input_dim
        self.num_objects = num_objects
        self.heads = heads
        self.layers = layers
        self.batch_size = 0
        if trans_model == 'add':
            self.transform = Transformeradd(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'diradd':
            self.transform = Transformerdiradd(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'trans':
            self.transform = Transformertrans(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'cat':
            self.transform = Transformercat(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'catlstm':
            self.transform = Transformercatlstm(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'elstm':
            self.transform = Transformerelstm(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'lstm':
            self.transform = Transformerlstm(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'subsat':
            self.transform = Transformersubsat(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'subsatall':
            self.transform = Transformersubsatall(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        elif trans_model == 'subunsat':
            self.transform = Transformersubunsat(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)
        else:
            self.transform = Transformersub(self.input_dim, self.output_dim, self.heads, self.layers, self.num_objects)

    def forward(self, states, next_states):
        batch_size = states.size(0)
        seq_num = states.size(1)
        num_nodes = states.size(2)
        node_attr = self.transform(states, next_states)
        node_attr = node_attr.reshape(batch_size, seq_num, num_nodes, -1)
        return node_attr


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))


class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='leaky_relu'):
        super(EncoderCNNMedium, self).__init__()

        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.cnn2(h)
        h = self.act2(h)
        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = utils.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)

