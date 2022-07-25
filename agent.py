import random
import numpy as np
import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

random.seed(208)

task_feat_map = {
    'binary.classification': 0,
    'multiclass.classification': 1,
    'multilabel.classification': 2,
    'regression': 3
}


def expand_algorithms(features):
    """Expand algorithms with p."""
    p_list = np.arange(0.1, 1.0, 0.1).tolist()
    newfeats = {}
    for k, v in features.items():
        for _, p in enumerate(p_list):
            key = "{},{}".format(k, p)
            v["p"] = p
            newfeats[key] = v
    return newfeats


def get_data_feature(data):
    """Get data feature."""
    print(data)
    #x1 = float(data['has_missing'])
    x0 = float(task_feat_map.get(data['task']))
    #x1 = float(data['train_num']) * np.random.uniform(low=0.8, high=1.2)
    #x2 = float(data['time_budget']) * np.random.uniform(low=0.8, high=1.2)
    return np.array([x0])


DATA_INPUT_SIZE = 1


class DataNet(nn.Module):
    def __init__(self,
                 input_size=DATA_INPUT_SIZE,
                 hidden_size=10,
                 embed_size=10):
        super(DataNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc(x))
        x = self.fc1(x)
        return x


class AlgoNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=10, embed_size=10):
        super(AlgoNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc(x))
        x = self.fc1(x)
        return x


def compute_alc(metrics, total_time_budget, p=1):
    """Compute ALC.

    Args:
        metrics: (times, scores).
    """
    alc = 0.0

    if total_time_budget < 1e-7:
        return 0.0

    times = metrics.times
    scores = metrics.scores
    time_budget = total_time_budget * p

    for i in range(len(times)):
        time_budget -= times[i]
        if time_budget < 0:
            break
        if i == 0:
            alc += scores[i] * (total_time_budget - times[i])
        elif i > 0:
            alc += (scores[i] - scores[i - 1]) * (total_time_budget - times[i])

    alc /= total_time_budget
    return round(alc, 2)


class Agent():
    """
    RANDOM SEARCH AGENT
    """

    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        self.nA = number_of_algorithms
        datanet = DataNet()
        self.datanet = datanet
        self.algonet = None
        self.iters = 1000
        self.lr = 1e-3

    def get_random_algo_feat(self, featname):
        """Get a random value for featname."""
        if featname not in self.algo_values:
            raise ValueError('featname %s not in algo_values %s' %
                             (featname, self.algo_values))
        return random.choice(self.algo_values[featname])

    def get_algo_feature(self, algo):
        """Get algo feature."""
        f = []
        m = []
        for featname in self.algo_featnames:
            if featname in algo:
                f.append(float(algo[featname]))
                m.append(float(1))
            else:
                f.append(self.get_random_algo_feat(featname))
                m.append(float(0))
        return np.array(f + m)

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = 'binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'
                target_type = 'Binary', 'Categorical', 'Numerical'
                feat_type = 'Binary', 'Categorical', 'Numerical', 'Mixed'
                metric = 'bac_metric', 'auc_metric', 'f1_metric', 'pac_metric', 'a_metric', 'r2_metric'
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm, for example:
                meta_feature_0 = 0, 1, 2, …
                meta_feature_1 = 0, 1, 2, …
                meta_feature_2 = 0.000001, 0.0001 …


        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'dataset01', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}
        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0', meta_feature_2 : '0.000001'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '1', meta_feature_2 : '0.0001'},
         ...
         '39': {'meta_feature_0': '2', 'meta_feature_1': '2', meta_feature_2 : '0.01'},
         }
        """

        # algorithms_meta_features = expand_algorithms(algorithms_meta_features)

        self.R_best = None

        self.dataset_meta_features = dataset_meta_features
        self.algorithms_meta_features = algorithms_meta_features
        self.past_actions = []
        self.time_budget = float(dataset_meta_features['time_budget'])
        self.remaining_time_budget = float(
            dataset_meta_features['time_budget'])
        self.data_feature = get_data_feature(dataset_meta_features)

        def get_algo_feat(kv):
            return kv[0], self.get_algo_feature(kv[1])

        self.algo_features = dict(
            map(get_algo_feat, algorithms_meta_features.items()))
        self.data_embed = self.get_data_embed(self.data_feature)
        algos = []
        for k, v in self.algo_features.items():
            x = self.get_algo_embed(v)
            logit = torch.dot(self.data_embed, x)
            algos.append((k, logit))
        self.algos = sorted(algos, key=lambda x: x[1], reverse=True)
        self.algo_idx = None
        print("algos:", self.algos)

        self.current_p = 0.1
        self.ranklist = []
        for _, a in enumerate(self.algos):
            _a = a[0]
            if _a not in [x[0] for x in self.ranklist]:
                self.ranklist.append((_a, a[1]))
        self.alc_ranklist = self.ranklist
        self.current_idx = 0
        self.next_ranklist = []

    def get_data_embed(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        return self.datanet(x)

    def get_algo_embed(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        return self.algonet(x)

    def meta_train(self, datasets_meta_features, algorithms_meta_features,
                   train_learning_curves, validation_learning_curves,
                   test_learning_curves):
        """
        Start meta-training the agent

        Parameters
        ----------
        datasets_meta_features : dict of {str : dict of {str : str}}
            Meta-features of meta-training datasets

        algorithms_meta_features : dict of {str : dict of {str : str}}
            The meta_features of all algorithms

        train_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            TRAINING learning curves of meta-training datasets

        validation_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            VALIDATION learning curves of meta-training datasets

        test_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            TEST learning curves of meta-training datasets

        Examples:
        To access the meta-features of a specific dataset:
        >>> datasets_meta_features['dataset01']
        {'name':'dataset01', 'time_budget':'1200', ...}

        To access the validation learning curve of Algorithm 0 on the dataset 'dataset01' :

        >>> validation_learning_curves['dataset01']['0']
        <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>

        >>> validation_learning_curves['dataset01']['0'].times
        [196, 319, 334, 374, 409]

        >>> validation_learning_curves['dataset01']['0'].scores
        [0.6465293662860659, 0.6465293748988077, 0.6465293748988145, 0.6465293748988159, 0.6465293748988159]
        """
        # algorithms_meta_features = expand_algorithms(algorithms_meta_features)

        algo_keys = list(algorithms_meta_features.keys())
        algo_featnames = list(algorithms_meta_features[algo_keys[0]].keys())
        algo_mapping = {}
        algo_values = {}
        for featname in algo_featnames:
            algo_values[featname] = []
            for k, v in algorithms_meta_features.items():
                if featname in v and v[featname] not in algo_values[featname]:
                    algo_values[featname].append(v[featname])
            n_values = len(algo_values[featname])
            # map to one-hot vector.
            # onehotters = np.eye(n_values)
            algo_mapping[featname] = {
                v: [float(v)]  # onehotters[k]
                for k, v in enumerate(algo_values[featname])
            }
        if not self.algonet:
            self.algo_mapping = algo_mapping
            self.algo_values = algo_values
            self.algo_featnames = algo_featnames
            input_size = len(self.algo_featnames) * 2
            #sum([len(self.algo_values[x]) for x in self.algo_featnames])
            self.algonet = AlgoNet(input_size)
            self.optimizer = optim.SGD(
                list(self.datanet.parameters()) +
                list(self.algonet.parameters()),
                lr=self.lr)
        self.train_learning_curves = train_learning_curves
        self.validation_learning_curves = validation_learning_curves
        self.test_learning_curves = test_learning_curves
        self.datasets_meta_features = datasets_meta_features
        self.algorithms_meta_features = algorithms_meta_features

        data_features = dict(
            map(lambda kv: (kv[0], get_data_feature(kv[1])),
                datasets_meta_features.items()))

        def get_algo_feat(kv):
            return kv[0], self.get_algo_feature(kv[1])

        algo_features = dict(
            map(get_algo_feat, algorithms_meta_features.items()))
        print(data_features)
        print(algo_features)

        datanet = self.datanet
        algonet = self.algonet
        optimizer = self.optimizer
        data_keys = list(data_features.keys())
        algo_keys = list(algo_features.keys())

        for _iter in range(self.iters):
            # optimization.
            print('# %s' % _iter)

            data_idx = random.choice(data_keys)
            algo_idx = random.choice(algo_keys)
            algo2_idx = random.choice(algo_keys)

            total_time_budget = float(
                datasets_meta_features[data_idx]['time_budget'])

            data_feat = datasets_meta_features[data_idx]
            data_input = get_data_feature(data_feat)
            data_input = torch.from_numpy(
                data_input.reshape(1, -1).astype(np.float32))
            data_embed = datanet(data_input)

            # early model.
            algo_feat = algorithms_meta_features[algo_idx]
            algo_input = self.get_algo_feature(algo_feat)
            algo_input = torch.from_numpy(
                algo_input.reshape(1, -1).astype(np.float32))
            algo_embed = algonet(algo_input)
            logit1 = torch.tensordot(data_embed, algo_embed, dims=2)

            algo2_feat = algorithms_meta_features[algo2_idx]
            algo2_input = self.get_algo_feature(algo2_feat)
            algo2_input = torch.from_numpy(
                algo2_input.reshape(1, -1).astype(np.float32))
            algo2_embed = algonet(algo2_input)
            logit2 = torch.tensordot(data_embed, algo2_embed, dims=2)

            lc1 = test_learning_curves[data_idx][algo_idx]
            lc2 = test_learning_curves[data_idx][algo2_idx]

            alc1 = compute_alc(lc1, total_time_budget, p=1)
            alc2 = compute_alc(lc2, total_time_budget, p=1)

            if abs(alc1 - alc2) < 1e-6:
                continue

            def get_early_label(alc1, alc2):
                if abs(alc1 - alc2) < 1e-6:
                    return np.array([0.5, 0.5])
                if alc1 > alc2:
                    return np.array([1, 0])
                return np.array([0, 1])

            def soft_cross_entropy(input, target):
                logprobs = F.log_softmax(input, dim=1)
                return -(target * logprobs).sum() / input.shape[0]

            label = get_early_label(alc1, alc2)
            output = torch.stack([logit1, logit2])
            output = output.reshape(1, -1)
            label = torch.from_numpy(label.reshape(1, -1).astype(np.float32))
            loss = soft_cross_entropy(output, label)
            print("loss:", loss)

            # optimize.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def suggest(self, observation):
        """
        Return a new suggestion based on the observation

        Parameters
        ----------
        observation : tuple of (int, float, float, float, float)
            An observation containing: (A, p, t, R_train_A_p, R_validation_A_p)
                1) A: index of the algorithm provided in the previous action,
                2) p: decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]
                3) t: amount of time it took to train A with training data size of p,
                      and make predictions on the training/validation/test sets.
                4) R_train_A_p: performance score on the training set
                5) R_validation_A_p: performance score on the validation set

        Returns
        ----------
        action : tuple of (int, float)
            The suggested action consisting of 2 things:
                (2) A: index of the algorithm to be trained and tested
                (3) p: decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]

        Examples
        ----------
        >>> action = agent.suggest((9, 0.5, 151.73, 0.9, 0.6))
        >>> action
        (9, 0.9)
        """

        # Get observation
        if observation != None:
            A, p, t, R_train_A_p, R_validation_A_p = observation
            self.remaining_time_budget -= t

            if self.R_best is None or R_validation_A_p > self.R_best:
                self.R_best = R_validation_A_p
                self.A_best = A
                self.p_best = p

            self.next_ranklist.append((A, R_validation_A_p))

        print(observation)
        if self.current_p > 1:
            # find remaining A,p.
            A_next = random.randint(0, self.nA - 1)
            p_next = random.choice(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            action = (A_next, p_next)
            self.past_actions.append(action)
            return action

        # print("ranklist: ", self.ranklist)
        A_next = self.ranklist[self.current_idx][0]
        p_next = self.current_p
        self.current_idx += 1

        if (self.remaining_time_budget / self.time_budget <
            (1 - self.current_p)) or (self.current_idx >= len(self.ranklist)):
            self.current_idx = 0
            self.current_p += 0.1
            self.ranklist = sorted(self.next_ranklist, reverse=True)
            for _, r in enumerate(self.alc_ranklist):
                if r[0] not in [x[0] for x in self.ranklist]:
                    self.ranklist.append(r)
            self.next_ranklist = []

        action = (A_next, p_next)
        self.past_actions.append(action)

        return action
