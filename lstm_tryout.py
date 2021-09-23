import tensorflow as tf
import numpy as np


class DataLoader():
    def __init__(self):
        # 训练数据路径
        path = tf.keras.utils.get_file('nietzsche.txt',
                                       origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()  # 读取到的文本数据字符串
        self.chars = sorted(list(set(self.raw_text)))  # 将字符串的每一个字母或符号拆分，并按ASCII排序
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))  # 字符串为key, chars的index为value，做一个字典char_indices
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))  # chars的index为value，字符串为key，做一个字典indices_char
        self.text = [self.char_indices[c] for c in self.raw_text]  # 以字符的index写成一个序列(即一个int序列)，供后续使用

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index + seq_length])  # batch_size组训练样本
            next_char.append(self.text[index + seq_length])  # batch_size个ground truth，即每组训练样本的下一个字符序号
        # 返回batch_size个seq_length做训练数据，batch_size个[下一个字符]作为ground truth
        return np.array(seq), np.array(next_char)  # [batch_size, seq_length], [num_batch]


class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        # self.cell = tf.keras.layers.LSTMCell(units=256)
        self.emb_1 = tf.keras.layers.Embedding(input_dim=self.num_chars, output_dim=20, input_length=self.seq_length)
        self.emb_2 = tf.keras.layers.Embedding(input_dim=self.num_chars, output_dim=30, input_length=self.seq_length)
        self.cell = tf.keras.layers.LSTM(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        # inputs = tf.one_hot(inputs, depth=self.num_chars)  # [batch_size, seq_length, num_chars] (50,40,57)
        inputs_1 = self.emb_1(inputs)  # [batch_size, seq_length, emb_dim] (50,40,20)
        inputs_2 = self.emb_2(inputs)
        inputs = tf.keras.backend.concatenate([inputs_1, inputs_2])  # concat two embedding tensors as inputs
        state = self.cell.get_initial_state(inputs)  # 获得 RNN 的初始状态
        # state = self.cell.get_initial_state(batch_size=)
        # for t in range(self.seq_length):
        #     output, state = self.cell(inputs[:, t, :], state)  # 通过当前输入和前一时刻的状态，得到输出和当前时刻的状态
        # output, state = self.cell(inputs, initial_state=state)
        output = self.cell(inputs, initial_state=state)
        logits = self.dense(output)
        if from_logits:  # from_logits 参数控制输出是否通过 softmax 函数进行归一化
            return logits
        else:
            return tf.nn.softmax(logits)

    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs, from_logits=True)  # 调用训练好的RNN模型，预测下一个字符的概率分布
        prob = tf.nn.softmax(logits / temperature).numpy()  # 使用带 temperature 参数的 softmax 函数获得归一化的概率分布值
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])  # 使用 np.random.choice 函数，
                         for i in range(batch_size.numpy())])  # 在预测的概率分布 prob 上进行随机取样


data_loader = DataLoader()
num_batches = 100
seq_length = 40
batch_size = 50
learning_rate = 1e-3

model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(seq_length, batch_size)  # 拿到训练数据和ground truth
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:  # 丰富度（即temperature）分别设置为从小到大的 4 个值
    X = X_
    print("diversity %f:" % diversity)
    for t in range(40):
        y_pred = model.predict(X, diversity)  # 预测下一个字符的编号
        print(data_loader.indices_char[y_pred[0]], end='', flush=True)  # 输出预测的字符
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)],
                           axis=-1)  # 将预测的字符接在输入 X 的末尾，并截断 X 的第一个字符，以保证 X 的长度不变
    print("\n")
