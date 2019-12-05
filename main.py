from EnvPreprocessing import make_env
from parameters import *
from utils import *
from DQNAgent import DQNAgent
from ExperienceBuffer import ExperienceBuffer

print('TF GPU availability:', tf.test.is_gpu_available())


def sample_batch(exp_buffer, batch_size=32):
    """
    Get tuples (s, a, r, next_s, done) for training from experience buffer
    :param exp_buffer: experience replay buffer
    :param batch_size: size of the batch
    :return: dictionary with batch values
    """
    state_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_buffer.sample(batch_size)
    next_obs_batch = next_obs_batch.astype('float32') / 255
    state_batch = state_batch.astype('float32') / 255
    return {
        state_ph: state_batch, action_ph: act_batch, reward_ph: reward_batch,
        next_state_ph: next_obs_batch, done_ph: is_done_batch
    }


def compute_loss(gamma, double_dqn=True, huber_loss=True, clip_delta=1.0):
    """
    Loss computation
    :param gamma: discount coefficient
    :param double_dqn: whether to compute reference value with double DQN method or vanilla one
    :param huber_loss: whether to use huber loss or mse
    :param clip_delta: hyperparameter for huber loss
    :return: loss
    """
    qvalues_agent = agent.get_symb_qvalues(state_ph)
    current_action_qval = tf.reduce_sum(tf.one_hot(action_ph, n_actions) * qvalues_agent, axis=1)

    if double_dqn:
        qvalues_agent_next = agent.get_symb_qvalues(next_state_ph)
        actions_target = tf.argmax(qvalues_agent_next, axis=-1)
        qvalues_target = target_net.get_symb_qvalues(next_state_ph)
        next_action_qval = tf.reduce_sum(tf.one_hot(actions_target, n_actions) * qvalues_target, axis=1)
        reference_qvalues = reward_ph + tf.stop_gradient(gamma * next_action_qval * not_done)
    else:
        qvalues_target = target_net.get_symb_qvalues(next_state_ph)
        reference_qvalues = reward_ph + gamma * tf.reduce_max(qvalues_target, axis=-1) * not_done

    if huber_loss:
        error = current_action_qval - reference_qvalues
        cond = tf.abs(error) < clip_delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = clip_delta * (tf.abs(error) - 0.5 * clip_delta)
        loss_tf = tf.where(cond, squared_loss, linear_loss)
    else:
        loss_tf = (current_action_qval - reference_qvalues) ** 2

    loss_tf = tf.reduce_mean(loss_tf)
    return loss_tf


env = make_env(game_name=GAME_NAME, img_size=IMG_SIZE, make_grey=MAKE_GREY, crop=CROP_FUNC, n_frames=N_FRAMES)

n_actions = env.action_space.n
obs_shape = env.observation_space.shape

tf.reset_default_graph()
sess = tf.InteractiveSession()

agent = DQNAgent('main', input_shape=obs_shape, output_shape=n_actions, eps=EPS_START)
target_net = DQNAgent(scope='Target_net', input_shape=obs_shape, output_shape=n_actions)

state_ph = tf.placeholder(dtype='float32', shape=(None,) + obs_shape, name='state_ph')
action_ph = tf.placeholder(dtype='int32', shape=(None), name='action_ph')
reward_ph = tf.placeholder(dtype='float32', shape=(None), name='reward_ph')
next_state_ph = tf.placeholder(dtype='float32', shape=(None,) + obs_shape, name='next_state_ph')
done_ph = tf.placeholder(dtype='float32', shape=(None), name='done_ph')
not_done = 1 - done_ph

reward_sum_buffer = tf.placeholder(dtype='float32', shape=(None), name='reward_sum_buffer')
mean_reward_ph = tf.placeholder(dtype='float32', shape=(None), name='mean_reward_ph')
mean_reward = tf.reduce_mean(mean_reward_ph)

loss = compute_loss(gamma=GAMMA, double_dqn=DOUBLE_DQN, huber_loss=HUBER_LOSS, clip_delta=CLIP_DELTA)

loss_summary = tf.summary.scalar('DQN loss', loss)
reward_summary = tf.summary.scalar('Mean reward', mean_reward)
reward_sum_summary = tf.summary.scalar('Mean reward', reward_sum_buffer)

train_step = tf.train.AdamOptimizer(0.0000625).minimize(loss, var_list=agent.weights)

sess.run(tf.global_variables_initializer())

load_weights_to_target(agent, target_net)
train_sw = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), sess.graph)
buffer = ExperienceBuffer(EXP_BUFFER_SIZE)

s = env.reset()
s, last_lives = play_and_collect(agent, env, s, env.unwrapped.ale.lives(), buffer, session_n=EXP_BUFFER_START_SIZE)

print('Reward sum in the buffer:', buffer.sum_rewards)

########################
####### Training #######
########################

for frame_num in range(MAX_FRAMES):

    s, last_lives = play_and_collect(agent, env, s, last_lives, buffer, session_n=1)

    if frame_num % UPDATE_NET_N_FRAMES == 0:
        _, loss_val, loss_sum = sess.run([train_step, loss, loss_summary],
                                         sample_batch(buffer, batch_size=BATCH_SIZE))
        if frame_num % 500 == 0:
            train_sw.add_summary(loss_sum, global_step=frame_num)
        if frame_num % 2000 == 0:
            print('Frame number:', frame_num, 'loss', loss_val)
            reward_sum_sum = sess.run(reward_sum_summary, feed_dict={reward_sum_buffer: buffer.sum_rewards})
            train_sw.add_summary(reward_sum_sum, global_step=frame_num)
            print('Buffer size: ', len(buffer), 'Sum of rewards in the buffer: ', buffer.sum_rewards)
            print('Epsilon: ', agent.eps)
            rewards_per_game = evaluate(env, agent, SESSIONS_TO_EVALUATE)
            reward_sum = sess.run(reward_summary, feed_dict={mean_reward_ph: rewards_per_game})
            train_sw.add_summary(reward_sum, global_step=frame_num)
            print('Average reward per game:', np.mean(rewards_per_game))

    if frame_num < EPS_DECAY_STEPS:
        agent.eps -= (EPS_START - EPS_END) / EPS_DECAY_STEPS

    if frame_num % UPDATE_TARGET_NET_N_FRAMES == 0:
        print("Load weights to target")
        load_weights_to_target(agent, target_net)
        if frame_num > EPS_DECAY_STEPS:
            agent.eps = max(agent.eps * 0.99, 0.01)

    if (frame_num + 1) % SAVE_MODEL_FREQ == 0:
        agent.model.save(os.path.join(checkpoint_dir, f"my_model{int((frame_num + 1)//SAVE_MODEL_FREQ)}.h5"))
