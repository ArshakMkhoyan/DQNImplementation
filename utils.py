import numpy as np
import tensorflow as tf


def evaluate(env, agent, n_games=10, t_max=2000, n_random=10, action_to_make=1, breakout=True):
    """
    Evaluate mean reward among n_games
    :param n_random: Number of certain action to make before game evaluation
    :param action_to_make: Certain action to make before game evaluation
    :param env: game environment
    :param agent: agent to play the game
    :param n_games: number of games
    :param t_max: maximum number of actions to play in each game
    :param breakout: whether game is breakout
    :return: mean reward
    """
    rewards_all = []
    for session in range(n_games):
        fire_next = True
        reward_sess = []
        s = env.reset()
        last_lives = env.unwrapped.ale.lives()
        n_random_steps = np.random.randint(1, n_random)
        for _ in range(n_random_steps):
            env.step(action_to_make)
        for _ in range(t_max):
            if fire_next:
                a = 1
                fire_next = False
            else:
                s_type = s.astype('float32') / 255
                qvalues = agent.get_qvalues([s_type])
                a = agent.sample_action(qvalues)[0]
            next_s, r, done, info = env.step(a)
            if breakout:
                if info['ale.lives'] < last_lives:
                    fire_next = True
                    last_lives = info['ale.lives']
                else:
                    fire_next = False
            reward_sess.append(r)
            if done:
                break
            s = next_s
        rewards_all.append(np.sum(reward_sess))
    return np.array(rewards_all)


def play_and_collect(agent, env, cur_state, last_lives, buffer, session_n=1):
    """
    Update experience buffer
    :param agent: agent to play the game
    :param env: game environment
    :param cur_state: current state of the game
    :param last_lives: number of lives left in the game
    :param buffer: experience buffer
    :param session_n: number of observations to add
    :return: next state
    """
    for i in range(session_n):
        s_type = cur_state.astype('float32') / 255
        qvalues = agent.get_qvalues([s_type])
        a = agent.sample_action(qvalues)[0]
        next_s, r, done, info = env.step(a)
        if info['ale.lives'] < last_lives:
            terminal_life = True
        else:
            terminal_life = done
        buffer.add(next_s[..., -1], a, r, terminal_life)
        if done:
            cur_state = env.reset()
            last_lives = env.unwrapped.ale.lives()
        else:
            cur_state = next_s
            last_lives = info['ale.lives']
    return cur_state, last_lives


def load_weights_to_target(agent, target_net):
    """
    Update target network weights, by coping from main agent
    :param agent: main agent
    :param target_net: target network
    :return: None
    """
    assigns = []
    for w_a, w_t in zip(agent.weights, target_net.weights):
        assigns.append(tf.assign(w_t, w_a, validate_shape=True))
    tf.get_default_session().run(assigns)
