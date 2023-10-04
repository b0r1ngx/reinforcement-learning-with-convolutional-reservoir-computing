import gymnasium as gym


def test_agent(env_name, policy_update_filename, frame_stack):
    render = True
    render = 'human' if render else None

    policy_update = torch.load(policy_update_filename)
    policy_update.eval()
    logger.debug(f"Loaded: {policy_update}")
    env = make_env(
        env_name, 1,
        render_mode=render,
        apply_api_compatibility=True,
        frame_stack=frame_stack
    )

    # env = gym.make(
    #     id=env_name,
    #
    # )

    obs = env.reset()
    done = np.array([False])
    policy = policy_update.policy
    total_reward = 0
    steps = 0
    while not done.all():
        env.render()
        dist = policy(torch.from_numpy(obs).float())
        act = dist.sample()
        obs, rew, term, trunc, _ = env.step(act.numpy())
        done = term or trunc
        total_reward += rew
        steps += 1

    logger.debug(f"Total reward: {total_reward}")
    logger.debug(f"Episode length: {steps}")
    env.close()


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from rl_baselines.core import logger
    # Necessary for lazy torch.load
    from rl_baselines.rdn import *
    from rl_baselines.rcrc import *

    parser = argparse.ArgumentParser()
    checkpoint = 'runs/Oct04_15-52-39/checkpoint.pth'  # None
    parser.add_argument("--policy-update", "--model", type=str, default=checkpoint)
    # for pcpc use
    env = 'CarRacing-v2'
    parser.add_argument("--env-name", "--env", type=str, default=env)
    parser.add_argument("--frame-stack", type=int, default=None)
    args = parser.parse_args()

    test_agent(
        args.env_name,
        args.policy_update,
        args.frame_stack
    )
