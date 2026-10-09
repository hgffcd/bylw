"""Keep training-only reset curricula out of all formal evaluation episodes."""

from functools import wraps


def standard_reset_evaluation(function):
    """Restore both the training reset distribution and its observations on exit.

    The switch remains off for the entire evaluator, including auto-resets.
    The wrapped function must not reset back to training by itself.
    """
    @wraps(function)
    def wrapped(env, agent, episodes, eval_seed, training_seed, *args, **kwargs):
        enabled = env.cfg.near_ring_reset_curriculum_enabled
        env.cfg.near_ring_reset_curriculum_enabled = False
        try:
            return function(
                env, agent, episodes, eval_seed, training_seed, *args, **kwargs
            )
        finally:
            env.cfg.near_ring_reset_curriculum_enabled = enabled
            env.clear_forced_episode_seed_plan()
            env.cfg.seed = training_seed
            env.reset(seed=training_seed)

    return wrapped
