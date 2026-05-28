import grid2op

env_name = "l2rpn_case14_sandbox"
env = grid2op.make(env_name)

obs = env.reset(options={"init ts": 123})
