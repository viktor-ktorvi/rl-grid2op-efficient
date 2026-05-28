import grid2op

env = grid2op.make("l2rpn_case14_sandbox")

available = env.chronics_handler.available_chronics()

print("Number of chronics:", len(available))

for i, chron in enumerate(available):
    print(i, chron)

env.set_id(3)

obs = env.reset()
