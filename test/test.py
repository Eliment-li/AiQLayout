
_max_total_r = -0.2795325836632333
_agent_total_r = -0.2795325836632333
dist = 17.0012
_max_dist = 16.6989

r = (_max_total_r - _agent_total_r * 0.99) * (1 + (dist - _max_dist) / (_max_dist + 1))

print(r)
