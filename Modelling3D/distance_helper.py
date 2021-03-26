from pymol import cmd


def calculate_minimal_distances(epitope, receptor):
    min_dist = 99999
    at1_best = None
    at2_best = None
    for at1 in cmd.index(epitope):
        for at2 in cmd.index(receptor):
            dist = cmd.get_distance(at1, at2)
            # print(dist)
            if dist < min_dist:
                min_dist = dist
                at1_best = at1
                at2_best = at2

    cmd.distance(None, "%s`%d"%at1_best, "%s`%d"%at2_best)
    print(min_dist)


print('Base TRB:')
calculate_minimal_distances('ag_base', 'TRB_base')
print('Base TRA:')
calculate_minimal_distances('ag_base', 'TRA_base')


print('Low TRB:')
calculate_minimal_distances('ag_low', 'TRB_low')
print('Low TRA:')
calculate_minimal_distances('ag_low', 'TRA_low')


print('High TRB:')
calculate_minimal_distances('ag_high', 'TRB_high')
print('High TRA:')
calculate_minimal_distances('ag_high', 'TRA_high')
