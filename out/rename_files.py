import os

directory = 'en_pud-ud-test/tikz'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f): 
        separated = filename.split('_')
        if len(separated) != 2:
            continue
        if separated[1] == 'perturbed.tikz':
            new_name=separated[0]+'_ssud.tikz'
            new_name = os.path.join(directory, new_name)
            os.rename(f, new_name)
        elif separated[1] == 'target.tikz':
            new_name=separated[0]+'_target_only.tikz'
            new_name = os.path.join(directory, new_name)
            os.rename(f, new_name) 