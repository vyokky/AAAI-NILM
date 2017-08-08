import os
import sys

SCRIPT = './s2s_training.py'

app_list = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']

for app in app_list:
    logdir = './log/%s'%app
    if not os.path.exists(logdir):
        os.system('mkdir -p %s'%logdir)

    command = 'python -u %s %s > %s/train_val.log'%(SCRIPT, app, logdir)
    print 'Current process:', command
    os.system(command)

print 'Done.'
