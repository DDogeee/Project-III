import os

for i in os.listdir('instances'):
    instance = 'instances/' + i 
    cmd = 'python controller.py --model GAT_3_64 --instance {instance} --epoch_tlim 5 -- baselines/supervised/run.sh'.format(instance = instance)
    os.system(cmd)