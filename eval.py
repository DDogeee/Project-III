import os

for i in os.listdir('instances')[80:]:
    instance = 'instances/' + i 
    cmd = 'python controller.py --model GCN_6_256 --instance {instance} --epoch_tlim 5 -- baselines/supervised/run.sh'.format(instance = instance)
    os.system(cmd)