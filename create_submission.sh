#!/bin/bash
DATE=`date "+%Y-%m-%d-%H-%M-%S"`
mkdir -p submissions
cd submissions
rm -rf tmp
mkdir -p tmp/baselines/hgs_vrptw
cp -r ../baselines/hgs_vrptw/{*.cpp,*.h,Makefile} tmp/baselines/hgs_vrptw
cp -r ../baselines/strategies tmp/baselines
cp ../{solver.py,tools.py,environment.py,run.sh,install.sh,requirements.txt,metadata} tmp
cd tmp
zip -o -r --exclude=*.git* --exclude=*__pycache__* --exclude=*.DS_Store* ../submission_$DATE.zip .;
cd ../..
echo "Created submissions/submission_$DATE.zip"