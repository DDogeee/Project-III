from flask import Flask
from flask_restful import reqparse
from flask_restful import Api, Resource
from environment import VRPEnvironment
import functools
import utils
import tools
import numpy as np
import gdown
import os, shutil
from typing import Dict
State = Dict[str, np.ndarray]
from solver import _filter_instance, _supervised, run_baseline

app = Flask(__name__)
api = Api(app)

net = utils.load_model(path = 'GAT_3_128', device='cpu')
parser = reqparse.RequestParser()

predict_args = reqparse.RequestParser()
predict_args.add_argument('instance', type=str, required=True, help='path to instance')
predict_args.add_argument('download', type=bool, default = False, help='download file from drive')




class HelloWorld(Resource):
    def get(self):
        return {'data': 'Hello, my name is PVD'}

class Solve(Resource):
    def post(self):
        args = predict_args.parse_args()
        if args.download:
            if not os.path.exists("temp"):
                os.mkdir("temp")
            url = args.instance
            gdown.download(url, output= "temp/instance.txt", quiet=True, fuzzy=True)
            instance = 'temp/instance.txt'
        else:
            instance = args.instance
        env = VRPEnvironment(seed=1, instance=tools.read_vrplib(instance), epoch_tlim=5, is_static=False)
        strategy = functools.partial(_supervised, net=net)
        cost = str(run_baseline(env, strategy=strategy))
        shutil.rmtree("temp")
        return {"cost" : cost}

api.add_resource(HelloWorld, '/')
api.add_resource(Solve, '/solve')

if __name__ == '__main__':
    app.run()