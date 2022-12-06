import numpy as np
import sys
import os
import platform
import subprocess
import utils
import uuid
from typing import Dict
import tools


State = Dict[str, np.ndarray]

def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items():
        if key in ('observation', 'static_info'):
            continue

        if key == 'capacity':
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res
    
def _supervised(observation: State, rng: np.random.Generator, net):
    from transform import transform_one
    mask = np.copy(observation['must_dispatch'])
    mask = mask | net(transform_one(observation)).argmax(-1).bool().numpy()
    mask[0] = True
    return _filter_instance(observation, mask)
def solve_static_vrptw(instance, time_limit=3600, tmp_dir="tmp", seed=1, initial_solution=None):

    # Prevent passing empty instances to the static solver, e.g. when
    # strategy decides to not dispatch any requests for the current epoch
    if instance['coords'].shape[0] <= 1:
        yield [], 0
        return

    if instance['coords'].shape[0] <= 2:
        solution = [[1]]
        cost = tools.validate_static_solution(instance, solution)
        yield solution, cost
        return

    os.makedirs(tmp_dir, exist_ok=True)
    instance_filename = os.path.join(tmp_dir, "problem.vrptw")
    tools.write_vrplib(instance_filename, instance, is_vrptw=True)

    executable = os.path.join('baselines', 'hgs_vrptw', 'genvrp')
    # On windows, we may have genvrp.exe
    if platform.system() == 'Windows' and os.path.isfile(executable + '.exe'):
        executable = executable + '.exe'
    assert os.path.isfile(executable), f"HGS executable {executable} does not exist!"
    # Call HGS solver with unlimited number of vehicles allowed and parse outputs
    # Subtract two seconds from the time limit to account for writing of the instance and delay in enforcing the time limit by HGS

    hgs_cmd = [
        executable, instance_filename, str(max(time_limit - 2, 1)),
        '-seed', str(seed), '-veh', '-1', '-useWallClockTime', '1'
    ]
    if initial_solution is None:
        initial_solution = [[i] for i in range(1, instance['coords'].shape[0])]
    if initial_solution is not None:
        hgs_cmd += ['-initialSolution', " ".join(map(str, tools.to_giant_tour(initial_solution)))]
    with subprocess.Popen(hgs_cmd, stdout=subprocess.PIPE, text=True) as p:
        routes = []
        for line in p.stdout:
            line = line.strip()
            # Parse only lines which contain a route
            if line.startswith('Route'):
                label, route = line.split(": ")
                route_nr = int(label.split("#")[-1])
                assert route_nr == len(routes) + 1, "Route number should be strictly increasing"
                routes.append([int(node) for node in route.split(" ")])
            elif line.startswith('Cost'):
                # End of solution
                solution = routes
                cost = int(line.split(" ")[-1].strip())
                check_cost = tools.validate_static_solution(instance, solution)
                assert cost == check_cost, "Cost of HGS VRPTW solution could not be validated"
                yield solution, cost
                # Start next solution
                routes = []
            elif "EXCEPTION" in line:
                raise Exception("HGS failed with exception: " + line)
        assert len(routes) == 0, "HGS has terminated with imcomplete solution (is the line with Cost missing?)"

def log(obj, newline=True, flush=False):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()

def run_baseline(env, oracle_solution=None, strategy=None, seed=None):

    strategy = strategy
    strategy = _supervised if isinstance(strategy, str) else strategy
    seed = seed
    tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
    cleanup_tmp_dir = True
    rng = np.random.default_rng(seed)

    total_reward = 0
    done = False
    # Note: info contains additional info that can be used by your solver
    observation, static_info = env.reset()
    epoch_tlim = static_info['epoch_tlim']
    num_requests_postponed = 0
    while not done:
        epoch_instance = observation['epoch_instance']

        if oracle_solution is not None:
            request_idx = set(epoch_instance['request_idx'])
            epoch_solution = [route for route in oracle_solution if len(request_idx.intersection(route)) == len(route)]
            cost = tools.validate_dynamic_epoch_solution(epoch_instance, epoch_solution)
        else:
            # Select the requests to dispatch using the strategy
            # Note: DQN strategy requires more than just epoch instance, bit hacky for compatibility with other strategies
            epoch_instance_dispatch = strategy({**epoch_instance, 'observation': observation, 'static_info': static_info}, rng)

            # Run HGS with time limit and get last solution (= best solution found)
            # Note we use the same solver_seed in each epoch: this is sufficient as for the static problem
            # we will exactly use the solver_seed whereas in the dynamic problem randomness is in the instance
            solutions = list(solve_static_vrptw(epoch_instance_dispatch, time_limit=epoch_tlim, tmp_dir=tmp_dir, seed=1))
            assert len(solutions) > 0, f"No solution found during epoch {observation['current_epoch']}"
            epoch_solution, cost = solutions[-1]

            # Map HGS solution to indices of corresponding requests
            epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]

        
        num_requests_dispatched = sum([len(route) for route in epoch_solution])
        num_requests_open = len(epoch_instance['request_idx']) - 1
        num_requests_postponed = num_requests_open - num_requests_dispatched

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)
        assert cost is None or reward == -cost, "Reward should be negative cost of solution"
        assert not info['error'], f"Environment error: {info['error']}"

        total_reward += reward

    

    return -total_reward