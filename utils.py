import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import time

# Helper functions for loading, processing, and plotting

def make_label(agent, label_append=''):
    """Make label based on agent configuration"""
    label = "{}_batch{}_updateEvery{}_x{}_doubleQ{}_delayPol{}_smoothPol{}_nstep{}".format(
            agent.name, agent.batch_size, agent.update_every, agent.num_updates, \
            agent.doubleQ, agent.delay_policy, agent.smooth_policy, agent.nstep)
    if not len(label_append) == 0:
        label = label + '_' + label_append
    return label

def save_logs(scores, label, results_dir="results/",logs={}):
    """Log scores and save"""
    logs[label] = scores

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)  

    #timestamp = time.strftime("%H%M%S")
    filename = label + ".pkl"
    with open(results_dir + filename, 'wb') as f:
        pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)
    print("Logs saved for: \n" + label + "\n")
    
    return logs

def load_pkl(filepath, verbose=1):
    """Load pkl log file"""
    
    with open(filepath, 'rb') as f:
        logs_loaded = pickle.load(f)
    if verbose:
        print("Loaded: \n{}".format(filepath))
    
    return logs_loaded

