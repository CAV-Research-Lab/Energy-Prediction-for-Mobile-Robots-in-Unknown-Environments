import os
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import models
import math
from scipy import stats
from sklearn.metrics import r2_score
os.environ["CUDA_VISIBLE_DEVICES"] = ""
np.random.seed(0)
random.seed(0)

IDS_TRAIN = []
IDS_TRAIN.append([12, 0, 1, 11])
IDS_TRAIN.append([7, 0, 17, 11])
IDS_TRAIN.append([12, 5, 15, 9])
IDS_TRAIN.append([8, 22, 3, 11])
IDS_TRAIN.append([7, 0, 14, 9])

if not len(IDS_TRAIN):
    n_combs = 5
else:
    n_combs = len(IDS_TRAIN)

params = {}
params["SCRIPT"] = "evaluate_Meta-Plane.py"
params["path_data"] = []
params["path_data"].append("../Dataset_Collection/Datasets/Exp_2021-08-01 15-18-24/Merged_Data/plane_data.csv")
params["path_data"].append("../Dataset_Collection/Datasets/Exp_2021-08-04 16-58-16/Merged_Data/plane_data.csv")


params["BATCH_TYPE"] = "mixed"
params["BATCH_SIZE"] = 32
params["N_ITERATIONS_VAL"] = 0.0625*2
params["MODEL"] = "model_sequence_single_output"
params["N_COMPONENTS"] = 0
params["DISTRIBUTION_TYPE"] = "deterministic"

params["REMOVE_ROUGH"] = False


params["INPUT_FEATURES_NORM_TYPE"] = "standardize"
params["EXTRA_INFO_NORM_TYPE"] = ""
params["ENERGY_NORM_TYPE"] = "standardize"
params["EPS"] = 0

params["LENGTH_SEQUENCE"] = 3
params["MERGED_MODEL"] = True
params["LENGTH_PAST"] = 0
params["LENGTH_SHOTS"] = 3
params["LENGTH_META"] = 3
params["N_SHOTS"] = 3
params["N_META"] = 1
params["MERGED_SHOTS_GEOM"] = True
params["MERGED_META_GEOM"] = True
params["MERGED_SHOTS_OUTPUT"] = True
params["MERGED_META_OUTPUT"] = True

params["path_sum_indices"] = []
params["path_sum_indices"].append("../Dataset_Collection/Datasets/Exp_2021-08-01 15-18-24/Merged_Data/sum_indices.csv")
params["path_sum_indices"].append("../Dataset_Collection/Datasets/EExp_2021-08-04 16-58-16/Merged_Data/sum_indices.csv")

params["TERRAIN_IDS"] = []
params["TRAIN_PERC"] = 1

params["LOG_DIR"] = "./Exp00/log_meta_plane/"
FILES = os.listdir(params["LOG_DIR"])
FILES.sort()

params["INPUT_FEATURES"] = ["mean_pitch_est", "mean_roll_est", "std_pitch_est", "std_roll_est"]

# params["SHOTS_EXTRA_INFO"] = ["mean_pitch_meas", "mean_roll_meas", "var_pitch_meas", "var_roll_meas",
#                               "initial_speed_long", "mean_speed_long", "max_speed_long", "min_speed_long",
#                               "initial_speed_lat", "mean_speed_lat", "max_speed_lat", "min_speed_lat"]
params["SHOTS_EXTRA_INFO"] = []

CATEGORIES = []
CATEGORIES.append([7, 8, 10, 12])  # Clay high moisture content
CATEGORIES.append([5, 0, 22])  # Loose frictional
CATEGORIES.append([1, 3, 4, 13, 14, 15, 16, 17])  # Compact frictional
CATEGORIES.append([9, 11])  # Dry clay

params["TRAINING_DATASETS_PER_CATEGORY"] = [1, 1, 1, 1]


class Batch_Generator():
    """
    Keras Sequence object to train a model on larger-than-memory data.
    modified from: https://stackoverflow.com/questions/51843149/loading-batches-of-images-in-keras-from-pandas-dataframe
    """

    def __init__(self, df, df_idx, batch_size,  mode='train'):
        self.len_df = len(df)
        self.batch_size = batch_size
        self.mode = mode  # shuffle when in train mode
        self.df = {}
        self.df_idx = {}
        self.terrain_ids = list(df["terrain_id"].drop_duplicates().values)
        self.datasets = list(df["dataset"].drop_duplicates().values)
        for terrain_id in self.terrain_ids:
            self.df["{}".format(terrain_id)] = {}
            dfi = df[df.terrain_id==terrain_id]
            self.df_idx["{}".format(terrain_id)] = df_idx[df_idx.terrain_id==terrain_id]
            for dataset in self.datasets:
                self.df["{}".format(terrain_id)]["{}".format(dataset)] = dfi[dfi.dataset==dataset]
                
        if self.mode == 'train':
            self.total_steps =  int(math.ceil(self.len_df / float(self.batch_size))*params["N_ITERATIONS_TRAIN"])
        elif self.mode == 'validation':
            self.total_steps =  int(math.ceil(self.len_df / float(self.batch_size))*params["N_ITERATIONS_VAL"])
        self.step = 0
        self.epoch = 0
    
        self.on_epoch_end()
        
    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(self.len_df)
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch(self):
        for i in range(self.batch_size):
            terrain_id = random.sample(self.terrain_ids,1)[0]
            dfx = self.df_idx["{}".format(terrain_id)]
            
            if params["BATCH_TYPE"] == "mixed" and len(self.datasets)>1:
                if random.random() > 1/3:
                    dataset_id = random.sample(self.datasets, 1)[0]
                    dfx = dfx[dfx["dataset"]==dataset_id]
            
            row = dfx.sample(n=params["N_SHOTS"]+params["N_META"])
            row_shots = row.sample(n=params["N_SHOTS"])
            row_meta = row.drop(row_shots.index)
            
            if len(self.datasets)>1:
                shot_tot = pd.DataFrame()
                for (k,d) in zip(row_shots.k.values,row_shots.dataset.values):
                    shot = self.df["{}".format(terrain_id)]["{}".format(d)].iloc[[k+z for z in range(params["LENGTH_SHOTS"])]]
                    shot_tot = pd.concat([shot_tot,shot])
                meta_tot = pd.DataFrame()
                for (k,d) in zip(row_meta.k.values,row_meta.dataset.values):
                    meta = self.df["{}".format(terrain_id)]["{}".format(d)].iloc[[k+z for z in range(params["LENGTH_META"])]]
                    meta_tot = pd.concat([meta_tot,meta])
            else:
                idx = []
                for k in row_shots.k.values:
                    idx.extend(k+z for z in range(params["LENGTH_SHOTS"]))
                shot_tot = self.df["{}".format(terrain_id)]["0"].iloc[idx]
                idx = []
                for k in row_meta.k.values:
                    idx.extend(k+z for z in range(params["LENGTH_META"]))
                meta_tot = self.df["{}".format(terrain_id)]["0"].iloc[idx]

            xe_shots = shot_tot.loc[:,["energy"]].values
            ye_meta = meta_tot.loc[:,["energy"]].values
            
            if params["SHOTS_EXTRA_INFO"]:
                xei_shots = shot_tot.loc[:, params["SHOTS_EXTRA_INFO"]].values
            
            xg_shots = shot_tot.loc[:,params["INPUT_FEATURES"]].values
            xg_meta = meta_tot.loc[:,params["INPUT_FEATURES"]].values

            if not i:
                XG_SHOTS = np.expand_dims(xg_shots,axis=0)
                XE_SHOTS = np.expand_dims(xe_shots,axis=0)
                XG_META = np.expand_dims(xg_meta,axis=0)
                YE_META = np.expand_dims(ye_meta,axis=0)
                if params["SHOTS_EXTRA_INFO"]:
                    XEI_SHOTS = np.expand_dims(xei_shots, axis=0)
            else:
                XG_SHOTS = np.concatenate([XG_SHOTS,np.expand_dims(xg_shots,axis=0)],axis=0)
                XE_SHOTS = np.concatenate([XE_SHOTS,np.expand_dims(xe_shots,axis=0)],axis=0)
                XG_META = np.concatenate([XG_META,np.expand_dims(xg_meta,axis=0)],axis=0)
                YE_META = np.concatenate([YE_META,np.expand_dims(ye_meta,axis=0)],axis=0)
                if params["SHOTS_EXTRA_INFO"]:
                    XEI_SHOTS = np.concatenate([XEI_SHOTS, np.expand_dims(xei_shots, axis=0)], axis=0)
            
        if params["SHOTS_EXTRA_INFO"]:
            XEI_SHOTS_TOT = np.concatenate([XE_SHOTS, XEI_SHOTS], axis = -1)
        else:
            XEI_SHOTS_TOT = XE_SHOTS
        X = [XG_SHOTS, XEI_SHOTS_TOT, XG_META]
        Y = []
        for sh in range(params["N_SHOTS"]):
            if params["MERGED_META_OUTPUT"]:
                for z in range(params["N_META"]):
                    Y.append(np.sum(YE_META[:,z*params["LENGTH_META"]:(z+1)*params["LENGTH_META"]],axis=1)) 
            else:
                for z in range(params["N_META"]*params["LENGTH_META"]):
                    Y.append(YE_META[:,z])
        
        if self.step == self.total_steps-1:
            self.step = 0
            self.epoch += 1
            self.on_epoch_end()
        else:
            self.step += 1
            
        if self.mode == "prediction":
            return X
        else:
            return X, Y
        
def isfloat(x):
    if '[' in x:
        x = x[1:-1]
    try:
        float(x)
        return True
    except ValueError:
        return False
def main():
    df = pd.DataFrame()
    for i, path_data in enumerate(params["path_data"]):
        dfi = pd.read_csv(path_data)
        ## Removing some of data:
        # Samples without failures
        #df = df[df.goal==1]
        # Samples without initial acceleration
        dfi = dfi[dfi.segment!=0]
        try:
            # Samples without low mean speed
            dfi = dfi[dfi.mean_speed>0.87]
            # Samples without low initial speed
            dfi = dfi[dfi.initial_speed>0.88]
            dfi["mean_speed_long"] = dfi.mean_speed
            dfi["initial_speed_long"] = dfi.initial_speed
        except:
            # Samples without low mean speed
            dfi = dfi[dfi.mean_speed_long>0.87]
            # Samples without low initial speed
            dfi = dfi[dfi.initial_speed_long>0.88]
            
        if params["REMOVE_ROUGH"]:
            # Samples without rough pitch/roll variations
            dfi = dfi.loc[(dfi.var_pitch_est <=params["MAX_VAR_ROUGH"]) | (dfi.var_roll_est <=params["MAX_VAR_ROUGH"])]
        dfi["std_pitch_est"] = dfi["var_pitch_est"].pow(0.5)
        dfi["std_roll_est"] = dfi["var_roll_est"].pow(0.5)
        dfi["curvature"] = dfi["curvature"]*100
        dfi["curvature_tm1"] = dfi["curvature_tm1"]*100
        dfi["index"] = dfi.index
        
        try:
            dfi = dfi.drop(columns=['wheel_types'])
        except:
            pass
        dfi["energy"] = dfi["energy"].clip(lower = 0.0)
        
        dfi["dataset"] = [i]*len(dfi)
        df = pd.concat([df,dfi])
    
    df_sum_indices = pd.DataFrame()
    for i, path_idx in enumerate(params["path_sum_indices"]):         
        dfi = pd.read_csv(path_idx).drop_duplicates()
        dfi["dataset"] = [i]*len(dfi)
        df_sum_indices = pd.concat([df_sum_indices,dfi], ignore_index=True)
    try:
        df = df.drop(columns=['wheel_types'])
    except:
        pass
    df["energy"] = df["energy"].clip(lower = 0.0)
    df["energy"] += params["EPS"]
    
    results = []
    for comb in range(n_combs):
        if not len(IDS_TRAIN):
            # Selection of terrains for training and validation by category
            id_val = []
            id_train = []
            for cat in range(len(CATEGORIES)):
                t_ids = CATEGORIES[cat]
                id_train.extend(random.sample(
                    t_ids, params["TRAINING_DATASETS_PER_CATEGORY"][cat]))
                id_val.extend([t for t in t_ids if t not in id_train])
        else:
            id_train = IDS_TRAIN[comb]
            id_val = []
            for cat in range(len(CATEGORIES)):
                t_ids = CATEGORIES[cat]
                id_val.extend([t for t in t_ids if t not in id_train])
        params["TERRAIN_IDS_TRAIN"] = id_train
        params["TERRAIN_IDS_VAL"] = id_val

        df_train = df[df["terrain_id"].isin(id_train)]
        df_val = df[df["terrain_id"].isin(id_val)]
        if params["TRAIN_PERC"] < 1:
            df_train = df_train.sample(frac=params["TRAIN_PERC"])
        
        # Shots extra info to implement
        log_dir = params["LOG_DIR"] + FILES[comb] + "/log_params_{}.txt".format(FILES[comb])
        l = open(log_dir, "r")
        cont = l.readlines()
        for line, c in enumerate(cont):
            if params["ENERGY_NORM_TYPE"]:
                if params["ENERGY_NORM_TYPE"] == 'standardize' or params["ENERGY_NORM_TYPE"] == 'standardize_shift':
                    val1 = 'mean'
                    val2 = 'std'
                elif params["ENERGY_NORM_TYPE"] == 'normalize':
                    val1 = 'min'
                    val2 = 'int'
                if "energy_{}".format(val1) in c:
                    params["energy_val1"] = [x for x in c.split() if isfloat(x)][0]
                    if '[' in params["energy_val1"]:
                        params["energy_val1"] = params["energy_val1"][1:-1]
                    params["energy_val1"] = float(params["energy_val1"])
                elif "energy_{}".format(val2) in c:
                    params["energy_val2"] = [x for x in c.split() if isfloat(x)][0]
                    if '[' in params["energy_val2"]:
                        params["energy_val2"] = params["energy_val2"][1:-1]
                    params["energy_val2"] = float(params["energy_val2"])
            
            if params["EXTRA_INFO_NORM_TYPE"] and params["SHOTS_EXTRA_INFO"]:
                if params["EXTRA_INFO_NORM_TYPE"] == 'standardize' or params["EXTRA_INFO_NORM_TYPE"] == 'standardize_shift':
                    val1 = 'mean'
                    val2 = 'std'
                elif params["ENERGY_NORM_TYPE"] == 'normalize':
                    val1 = 'min'
                    val2 = 'int'
                if "SHOTS_EXTRA_INFO_{}".format(val1) in c:
                    for n, inp in enumerate(params["SHOTS_EXTRA_INFO"]):
                        val = [x for x in cont[line+n].split() if isfloat(x)][0]
                        if '[' in val:
                            val = val[1:-1]
                        val = float(val)
                        if not n:
                            params["SHOTS_EXTRA_INFO_val1"] = [val]
                        else:
                            params["SHOTS_EXTRA_INFO_val1"].append(val)
                elif "SHOTS_EXTRA_INFO_{}".format(val2) in c:
                    for n, inp in enumerate(params["SHOTS_EXTRA_INFO"]):
                        val = [x for x in cont[line+n].split() if isfloat(x)][0]
                        if '[' in val:
                            val = val[1:-1]
                        val = float(val)
                        if not n:
                            params["SHOTS_EXTRA_INFO_val2"] = [val]
                        else:
                            params["SHOTS_EXTRA_INFO_val2"].append(val)
                            
            if params["INPUT_FEATURES_NORM_TYPE"]:
                if params["INPUT_FEATURES_NORM_TYPE"] == 'standardize' or params["INPUT_FEATURES_NORM_TYPE"] == 'standardize_shift':
                    val1 = 'mean'
                    val2 = 'std'
                elif params["INPUT_FEATURES_NORM_TYPE"] == 'normalize':
                    val1 = 'min'
                    val2 = 'int'
                if "INPUT_FEATURES_{}".format(val1) in c:
                    for n, inp in enumerate(params["INPUT_FEATURES"]):
                        val = [x for x in cont[line+n].split() if isfloat(x)][0]
                        if '[' in val:
                            val = val[1:-1]
                        val = float(val)
                        if not n:
                            params["INPUT_FEATURES_val1"] = [val]
                        else:
                            params["INPUT_FEATURES_val1"].append(val)
                elif "INPUT_FEATURES_{}".format(val2) in c:
                    for n, inp in enumerate(params["INPUT_FEATURES"]):
                        val = [x for x in cont[line+n].split() if isfloat(x)][0]
                        if '[' in val:
                            val = val[1:-1]
                        val = float(val)
                        if not n:
                            params["INPUT_FEATURES_val2"] = [val]
                        else:
                            params["INPUT_FEATURES_val2"].append(val)
        
        # Standardize Energy
        if params["ENERGY_NORM_TYPE"] == 'standardize':
            df_train["energy"] = (df_train["energy"]-params["energy_val1"])/params["energy_val2"]
            df_val["energy"] = (df_val["energy"]-params["energy_val1"])/params["energy_val2"]
        elif params["ENERGY_NORM_TYPE"] == 'normalize':
            df_train["energy"] = (df_train["energy"]-params["energy_val1"])/params["energy_val2"]
            df_val["energy"] = (df_val["energy"]-params["energy_val1"])/params["energy_val2"]
        elif params["ENERGY_NORM_TYPE"] == 'standardize_shift':
            df_train["energy"] = (df_train["energy"]-params["energy_val1"])/params["energy_val2"] + params["energy_val1"]/params["energy_val2"]
            df_val["energy"] = (df_val["energy"]-params["energy_val1"])/params["energy_val2"] + params["energy_val1"]/params["energy_val2"]
        
        # Standardize Extra Info
        if params["SHOTS_EXTRA_INFO"]:
            if params["EXTRA_INFO_NORM_TYPE"] == 'standardize':
                df_train[params["SHOTS_EXTRA_INFO"]] = (df_train[params["SHOTS_EXTRA_INFO"]]-params["SHOTS_EXTRA_INFO_val1"])/params["SHOTS_EXTRA_INFO_val2"]
                df_val[params["SHOTS_EXTRA_INFO"]] = (df_val[params["SHOTS_EXTRA_INFO"]]-params["SHOTS_EXTRA_INFO_val1"])/params["SHOTS_EXTRA_INFO_val2"]
            elif params["EXTRA_INFO_NORM_TYPE"] == 'normalize':
                df_train[params["SHOTS_EXTRA_INFO"]] = (df_train[params["SHOTS_EXTRA_INFO"]]-params["SHOTS_EXTRA_INFO_val1"])/params["SHOTS_EXTRA_INFO_val2"]
                df_val[params["SHOTS_EXTRA_INFO"]] = (df_val[params["SHOTS_EXTRA_INFO"]]-params["SHOTS_EXTRA_INFO_val1"])/params["SHOTS_EXTRA_INFO_val2"]
            elif params["EXTRA_INFO_NORM_TYPE"] == 'standardize_shift':
                df_train[params["SHOTS_EXTRA_INFO"]] = (df_train[params["SHOTS_EXTRA_INFO"]]-params["SHOTS_EXTRA_INFO_val1"])/params["SHOTS_EXTRA_INFO_val2"] + params["SHOTS_EXTRA_INFO_val1"]/params["SHOTS_EXTRA_INFO_val2"]
                df_val[params["SHOTS_EXTRA_INFO"]] = (df_val[params["SHOTS_EXTRA_INFO"]]-params["SHOTS_EXTRA_INFO_val1"])/params["SHOTS_EXTRA_INFO_val2"] + params["SHOTS_EXTRA_INFO_val1"]/params["SHOTS_EXTRA_INFO_val2"]
            
        # Standardize Input Feautures
        if params["INPUT_FEATURES_NORM_TYPE"] == 'standardize':
            df_train[params["INPUT_FEATURES"]] = (df_train[params["INPUT_FEATURES"]]-params["INPUT_FEATURES_val1"])/params["INPUT_FEATURES_val2"]
            df_val[params["INPUT_FEATURES"]] = (df_val[params["INPUT_FEATURES"]]-params["INPUT_FEATURES_val1"])/params["INPUT_FEATURES_val2"]
        elif params["INPUT_FEATURES_NORM_TYPE"] == 'normalize':
            df_train[params["INPUT_FEATURES"]] = (df_train[params["INPUT_FEATURES"]]-params["INPUT_FEATURES_val1"])/params["INPUT_FEATURES_val2"]
            df_val[params["INPUT_FEATURES"]] = (df_val[params["INPUT_FEATURES"]]-params["INPUT_FEATURES_val1"])/params["INPUT_FEATURES_val2"]
        elif params["INPUT_FEATURES_NORM_TYPE"] == 'standardize_shift':
            df_train[params["INPUT_FEATURES"]] = (df_train[params["INPUT_FEATURES"]]-params["INPUT_FEATURES_val1"])/params["INPUT_FEATURES_val2"] + params["INPUT_FEATURES_val1"]/params["INPUT_FEATURES_val2"]
            df_val[params["INPUT_FEATURES"]] = (df_val[params["INPUT_FEATURES"]]-params["INPUT_FEATURES_val1"])/params["INPUT_FEATURES_val2"] + params["INPUT_FEATURES_val1"]/params["INPUT_FEATURES_val2"]
            
            
        print()
        print()
        print("Samples: {}".format(len(df)))
        print("Training Samples: {}".format(len(df_train)))
        print("Validation Samples: {}".format(len(df_val)))
        print("Training Terrains {}".format(id_train))
        print("Validation Terrains {}".format(id_val))

        model = models.get_model(params, summary=True)
        path_weights = params["LOG_DIR"] + FILES[comb] + "/model_best.hdf5"
        model.load_weights(path_weights)
        
        # Initialise data sequences and callbacks
        bg_val = Batch_Generator(df_val, df_sum_indices,params["BATCH_SIZE"], mode='validation')
        
        del df_train, df_val
        
        loss_tot = []
        r2_tot = []
        mse_tot = []
        for i in range(params["N_SHOTS"]):
            mse_tot.append(tf.keras.metrics.MeanSquaredError(name='mse_{}'.format(i)))
            r2_tot.append(tf.keras.metrics.Mean(name='r2_{}'.format(i)))
            loss_tot.append(tf.keras.metrics.Mean(name='loss_{}'.format(i)))
        
            mse_tot[i].reset_states()
            r2_tot[i].reset_states()
            loss_tot[i].reset_states()
        
        while bg_val.epoch < 1:
            if not bg_val.step%10:
                print("Comb: {}. Perc: {}%".format(comb, round(bg_val.step/bg_val.total_steps*100,2)))
            X, Y = bg_val.get_batch()
            
            Y_P = [y.numpy() for y in model(X)]
                
            for i in range(params["N_SHOTS"]):
                # Denormalize energies
                if "shift" in params["ENERGY_NORM_TYPE"]:
                    Y[i] -= params["energy_val1"]/params["energy_val2"]*params["LENGTH_SHOTS"]
                    Y_P[i] -= params["energy_val1"]/params["energy_val2"]*params["LENGTH_SHOTS"]
                    
                if params["ENERGY_NORM_TYPE"]:
                    Y[i] = Y[i]*params["energy_val2"] + params["LENGTH_SHOTS"]*params["energy_val1"]
                    Y_P[i] = Y_P[i]*params["energy_val2"] + params["LENGTH_SHOTS"]*params["energy_val1"]
                
                if params["EPS"]:
                    Y[i] -= params["EPS"]*params["LENGTH_SHOTS"]
                    Y_P[i] -= params["EPS"]*params["LENGTH_SHOTS"]
                
                mse_tot[i](Y[i],Y_P[i])
                r2_tot[i](r2_score(Y[i],Y_P[i]))
                
        
        r = {"comb": comb}
        for i in range(params["N_SHOTS"]-1+params["N_META"]):
            r["mse_{}".format(i)] = mse_tot[i].result().numpy()
            r["r2_{}".format(i)] = r2_tot[i].result().numpy()
                
        results.append(r)
        print(results[-1])
    
    print()
    print(results)

if __name__ == "__main__":
    main()
