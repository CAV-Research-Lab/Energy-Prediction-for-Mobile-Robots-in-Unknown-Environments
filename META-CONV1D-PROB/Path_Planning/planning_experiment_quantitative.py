# Planning Experiment in environments with different obstacles per unit area
# Effect of probability prediction
import sys
sys.path.append("../Dataset_Collection/")
import numpy as np
import my_chrono_simulator as mcs
import terrain_generator as tg
from PIL import Image
import math
import random
import A_star
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import interpolate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(0)
VISUALISATION = False
plot_graphs = False

path_data = "./Planning_performance_quantitative/Exp00/"


params = {}
params["Version"] = "planning_experiment_quantitative"
params["map_size_x"] = 30
params["map_size_y"] = 12
params["x0"] = -11
params["y0"] = [-2,2]
params["yaw0"] = [-30,30] # in deg

params["N_RUNS"] = 120
params["DATA_POINTS_RESULTS_CHECKPOINT"] = []

params["run_delay"] = 0.5
params["target_speed"] = 1
params["always_straight"] = False

params["meta_h_scaling"] = 0
params["goal_points"] = [(params["x0"]+2.7*4,0),(params["x0"]+2.7*7,0)]

params["goal_type"] = 'rectangle'
params["goal_radius"] = None
params["goal_depth"] = 4
params["goal_width"] = 6

params["MAX_TIMEOUT"] = 1e5
params["IGNORE_FIRST_GOALS_ENERGY"] = 1
params["N_GOALS_STRAIGHT"] = 0

params["simplex_terrain_types"] = ["wavy","smooth","rough"]
params["simplex_types_probs"] = [0.5,0.5,0]
params["terrain_params_noise"] = 0.05
params["n_terrain_types"] = 1
# RGB colors of obstacle and terrains
OBSTACLE_COLOR = np.array([96,96,96])
TERRAIN_COLOR = np.array([226,221,173])
# params["categories"] = [] # length must be equal to n_terrain_types or empty

# Flag to augment with obstacles
params["AUGMENT_OBSTACLES"] = True
params["obstacles_per_unit_area"] = [0.1]


params["PLANNING_METHOD"] = "meta_conv1D"
params["WHICH_METHOD"] = ["meta_conv1D",
                          "meta_conv1D_prob_gamma",
                          "meta_conv1D_prob_lognorm"] 

params_model = {}

for method in params["WHICH_METHOD"]:
    if "meta" in method:
        params_model[method] = {}
        params_model[method]["LENGTH_SEQUENCE"] = 3
        params_model[method]["MERGED_MODEL"] = True
        params_model[method]["LENGTH_PAST"] = 0
        params_model[method]["LENGTH_SHOTS"] = 3
        params_model[method]["LENGTH_META"] = 3
        params_model[method]["N_SHOTS"] = 3
        params_model[method]["N_META"] = 1
        params_model[method]["MERGED_SHOTS_GEOM"] = True
        params_model[method]["MERGED_META_GEOM"] = True
        params_model[method]["MERGED_SHOTS_OUTPUT"] = True
        params_model[method]["MERGED_META_OUTPUT"] = True
        params_model[method]["REMOVE_CENTRAL_BAND"] = False
        params_model[method]["EXTRA_INFO_NORM_TYPE"] = ""
        params_model[method]["SHOTS_EXTRA_INFO"] = []
        
        if "gamma" in method or "lognorm" in method:
            params_model[method]["ENERGY_NORM_TYPE"] = 'standardize_shift'
            params_model[method]["EPS"] = 1e-3
        else:
            params_model[method]["ENERGY_NORM_TYPE"] = 'standardize'
            params_model[method]["EPS"] = 0
            
        
        method_opts = method.split("_")
        if "conv1D" in method:
            if "prob" not in method:
                params_model[method]["LIST_PPF_MIN"] = None
                params_model[method]["LIST_PPF_MAX"] = None
                params_model[method]["MODEL"] = "model_Meta-Conv1D"
                params_model[method]["MODEL_DIR"] = "../Training/Exp00/log_meta_conv1d/"
            else:
                indx = method_opts.index("prob")
                params_model[method]["MODEL"] = "model_META-CONV1D-{}".format(method_opts[-1].upper())
                params_model[method]["MODEL_DIR"] = "./Training/Exp00/log_meta_conv1d_{}/".format(method_opts[-1])
                
                params_model[method]["LIST_PPF_MIN"] = np.round((1-np.linspace(0.08,0.98,6))/2,2)
                params_model[method]["LIST_PPF_MAX"] = np.round((1+np.linspace(0.08,0.98,6))/2,2)
                                   
            params_model[method]["INPUT_FEATURES_NORM_TYPE"] = ''
            params_model[method]["INPUT_FEATURES"] = ["wheel_trace"]
            params_model[method]["WHEEL_TRACE_SHAPE"] = (78, 40)
            
            

#-------------------------Constants not to change-----------------------------------------------------------#
belly = 0.5
wheelbase = 1.688965*2
wheeltrack = 0.95*2
eps_base = 0.5
eps_track = 0.54
params["discr"] = 0.0625
y1_loc = -wheeltrack/2
x1_loc = wheelbase/2
y2_loc = wheeltrack/2
x2_loc = wheelbase/2
y3_loc = -wheeltrack/2
x3_loc = -wheelbase/2
y4_loc = wheeltrack/2
x4_loc = -wheelbase/2
DEM_size_x = int(params["map_size_x"]/params["discr"] +1)
DEM_size_y = int(params["map_size_y"]/params["discr"] +1)
x = np.linspace(-params["map_size_x"]/2,params["map_size_x"]/2,num=DEM_size_x)
y = np.linspace(-params["map_size_y"]/2,params["map_size_y"]/2,num=DEM_size_y)
Y , X = np.meshgrid(y,x)
# Terrain macro-categories
CATEGORIES = []
CATEGORIES.append([7, 8, 10, 12])  # Clay high moisture content
CATEGORIES.append([5, 0, 22])  # Loose frictional
CATEGORIES.append([1, 3, 4, 13, 14, 15, 16, 17])  # Compact frictional
CATEGORIES.append([9, 11])  # Dry clay
#------------------------------------------------------------------------------------------------------------#

def augment_with_obstacles(obst, Z, Z_obst = None):
    if Z_obst is None:
        Z_obst = np.zeros(Z.shape, dtype=np.int32)
    size = obst["size"]
    max_range = obst["max_range"]
    valley_prob = obst["valley_prob"]
    rigid_prob = obst["rigid_prob"]
    
    map_area_obst = params["map_size_x"]*params["map_size_y"]*(obst["pos_obst_x_max"]-obst["pos_obst_x_min"])*(obst["pos_obst_y_max"]-obst["pos_obst_y_min"])
    n_obstacles = int(map_area_obst*obst["obst_per_unit_area"])
    
    if hasattr(size, "__iter__"):
        if len(size) == 2:
            size = random.randint(size[0],size[1])
        elif len(size) == 1:
            size = size[0]
        else:
            size = 6
    
    for i in range(n_obstacles):
        pos_x = random.randint(int(DEM_size_x*obst["pos_obst_x_min"])+size+1,int(DEM_size_x*obst["pos_obst_x_max"])-size-1)
        pos_y = random.randint(int(DEM_size_y*obst["pos_obst_y_min"])+size+1,int(DEM_size_y*obst["pos_obst_y_max"])-size-1)
        dz = np.empty((size,size))
        dz[:] = np.nan
        size_intorno = size*2+1
        intorno_centre = int((size_intorno-1)/2)
        Z_intorno = Z[pos_x-int((size_intorno-1)/2):pos_x+int((size_intorno-1)/2+1),
                      pos_y-int((size_intorno-1)/2):pos_y+int((size_intorno-1)/2+1)]
        
        if random.random() < valley_prob:
            valley = True
        else:
            valley = False
        
        pixels = random.choices(range(size**2),k=2)
        for pixel in pixels:
            yi = int(pixel%size)
            xi = int(pixel//size)
            if valley:
                dz[xi,yi] = random.uniform(-max_range/2,0)
            else:
                dz[xi,yi] = random.uniform(0,max_range)
        if size%2:
            Z_intorno[intorno_centre-int((size-1)/2):intorno_centre+int((size-1)/2+1),
                      intorno_centre-int((size-1)/2):intorno_centre+int((size-1)/2+1)] += dz
            if random.random() < rigid_prob:
                Z_obst[pos_x-int((size-1)/2):pos_x+int((size-1)/2+1),
                       pos_y-int((size-1)/2):pos_y+int((size-1)/2+1)] = int(1)
        else:
            Z_intorno[intorno_centre-int(size/2-1):intorno_centre+int(size/2+1),
                      intorno_centre-int(size/2-1):intorno_centre+int(size/2+1)] += dz
            if random.random() < rigid_prob:
                Z_obst[pos_x-int(size/2-1):pos_x+int(size/2+1),
                       pos_y-int(size/2-1):pos_y+int(size/2+1)] = int(1)
        mask_missing = np.isnan(Z_intorno)
        if sum(sum(mask_missing)):
            try:
                Z_intorno = fill_missing_value(Z_intorno, mask_missing, 'cubic')
            except:
                pass
            missing_mask2 = np.isnan(Z_intorno)
            if sum(sum(missing_mask2)):
                Z_intorno = fill_missing_value(Z_intorno,missing_mask2,'nearest')
        
        Z[pos_x-int((size_intorno-1)/2):pos_x+int((size_intorno-1)/2+1),
          pos_y-int((size_intorno-1)/2):pos_y+int((size_intorno-1)/2+1)] = Z_intorno

    return Z, Z_obst 
def fill_missing_value(Z_input, mask_missing, method):
    Z_input[mask_missing] = np.nan
    x = np.arange(0, Z_input.shape[1])
    y = np.arange(0, Z_input.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(Z_input)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    Z_filled = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy),  method=method)
    return Z_filled
def generate_Simplex(path_image):
    map_size_x = params["map_size_x"]
    map_size_y = params["map_size_y"]
    DEM_size_x = int(map_size_x/params["discr"] +1)
    DEM_size_y = int(map_size_y/params["discr"] +1)
    # Create random simplex map
    simplex_terrain_type = random.choices(params["simplex_terrain_types"],
                                          weights=params["simplex_types_probs"],k=1)[0]
    simplex = tg.OpenSimplex_Map(map_size_x, map_size_y, params["discr"], terrain_type = simplex_terrain_type, plot = False)
    simplex.sample_generator(plot=plot_graphs)
    Z = simplex.Z
    if params["AUGMENT_OBSTACLES"]:
        Z_obst = np.zeros(Z.shape)
        for obst in params["obstacles"]:
            Z, Z_obst = augment_with_obstacles(obst, Z, Z_obst)
    else:
        Z_obst = np.zeros(Z.shape)
    minz = np.min(Z)      
    Z = Z-minz
    map_height = np.max(Z).item()
    # Save map as image
    Z_pixel = (Z/map_height*255).astype(np.uint8)
    im = Image.new('L', (Z_pixel.shape[1],Z_pixel.shape[0]))
    im.putdata(Z_pixel.reshape(Z_pixel.shape[0]*Z_pixel.shape[1]))
    im = im.rotate(90, expand=True)#
    im = im.resize((int(DEM_size_x/1),int(DEM_size_y/1)), Image.BILINEAR)
    im.save("{}".format(path_image))
    im = im.rotate(-90, expand=True)
    Z_pixel = np.array(list(im.getdata(0))).reshape((im.size[1],im.size[0]))
    Z = Z_pixel.astype(np.float32)/255.0*map_height
    
    # Create Texture for terrain and obstacles
    mask_terrain = Z_obst == 0
    mask_obst = Z_obst == 1
    Z_obst_p = np.ones(Z_obst.shape+(3,))
    Z_obst_p[mask_terrain] = TERRAIN_COLOR
    Z_obst_p[mask_obst] = OBSTACLE_COLOR
    im = Image.fromarray(Z_obst_p.astype('uint8'), 'RGB')
    im = im.rotate(90, expand=True)#
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = im.resize((int(DEM_size_x/1),int(DEM_size_y/1)), Image.NEAREST)
    im.save("{}".format(path_image[:-4]+'_obst.png'))
    
    if plot_graphs:
        simplex.plot_colormesh(Z)
    
    return Z, Z_obst
def isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

def isfloat(x):
    if '[' in x:
        x = x[1:-1]
    try:
        float(x)
        return True
    except ValueError:
        return False

# Create Directories
if not os.path.exists(path_data):
    os.mkdir(path_data)

def main():
    for obst_per_unit_area in params["obstacles_per_unit_area"]:
        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        path_experiment = path_data + "Exp_{}/".format(current_time)
        path_maps = path_experiment + "Simplex_Maps/"
        if not os.path.exists(path_experiment):
            os.mkdir(path_experiment)
        if not os.path.exists(path_maps):
                os.makedirs(path_maps)
        
        params["obst_per_unit_area_this_exp"] = obst_per_unit_area
        params["obstacles"] = []
        params["obstacles"].append({"obst_per_unit_area": obst_per_unit_area,
                                    "size": 6, "max_range": 0.15, 
                                    "valley_prob": 0, "rigid_prob": 1,
                                    "pos_obst_x_min": 0.05,"pos_obst_x_max": 0.9,
                                    "pos_obst_y_min": 0.1,"pos_obst_y_max": 0.9})
        # Save Log File Description Experiment
        file = open(path_experiment+"log_description.txt","w+")
        file.write("Experiment Params:\n")
        for key, val in params.items():
            file.write("{}: {}\n".format(key,val))
        file.write("\nModel Params:\n")
        for method in params["WHICH_METHOD"]:
            file.write("Method {}\n".format(method))
            for key, val in params_model[method].items():
                file.write("{}: {}\n".format(key,val))
            file.write("\n\n")
        file.close()
    
        # Total performance variables to monitor
        energy_true = []
        energy_pred_planner = []
        planning_time = []
        nodes_expanded = []
        evaluated_safe_branches = []
        mean_pitch = []
        mean_roll = []
        var_pitch = []
        var_roll = []
        mean_acc_long = []
        mean_acc_lat = []
        mean_acc_v = []
        run_time = 0
        n_failures = 0
        
        energy_pred_mode = {}
        energy_ppf_max_pred = {}
        energy_ppf_min_pred = {}
        for method in params["WHICH_METHOD"]:
            if "prob" in method:
                energy_pred_mode[method] = []
                energy_ppf_max_pred[method] = {}
                energy_ppf_min_pred[method] = {}
                for pmin, pmax in zip(params_model[method]["LIST_PPF_MIN"],params_model[method]["LIST_PPF_MAX"]):
                    energy_ppf_max_pred[method][pmax] = []
                    energy_ppf_min_pred[method][pmin] = []
        
        # Loop over N_RUNS
        run = 0
        while run < params["N_RUNS"]:    
            current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            # Create random simplex map
            path_image = path_maps + '{}.bmp'.format(current_time)
            Z, Z_obst = generate_Simplex(path_image)
            # Choose Random training file from the 5 trials in each model
            random_train_file = random.randint(0,4)
            # Set model params from log file
            for method in params["WHICH_METHOD"]:
                # Data normalization type
                if params_model[method]["ENERGY_NORM_TYPE"] == 'standardize' or params_model[method]["ENERGY_NORM_TYPE"] == 'standardize_shift':
                    en_val1 = 'mean'
                    en_val2 = 'std'
                elif params_model[method]["ENERGY_NORM_TYPE"] == 'normalize':
                    en_val1 = 'min'
                    en_val2 = 'int'
                else:
                    en_val1 = False
                    en_val2 = False
                    params_model[method]["energy_val1"] = 0
                    params_model[method]["energy_val2"] = 1   
                if params_model[method]["INPUT_FEATURES_NORM_TYPE"] == 'standardize':
                    inp_val1 = 'mean'
                    inp_val2 = 'std'
                elif params_model[method]["INPUT_FEATURES_NORM_TYPE"] == 'normalize':
                    inp_val1 = 'min'
                    inp_val2 = 'int'
                else:
                    inp_val1 = False
                    inp_val2 = False
                if params_model[method]["EXTRA_INFO_NORM_TYPE"] == 'standardize':
                    ex_val1 = 'mean'
                    ex_val2 = 'std'
                elif params_model[method]["EXTRA_INFO_NORM_TYPE"] == 'normalize':
                    ex_val1 = 'min'
                    ex_val2 = 'int'
                else:
                    ex_val1 = False
                    ex_val2 = False  
                # Open log files
                if 'meta' in method:
                    files = os.listdir(params_model[method]["MODEL_DIR"])
                    files.sort()
                    file = files[random_train_file]
                    params_model[method]["model_weights"] = params_model[method]["MODEL_DIR"] + file + '/model_best.hdf5'
                    log_params = params_model[method]["MODEL_DIR"] + file + "/log_params_{}.txt".format(file)
                    l = open(log_params, "r")
                    cont = l.readlines()
                    for line, c in enumerate(cont):
                        if "TERRAIN_IDS_TRAIN" in c:
                            c = c[len("TERRAIN_IDS_TRAIN")+3:-2]
                            id_train = [int(x) for x in c.split(",") if isint(x)]
                            params_model[method]["TERRAIN_IDS_TRAIN"] = id_train
                        if "TERRAIN_IDS_VAL" in c:
                            c = c[len("TERRAIN_IDS_VAL")+3:-2]
                            id_val = [int(x) for x in c.split(",") if isint(x)]
                            params_model[method]["TERRAIN_IDS_VAL"] = id_val   
                        if en_val1 and en_val2:
                            for i, valx in enumerate([en_val1,en_val2]):
                                if "energy_{}".format(valx) in c:
                                    val = [x for x in c.split() if isfloat(x)][0]
                                    if '[' in val:
                                        val = val[1:-1]
                                    params_model[method]["energy_val{}".format(i+1)] = float(val)
                        if inp_val1 and inp_val2:
                            for i, valx in enumerate([inp_val1,inp_val2]):
                                if "INPUT_FEATURES_{}".format(valx) in c:
                                    for n, inp in enumerate(params_model[method]["INPUT_FEATURES"]):
                                        val = [x for x in cont[line+n].split() if isfloat(x)][0]
                                        if '[' in val:
                                            val = val[1:-1]
                                        val = float(val)
                                        if not n:
                                            params_model[method]["INPUT_FEATURES_val{}".format(i+1)] = [val]
                                        else:
                                            params_model[method]["INPUT_FEATURES_val{}".format(i+1)].append(val)
                        if ex_val1 and ex_val2:
                            for i, valx in enumerate([ex_val1,ex_val2]):
                                if "SHOTS_EXTRA_INFO_{}".format(valx) in c:
                                    for n, inp in enumerate(params_model[method]["SHOTS_EXTRA_INFO"]):
                                        val = [x for x in cont[line+n].split() if isfloat(x)][0]
                                        if '[' in val:
                                            val = val[1:-1]
                                        val = float(val)
                                        if not n:
                                            params_model[method]["SHOTS_EXTRA_INFO_val{}".format(i+1)] = [val]
                                        else:
                                            params_model[method]["SHOTS_EXTRA_INFO_val{}".format(i+1)].append(val)
                        for valx in ["min","max"]:
                            if "energy_{}".format(valx) in c:
                                val = [x for x in c.split() if isfloat(x)][0]
                                if '[' in val:
                                    val = val[1:-1]
                                params_model[method]["energy_{}".format(valx)] = float(val)
                        
            # Eperiment terrain type is selected from validation terrains
            # category = [random.randint(0,len(CATEGORIES)-1)]
            # terrain_type = []
            # for cat in category:
            #     valid_types = [c for c in CATEGORIES[cat] if c in params_model[method]["TERRAIN_IDS_VAL"]]
            #     terrain_type.extend(random.sample(valid_types,1))
            
            terrain_type = random.sample(params_model[method]["TERRAIN_IDS_VAL"],params["n_terrain_types"])     
            
            if len(terrain_type)==1:
                terrain_type = terrain_type[0]   
            
                    
            
            
            # Initialise Path Planner
            path_planner = A_star.A_star(params)
            path_planner.set_map(Z,params["map_size_x"], params["map_size_y"], params["discr"])
            path_planner.set_models(params_model)
            
            print()
            print()
            print("Run: {}. Method {}".format(run, params["PLANNING_METHOD"]))
            print("Training terrains: ", params_model[params["PLANNING_METHOD"]]["TERRAIN_IDS_TRAIN"])
            print("Validation terrains: ", params_model[params["PLANNING_METHOD"]]["TERRAIN_IDS_VAL"])
            print("Experiment Terrain: {} (params noise: {}%)".format(terrain_type, params["terrain_params_noise"]*100))
            print()
            # Initialise simulator
            y_start = random.uniform(params["y0"][0],params["y0"][1])
            yaw_start = random.uniform(params["yaw0"][0],params["yaw0"][1])*math.pi/180
            xi, yi, yawi = params["x0"], y_start, yaw_start
            print("Initial pos (x,y,yaw): ({}m, {}m, {}deg)".format(round(xi,2),round(yi,2),round(yawi*180/math.pi,2)))
            z0, roll0, pitch0 = path_planner.starting_pose(xi, yi, yawi)
            sim = mcs.simulator(path_image, Z_obst, 
                                (params["map_size_x"],params["map_size_y"],Z.max().item()), 
                                (xi,yi,z0), (roll0,pitch0,yawi*180/math.pi), 
                                terrain_type, params["terrain_params_noise"], 
                                visualisation = VISUALISATION)
            
            # Planner run variables
            run_failure = False
            run_energy_true = []
            run_energy_pred_planner = []
            run_planning_time = []
            run_nodes_expanded = []
            run_evaluated_safe_branches = []
            run_mean_pitch = []
            run_mean_roll = []
            run_var_pitch = []
            run_var_roll = []
            run_mean_acc_long = []
            run_mean_acc_lat = []
            run_mean_acc_v = []
            run_run_time = 0
            energy_tot = 0
            energy_no_first = 0
            pred_energy_no_first = 0
            tot_segments = 0
            x_ref = []
            y_ref = []
            yaw_ref = []
            
            # Probabilistic model run variables
            run_energy_pred_mode = {}
            run_energy_ppf_max_pred = {}
            run_energy_ppf_min_pred = {}
            for method in params["WHICH_METHOD"]:
                if "prob" in method:
                    run_energy_pred_mode[method] = []
                    run_energy_ppf_max_pred[method] = {}
                    run_energy_ppf_min_pred[method] = {}
                    for pmin, pmax in zip(params_model[method]["LIST_PPF_MIN"],params_model[method]["LIST_PPF_MAX"]):
                        run_energy_ppf_max_pred[method][pmax] = []
                        run_energy_ppf_min_pred[method][pmin] = []
            
            # Loop over goal points
            for n_goal, goal_point in enumerate(params["goal_points"]):
                # Search Path
                print("Run {}. Method {}. Goal {}".format(run, params["PLANNING_METHOD"], n_goal))
                print("Planning Path to {}".format(goal_point))
                if params["always_straight"] or n_goal<params["N_GOALS_STRAIGHT"]:
                    straight_flag = True
                else:
                    straight_flag = False
                if n_goal:
                    solution = path_planner.search([xi, yi, yawi], goal_point, optimization_criteria = 'energy', method = params["PLANNING_METHOD"], straight_flag = straight_flag)
                else:
                    solution = path_planner.search([xi, yi, yawi], goal_point, optimization_criteria = 'distance', method = 'fastest', straight_flag = straight_flag)
                    path_planner.method = params["PLANNING_METHOD"]
                    
                if solution is None:
                    print("Path not found")
                    run_failure = True
                    break
                else:
                    print("Path Found:")
                    for key, val in solution.items():
                        print("{}: {}".format(key,val))
                    print()  
                if n_goal>=params["IGNORE_FIRST_GOALS_ENERGY"]:
                    pred_energy_no_first += sum(solution["cost"])
                    run_energy_pred_planner.extend(solution["cost"])
                    run_planning_time.append(solution["elapsed_time"])
                    run_nodes_expanded.append(solution["nodes_expanded"])
                    run_evaluated_safe_branches.append(solution["evaluated_safe_branches"])
                    for method in params["WHICH_METHOD"]:
                        if "prob" in method:
                            run_energy_pred_mode[method].extend(solution["mode_{}".format(method)])
                            for pmin, pmax in zip(params_model[method]["LIST_PPF_MIN"],params_model[method]["LIST_PPF_MAX"]):
                                run_energy_ppf_max_pred[method][pmax].extend(solution["ppf_max_{}_{}".format(pmax,method)])
                                run_energy_ppf_min_pred[method][pmin].extend(solution["ppf_min_{}_{}".format(pmin,method)])
                         
                # Plot planned path    
                if plot_graphs:
                    xx_ref, yy_ref, yyaw_ref = path_planner.points((xi, yi, yawi), solution["action"])
                    plot_planned_path(Z, xx_ref, yy_ref, yyaw_ref, xi, yi, yawi, [goal_point])
                    x_ref.extend(xx_ref)
                    y_ref.extend(yy_ref)
                    yaw_ref.extend(yyaw_ref)
                
                # Each Action of the path is run and its energy (divided in segments) is measured after execution
                print("Executing path ...")
                for ida in solution["action"]:
                    # Setting points for the controller to track
                    xv,yv,yaw_v = path_planner.points((xi, yi, yawi), [ida])
                    zv = [z0]*len(xv)
                    # Run controller
                    if sim.run((xv,yv,zv), params["target_speed"]):
                        print("Goal Reached!")
                    else:
                        print("Failure")
                        run_failure = True
                        break
                    # Next initial state
                    xi, yi, yawi = xv[-1], yv[-1], yaw_v[-1]
                    #xi, yi, yawi = sim.data_run["X"].values[-1], sim.data_run["Y"].values[-1], sim.data_run["Yaw"].values[-1]*math.pi/180
                    
                    # Retrieving energy from executed actions and saving in path planner memory
                    stat_data = path_planner.segments_stats(sim.data_run, (xv,yv,yaw_v))
                    energy_action = 0
                    for segment in range(len(stat_data)):
                        print("Segment ", segment)
                        print(" Energy {} kJ".format(np.round(stat_data[segment]["energy"],2)))
                        print(" Initial speed {} m/s. Mean speed {} m/s".format(np.round(stat_data[segment]["initial_speed_long"],2), np.round(stat_data[segment]["mean_speed_long"],2)))
                        print(" Pitch (mean,std) [{}, {}] deg".format(np.round(stat_data[segment]["mean_pitch_est"],2),np.round(stat_data[segment]["std_pitch_est"],4)))
                        print(" Roll (mean,std) [{}, {}] deg".format(np.round(stat_data[segment]["mean_roll_est"],2),np.round(stat_data[segment]["std_roll_est"],4)))
                        
                        
                        if plot_graphs and 'conv1D' in method:
                            # xg_shots_string = stat_data[segment]["wheel_trace"]
                            # # Rearranging points in 2D matrix
                            # W = np.array([float(v) for v in xg_shots_string[0].split(' ')]).reshape(params_model[method]["WHEEL_TRACE_SHAPE"])
                            # plot_colormesh(W, "Segment: {}".format(segment))
                            if segment == len(stat_data)-1:
                                plot_colormesh(stat_data[-1]["wheel_trace_tot"], "Merged")
                        
                        if tot_segments: # first ever segment contains initial acceleration (I exclude it)
                            path_planner.add_memory_shot(stat_data[segment], method = params["PLANNING_METHOD"])
                        
                        tot_segments += 1
                        energy_action += stat_data[segment]["energy"]
                        if n_goal>=params["IGNORE_FIRST_GOALS_ENERGY"]:
                            energy_no_first += stat_data[segment]["energy"]
                            run_mean_pitch.append(stat_data[segment]["mean_pitch_meas"])
                            run_mean_roll.append(stat_data[segment]["mean_roll_meas"])
                            run_var_pitch.append(stat_data[segment]["var_pitch_meas"])
                            run_var_roll.append(stat_data[segment]["var_roll_meas"])
                            run_mean_acc_long.append(stat_data[segment]["mean_acc_long"])
                            run_mean_acc_lat.append(stat_data[segment]["mean_acc_lat"])
                            run_mean_acc_v.append(stat_data[segment]["mean_acc_v"])
                    if n_goal>=params["IGNORE_FIRST_GOALS_ENERGY"]:
                        run_energy_true.append(energy_action)
                    print("Energy action: {} kJ".format(energy_action))
                    print()
                    energy_tot += energy_action
                    
                # End of a goal
                if run_failure:
                    break
                # Next initial state after single goal
                if params["always_straight"]:
                    xi, yi, yawi = xv[-1], yv[-1], yaw_v[-1]
                else:
                    xi, yi, yawi = sim.data_run["X"].values[-1], sim.data_run["Y"].values[-1], sim.data_run["Yaw"].values[-1]*math.pi/180
                    
            # End of single run
            run_run_time = sim.data_run.Time.values[-1]
            sim.close()
            print("Total energy: ", energy_tot)   
            print("Total energy without first {} goals true: {}".format(params["IGNORE_FIRST_GOALS_ENERGY"],energy_no_first))
            print("Total energy without first {} goals pred: {}".format(params["IGNORE_FIRST_GOALS_ENERGY"], pred_energy_no_first))
            print("Total planning time without first {} goals: {} s".format(params["IGNORE_FIRST_GOALS_ENERGY"], np.round(sum(run_planning_time),3)))
            print("Total nodes expanded without first {} goals: {}".format(params["IGNORE_FIRST_GOALS_ENERGY"], sum(run_nodes_expanded)))
            print("Total safe branches expanded without first {} goals: {}".format(params["IGNORE_FIRST_GOALS_ENERGY"], sum(run_evaluated_safe_branches)))
            if sum(run_nodes_expanded):
                print("Average node expansion time: {} s".format(np.round(sum(run_planning_time)/sum(run_nodes_expanded),3)))
            if sum(run_evaluated_safe_branches):
                print("Average branch expansion time: {} s".format(np.round(sum(run_planning_time)/sum(run_evaluated_safe_branches),3)))
            
            # Plot single run statistics
            if plot_graphs:
                plot_stats_path(sim.data, Z, x_ref,y_ref,yaw_ref)
              
            del sim 
            
            if run_failure:
                n_failures += int(run_failure)
            else:
                run += 1
                # Assigning single run statistics to total variables
                energy_true.extend(run_energy_true)
                energy_pred_planner.extend(run_energy_pred_planner)
                planning_time.extend(run_planning_time)
                nodes_expanded.extend(run_nodes_expanded)
                evaluated_safe_branches.extend(run_evaluated_safe_branches)
                mean_pitch.extend(run_mean_pitch)
                mean_roll.extend(run_mean_roll)
                var_pitch.extend(run_var_pitch)
                var_roll.extend(run_var_roll)
                mean_acc_long.extend(run_mean_acc_long)
                mean_acc_lat.extend(run_mean_acc_lat)
                mean_acc_v.extend(run_mean_acc_v)
                run_time += run_run_time
                for method in params["WHICH_METHOD"]:
                    if params_model[method]["LIST_PPF_MIN"] is not None and params_model[method]["LIST_PPF_MAX"] is not None:
                        energy_pred_mode[method].extend(run_energy_pred_mode[method])
                        for pmin, pmax in zip(params_model[method]["LIST_PPF_MIN"],params_model[method]["LIST_PPF_MAX"]):
                            energy_ppf_max_pred[method][pmax].extend(run_energy_ppf_max_pred[method][pmax])
                            energy_ppf_min_pred[method][pmin].extend(run_energy_ppf_min_pred[method][pmin])
                            
            # Write partial result into file
            if params["DATA_POINTS_RESULTS_CHECKPOINT"] and len(energy_true) in params["DATA_POINTS_RESULTS_CHECKPOINT"]:
                file = open(path_experiment+"log_results_{}.txt".format(len(energy_true)),"w+")
                file_data = open(path_experiment+"log_results_data_points_{}.txt".format(len(energy_true)),"w+")
            else:
                file = open(path_experiment+"log_results.txt","w+")
                file_data = open(path_experiment+"log_results_data_points.txt","w+")
            file.write("Num runs {}\n".format(run))
            
            r = 1
            file.write("Planner {}\n".format(params["PLANNING_METHOD"]))
            file.write("Num failed runs {}\n".format(n_failures))
            file.write("Data points energy: {}\n".format((len(energy_true))))
            file.write("Tot Energy true: {}\n".format(sum(energy_true)))
            file.write("Tot Energy pred planner: {}\n".format(sum(energy_pred_planner)))
            if len(energy_true):
                file.write("MSE: {}\n".format(mean_squared_error(energy_true, energy_pred_planner)))
                file.write("R2: {}\n".format(r2_score(energy_true, energy_pred_planner)))
                file.write("Perc Pred>=True: {}%\n".format(sum(np.round(np.array(energy_true),r)<=np.round(np.array(energy_pred_planner),r))/len(energy_true)*100))
            file.write("Total planning time: {} s\n".format(np.round(sum(planning_time),3)))
            file.write("Total nodes expanded: {}\n".format(sum(nodes_expanded)))
            file.write("Total safe branches expanded: {}\n".format(sum(evaluated_safe_branches)))
            if sum(nodes_expanded):
                file.write("Average node expansion time: {} s\n".format(np.round(sum(planning_time)/sum(nodes_expanded),3)))
            if sum(evaluated_safe_branches):
                file.write("Average branch expansion time: {} s\n".format(np.round(sum(planning_time)/sum(evaluated_safe_branches),3)))
            file.write("Mean Pitch: {} deg\n".format(np.mean(mean_pitch)))
            file.write("Mean |Roll|: {} deg\n".format(np.mean(np.abs(mean_roll))))
            file.write("Var Pitch: {} deg2\n".format(np.mean(var_pitch)))
            file.write("Var Roll: {} deg2\n".format(np.mean(var_roll)))
            file.write("Mean Acc Long: {} m/s2\n".format(np.mean(mean_acc_long)))
            file.write("Mean Acc Lat: {} m/s2\n".format(np.mean(mean_acc_lat)))
            file.write("Mean Acc V: {} m/s2\n".format(np.mean(mean_acc_v)))
            file.write("Run Time: {} s\n".format(run_time))
            file.write("\n\n")
            
            for method in params["WHICH_METHOD"]:
                if "prob" in method:
                    file.write("Method {}\n".format(method))
                    file.write("Tot Energy pred mode: {}\n".format(sum(energy_pred_mode[method])))
                    if len(energy_true):
                        file.write("MSE: {}\n".format(mean_squared_error(energy_true, energy_pred_mode[method])))
                        file.write("R2: {}\n".format(r2_score(energy_true, energy_pred_mode[method])))
                        file.write("Perc Pred Mode>=True: {}%\n".format(sum(np.round(np.array(energy_true),r)<=np.round(np.array(energy_pred_mode[method]),r))/len(energy_true)*100))  
                    for pmin, pmax in zip(params_model[method]["LIST_PPF_MIN"],params_model[method]["LIST_PPF_MAX"]):
                        file.write("Tot Energy PPF Max ({}) pred: {}\n".format(pmax,sum(energy_ppf_max_pred[method][pmax])))
                        file.write("Tot Energy PPF Min ({}) pred: {}\n".format(pmin,sum(energy_ppf_min_pred[method][pmin])))
                    if len(energy_true):
                        for pmin, pmax in zip(params_model[method]["LIST_PPF_MIN"],params_model[method]["LIST_PPF_MAX"]):
                            file.write("Perc PPF Max ({}) >=True: {}%\n".format(pmax, sum(np.round(np.array(energy_true),r)<=np.round(np.array(energy_ppf_max_pred[method][pmax]),r))/len(energy_true)*100))
                            file.write("Perc PPF Min ({}) >=True: {}%\n".format(pmin, sum(np.round(np.array(energy_true),r)<=np.round(np.array(energy_ppf_min_pred[method][pmin]),r))/len(energy_true)*100))
                            file.write("Perc PPF Min ({}) <=True<= PPF Max ({}) (CI {}): {}%\n".format(pmin,pmax,round(pmax-pmin,2),sum((np.round(np.array(energy_true),r)>=np.round(np.array(energy_ppf_min_pred[method][pmin]),r))*(np.round(np.array(energy_true),r)<=np.round(np.array(energy_ppf_max_pred[method][pmax]),r)))/len(energy_true)*100))  
                    file.write("\n\n")
            file.close()
            
            
            file_data.write("Planner {}\n".format(params["PLANNING_METHOD"]))
            file_data.write("Energy true: {}\n\n".format((energy_true)))
            file_data.write("Energy pred planner: {}\n\n".format(energy_pred_planner))
            for method in params["WHICH_METHOD"]:
                if "prob" in method:
                    file_data.write("Method {}\n".format(method))
                    file_data.write("Energy pred mode: {}\n\n".format(energy_pred_mode[method]))
                    for pmin, pmax in zip(params_model[method]["LIST_PPF_MIN"],params_model[method]["LIST_PPF_MAX"]):
                        file_data.write("Energy PPF Max ({}) pred: {}\n\n".format(pmax,energy_ppf_max_pred[method][pmax]))
                        file_data.write("Energy PPF Min ({}) pred: {}\n\n".format(pmin,energy_ppf_min_pred[method][pmin]))
            file_data.close()

        
        

def plot_stats_path(stats, Z, x_ref,y_ref,yaw_ref):
    dt = stats.Time.values[1]-stats.Time.values[0]
    len_stats = len(stats)
    stats["Motor_Power_kW"] = stats["Motor_Speed"]*stats["Motor_Torque"]/1000   
    energy_tot = 0
    energy_v = np.zeros(len_stats)
    for i in range(len_stats):
        energy_v[i] = energy_tot
        if stats["Motor_Power_kW"][i]>0:
            energy_tot += stats["Motor_Power_kW"][i]*dt
    
    
    plot_planned_path(Z,x_ref,y_ref,yaw_ref,stats["X"].values,stats["Y"].values,stats["Yaw"].values*math.pi/180, params["goal_points"])
    
    plt.figure()
    plt.plot(y_ref,x_ref)
    plt.plot(stats["Y"],stats["X"])
    plt.legend(["Reference","Real"])
    plt.show() 
    
    stats.plot(x="Time", y=["X"]) 
    stats.plot(x="Time", y=["Y"]) 
    stats.plot(x="Time", y=["I_Throttle","I_Braking"]) 
    stats.plot(x="Time", y=["I_Steering"]) 
    
    plt.figure()
    plt.plot(stats["Time"],stats["Motor_Torque"])
    plt.plot(stats["Time"],stats["Motor_Speed"])
    plt.plot(stats["Time"],stats["Motor_Power_kW"])
    plt.plot(stats["Time"],energy_v)
    plt.legend(["Motor Torque[Nm]", "Motor Speed [rad/s]", "Motor Power [kW]", "Energy [kJ]"])
    plt.show()
    
    plt.figure()
    plt.plot(stats["Time"],stats["FWD_Speed"])
    plt.plot(stats["Time"],[params["target_speed"]]*len_stats)
    plt.legend(["FWD Speed", "Target Speed"])
    plt.show()
    
    plt.figure()
    plt.plot(stats["Time"],stats["Roll"])
    plt.plot(stats["Time"],stats["Pitch"])
    plt.legend(["Roll", "Pitch"])
    plt.show()
    
    
def plot_planned_path(Z,x_ref,y_ref,yaw_ref,x_real,y_real,yaw_real, goal_points):
    # Plot DTM and Trajectory
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_title("Path on Map", fontsize = 35)
    ax.tick_params(labelsize=40)
    ax.tick_params(axis='x')
    ax.set_xlabel("[m]", fontsize = 35)
    ax.set_ylabel("[m]", fontsize = 35, rotation = 90, va= "bottom", labelpad = -25)
    im = ax.pcolormesh(Y,X,Z, cmap="Greys",shading='auto')
    cb = fig.colorbar(im, ax =ax)
    cb.ax.tick_params(labelsize=40)
    cb.set_label("[m]", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
    # Draw a Circle around the targets
    for goal_point in goal_points:
        if params["goal_type"] == 'circle':
            circle1 = plt.Circle((goal_point[1], goal_point[0]), params["goal_radius"], color='lightgray')
            ax.add_artist(circle1)
        elif params["goal_type"] == 'rectangle':
            rect = patches.Rectangle((goal_point[1]-params["goal_width"]/2, goal_point[0]-params["goal_depth"]/2), params["goal_width"], params["goal_depth"], color='lightgray',alpha = 0.4)
            ax.add_patch(rect)
    # Plot reference and real trajectory
    plt.plot(y_ref,x_ref)
    plt.plot(y_real,x_real)
    # Plot Robot
    if hasattr(x_real, "__iter__"):
        x0 = x_real[0]
        y0 = y_real[0]
        yaw0 = yaw_real[0]
        xf = x_real[-1]
        yf = y_real[-1]
        yawf = yaw_real[-1] 
    else:
        x0 = x_real
        y0 = y_real
        yaw0 = yaw_real  
    xbl, ybl = bottom_left_wheel((x0, y0, yaw0), wheeltrack, wheelbase)
    ax.add_patch(patches.Rectangle((ybl,xbl), wheeltrack, wheelbase, angle = -yaw0*180/math.pi, alpha = 0.4))
    vy = np.sin(yaw0)*0.8
    vx = np.cos(yaw0)*0.8
    plt.arrow(y0,x0,vy,vx, head_width=0.3, head_length=0.6, color = 'r' )
    if hasattr(x_real, "__iter__"):
        xbl, ybl = bottom_left_wheel((xf,yf,yawf), wheeltrack, wheelbase)
        ax.add_patch(patches.Rectangle((ybl,xbl), wheeltrack, wheelbase, angle = -yawf*180/math.pi, alpha = 0.4))
        vy = np.sin(yawf)*0.8
        vx = np.cos(yawf)*0.8
        plt.arrow(yf,xf,vy,vx, head_width=0.3, head_length=0.6, color = 'r' )
        plt.show()
    plt.show()

def plot_colormesh(Z, title):
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize = 35)
    ax.tick_params(labelsize=40)
    ax.set_xlabel("Y [m]", fontsize = 35)
    ax.set_ylabel("X [m]", fontsize = 35)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    im = plt.imshow(Z, cmap='viridis', interpolation='nearest')
    ax.set_ylim(ax.get_ylim()[::-1])
    cb = fig.colorbar(im, ax =ax)
    cb.ax.tick_params(labelsize=40)
    cb.set_label("Z [m]", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
    plt.show()
        
def bottom_left_wheel(state, width, length):
    (XC,YC,Theta) = state
    pcenter = np.array([[YC],[XC]])
    rcenter = np.matrix(((np.cos(Theta), np.sin(Theta)), (-np.sin(Theta), np.cos(Theta))))
    pbl=pcenter+rcenter*np.array([[-width/2],[-length/2]])
    xbl=pbl[1].item()
    ybl=pbl[0].item()
    return xbl,ybl



if __name__ == "__main__":
    main()
