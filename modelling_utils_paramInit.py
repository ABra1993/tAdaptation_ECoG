def paramInit(model_type, scaling):
    """ Initiate parameters as well as defines the initial values and lower/upper bounds.

    params
    -----------------------
    model : string
        model that will be used for fitting (e.g. CbDN, Zhou, Groen or Brands)

    returns
    -----------------------
    params_names: string
        array containing the names of the free model parameters
    x0 : float
        array containing the initial values of the model parameters
    lb: float
        array containing the lower bound values of the model parameters
    ub : float
        array containing the upper bound values of the model parameters

    """

    # scaling parameters
    if (scaling == 'S') | (scaling == 'P') | (scaling == 'S_withoutScrambled'):

        sf_bodies = 1
        sf_bodies_lb = 0.01
        sf_bodies_ub = 200

        sf_buildings = 1
        sf_buildings_lb = 0.01
        sf_buildings_ub = 200

        sf_faces = 1
        sf_faces_lb = 0.01
        sf_faces_ub = 200

        sf_objects = 1
        sf_objects_lb = 0.01
        sf_objects_ub = 200

        sf_scenes = 1
        sf_scenes_lb = 0.01
        sf_scenes_ub = 200

        if (scaling == 'S') | (scaling == 'P'):

            sf_scrambled = 1
            sf_scrambled_lb = 0.01
            sf_scrambled_ub = 200
    
    # model-specific parameters
    if model_type == 'DN':

        shift = 30
        shift_lb = 0
        shift_ub = 70

        scale = 1
        scale_lb = 0.01
        scale_ub = 200

        tau = 0.01
        tau_lb = 0.001
        tau_ub = 1

        n = 1.5
        n_lb = 0.1
        n_ub = 5

        sigma = 0.15
        sigma_lb = 0
        sigma_ub = 1

        tau_a = 0.07
        tau_a_lb = 0.01
        tau_a_ub = 2

        # initiate param for model fit
        if scaling == 'none':

            params_names = ['shift', 'scale', 'tau', 'n', 'sigma', 'tau_a']
            x0 = [shift, scale, tau, n, sigma, tau_a]
            lb = [shift_lb, scale_lb, tau_lb, n_lb, sigma_lb, tau_a_lb]
            ub = [shift_ub, scale_ub, tau_ub, n_ub, sigma_ub, tau_a_ub]

        else:

            if (scaling == 'S') | (scaling == 'P'):

                params_names = ['shift', 'scale', 'tau', 'n', 'sigma', 'tau_a', 'sf_bodies', 'sf_buildings', 'sf_faces', 'sf_objects', 'sf_scenes', 'sf_scrambled']
                x0 = [shift, scale, tau, n, sigma, tau_a, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]
                lb = [shift_lb, scale_lb, tau_lb, n_lb, sigma_lb, tau_a_lb, sf_bodies_lb, sf_buildings_lb, sf_faces_lb, sf_objects_lb, sf_scenes_lb, sf_scrambled_lb]
                ub = [shift_ub, scale_ub, tau_ub, n_ub, sigma_ub, tau_a_ub, sf_bodies_ub, sf_buildings_ub, sf_faces_ub, sf_objects_ub, sf_scenes_ub, sf_scrambled_ub]

            else:

                params_names = ['shift', 'scale', 'tau', 'n', 'sigma', 'tau_a', 'sf_bodies', 'sf_buildings', 'sf_faces', 'sf_objects', 'sf_scenes']
                x0 = [shift, scale, tau, n, sigma, tau_a, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes]
                lb = [shift_lb, scale_lb, tau_lb, n_lb, sigma_lb, tau_a_lb, sf_bodies_lb, sf_buildings_lb, sf_faces_lb, sf_objects_lb, sf_scenes_lb]
                ub = [shift_ub, scale_ub, tau_ub, n_ub, sigma_ub, tau_a_ub, sf_bodies_ub, sf_buildings_ub, sf_faces_ub, sf_objects_ub, sf_scenes_ub]

    elif model_type == 'TTC17':

        shift = 30
        shift_lb = 0
        shift_ub = 70

        scale = 1
        scale_lb = 0.0001
        scale_ub = 200

        weight = 0.5
        weight_lb = 0
        weight_ub = 1

        if scaling == 'none':

            params_names = ['shift', 'scale', 'weight']
            x0 = [shift, scale, weight]
            lb = [shift_lb, scale_lb, weight_lb]
            ub = [shift_ub, scale_ub, weight_ub]

        else:

            params_names = ['shift', 'scale', 'weight', 'sf_bodies', 'sf_buildings', 'sf_faces', 'sf_objects', 'sf_scenes', 'sf_scrambled']
            x0 = [shift, scale, weight, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]
            lb = [shift_lb, scale_lb, weight_lb, sf_bodies_lb, sf_buildings_lb, sf_faces_lb, sf_objects_lb, sf_scenes_lb, sf_scrambled_lb]
            ub = [shift_ub, scale_ub, weight_ub, sf_bodies_ub, sf_buildings_ub, sf_faces_ub, sf_objects_ub, sf_scenes_ub, sf_scrambled_ub]
    
    elif model_type == 'TTC19':

        # initial values: https://github.com/irisgroen/temporalECoG/blob/64593fbafb6d7397926213960236cc313ef51039/temporal_models/TTCSTIG19.json
        # "params":     "weight,        shift,      scale,      tau,        k_on,       k_off,      lambda,     alpha",
        # "x0": [       0.5,            0.06,       2,          4.93,       3,          3,          0.1,        1],
        # "lb": [       0,              0,          0.01,       0,          0.01,       0.01,       0.001,      1],
        # "ub": [       1,              0.1,        200,        100,        10,         10,         100,        100000],

        shift = 30 # ca. 0.06 s
        shift_lb = 0
        shift_ub = 50 # ca. 0.1 s

        scale = 2
        scale_lb = 0.01
        scale_ub = 200

        weight = 0.5
        weight_lb = 0
        weight_ub = 1

        tau = 4.93
        tau_lb = 0
        tau_ub = 100

        k_on = 0.1
        k_on_lb = 0.01
        k_on_ub = 10

        k_off = 0.1
        k_off_lb = 0.01
        k_off_ub = 10

        alpha = 60000
        alpha_lb = 1
        alpha_ub = 100000

        lamb = 0.1
        lamb_lb = 0.001
        lamb_ub = 100

        if scaling == 'none':

            params_names = ['shift', 'scale', 'weight', 'tau', 'k_on', 'k_off', 'alpha', 'lamb']
            x0 = [shift, scale, weight, tau, k_on, k_off, alpha, lamb]
            lb = [shift_lb, scale_lb, weight_lb, tau_lb, k_on_lb, k_off_lb, alpha_lb, lamb_lb]
            ub = [shift_ub, scale_ub, weight_ub, tau_ub, k_on_ub, k_off_ub, alpha_ub, lamb_ub]

        else:

            params_names = ['shift', 'scale', 'weight', 'tau', 'k_on', 'k_off', 'alpha', 'lamb', 'sf_bodies', 'sf_buildings', 'sf_faces', 'sf_objects', 'sf_scenes', 'sf_scrambled']
            x0 = [shift, scale, weight, tau, k_on, k_off, alpha, lamb, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]
            lb = [shift_lb, scale_lb, weight_lb, tau_lb, k_on_lb, k_off_lb, alpha_lb, lamb_lb, sf_bodies_lb, sf_buildings_lb, sf_faces_lb, sf_objects_lb, sf_scenes_lb, sf_scrambled_lb]
            ub = [shift_ub, scale_ub, weight_ub, tau_ub, k_on_ub, k_off_ub, alpha_ub, lamb_ub, sf_bodies_ub, sf_buildings_ub, sf_faces_ub, sf_objects_ub, sf_scenes_ub, sf_scrambled_ub]
    
    return params_names, x0, lb, ub