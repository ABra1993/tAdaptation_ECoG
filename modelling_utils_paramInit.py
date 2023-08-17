def paramInit(model):
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

    if (model == 'csDN') | (model == 'csDN_withoutGeneralScaling'):

        sf_bodies = 2
        sf_bodies_lb = 0.01
        sf_bodies_ub = 200

        sf_buildings = 2
        sf_buildings_lb = 0.01
        sf_buildings_ub = 200

        sf_faces = 2
        sf_faces_lb = 0.01
        sf_faces_ub = 200

        sf_objects = 2
        sf_objects_lb = 0.01
        sf_objects_ub = 200

        sf_scenes = 2
        sf_scenes_lb = 0.01
        sf_scenes_ub = 200

        sf_scrambled = 2
        sf_scrambled_lb = 0.01
        sf_scrambled_ub = 200

    if (model == 'csDN') | (model == 'DN'):

        sf = 1
        sf_lb = 0.01
        sf_ub = 200

    tau = 0.01
    tau_lb = 0.001
    tau_ub = 1

    tau_a = 0.07
    tau_a_lb = 0.01
    tau_a_ub = 2

    shift = 30
    shift_lb = 0
    shift_ub = 70

    n = 1.5
    n_lb = 0.1
    n_ub = 5

    sigma = 0.15
    sigma_lb = 0
    sigma_ub = 1

    # initiate params
    if (model == 'DN'):

        # initiate param for model fit
        params_names = ['tau', 'shift', 'scale', 'n', 'sigma', 'tau_a']
        x0 = [tau, shift, sf, n, sigma, tau_a]
        lb = [tau_lb, shift_lb, sf_lb, n_lb, sigma_lb, tau_a_lb]
        ub = [tau_ub, shift_ub, sf_ub, n_ub, sigma_ub, tau_a_ub]

    elif model == 'csDN':

        # initiate param for model fit
        params_names = ['tau', 'shift', 'scale', 'n', 'sigma', 'tau_a', 'sf_bodies', 'sf_buildings', 'sf_faces', 'sf_objects', 'sf_scenes', 'sf_scrambled']
        x0 = [tau, shift, sf, n, sigma, tau_a, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]
        lb = [tau_lb, shift_lb, sf_lb, n_lb, sigma_lb, tau_a_lb, sf_bodies_lb, sf_buildings_lb, sf_faces_lb, sf_objects_lb, sf_scenes_lb, sf_scrambled_lb]
        ub = [tau_ub, shift_ub, sf_ub, n_ub, sigma_ub, tau_a_ub, sf_bodies_ub, sf_buildings_ub, sf_faces_ub, sf_objects_ub, sf_scenes_ub, sf_scrambled_ub]

    elif model == 'csDN_withoutGeneralScaling':

        # initiate param for model fit
        params_names = ['tau', 'shift', 'n', 'sigma', 'tau_a', 'sf_bodies', 'sf_buildings', 'sf_faces', 'sf_objects', 'sf_scenes', 'sf_scrambled']
        x0 = [tau, shift, n, sigma, tau_a, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]
        lb = [tau_lb, shift_lb, n_lb, sigma_lb, tau_a_lb, sf_bodies_lb, sf_buildings_lb, sf_faces_lb, sf_objects_lb, sf_scenes_lb, sf_scrambled_lb]
        ub = [tau_ub, shift_ub, n_ub, sigma_ub, tau_a_ub, sf_bodies_ub, sf_buildings_ub, sf_faces_ub, sf_objects_ub, sf_scenes_ub, sf_scrambled_ub]

    return params_names, x0, lb, ub