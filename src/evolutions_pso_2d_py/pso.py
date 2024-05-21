import numpy as np


def pso(
    fitfn,
    popsize=10,
    maxiter=50,
    c1=1.49618,
    c2=1.49618,
    w=None,
    min_=0.0,
    max_=5.0,
    random_state=None,
):
    """
    perform the PSO algorithm
    yields the positions of the particles at each iteration
    """
    w = w or (0.729844, 0.729844)
    rng = np.random.default_rng(random_state)
    d = len(w)
    W = np.diag(w)
    positions = rng.uniform(min_, max_, (popsize, d))
    velocities = np.zeros(np.shape(positions), dtype=positions.dtype)
    pbest = np.copy(positions)
    gbest = np.array([min(pbest, key=fitfn)], dtype=positions.dtype)
    yield positions
    for _ in range(maxiter):
        # update personal best positions
        #   compute a True/False ndarray mask that is
        #     True for positions <= personal best and
        #     False otherwise
        cond = np.less_equal(
            np.fromiter(map(fitfn, positions), dtype=np.float32),
            np.fromiter(map(fitfn, pbest), dtype=np.float32),
        )
        #   apply mask to update positions that are better than the previous personal best positions
        pbest[cond, :] = positions[cond, :]
        # update neighborhood best position
        gbest = np.array([min(pbest, key=fitfn)], dtype=positions.dtype)
        # update velocities
        r1 = rng.random(np.shape(positions))
        r2 = rng.random(np.shape(positions))
        velocities @= W  # matrix multiplication -- effectively `w*v` for all velocities
        velocities += c1 * r1 * (pbest - positions)  # cognitive term
        velocities += c2 * r2 * (gbest - positions)  # social term
        # update positions
        positions += velocities
        yield positions


def pso_best(
    fitfn,
    popsize=10,
    maxiter=50,
    c1=1.49618,
    c2=1.49618,
    w=None,
    min_=0.0,
    max_=5.0,
    random_state=None,
):
    """
    perform the PSO algorithm
    returns the best position across all iterations
    """
    generator = pso(
        fitfn,
        popsize=popsize,
        maxiter=maxiter,
        c1=c1,
        c2=c2,
        w=w,
        min_=min_,
        max_=max_,
        random_state=random_state,
    )
    global_best = None
    for p in generator:
        iteration_best = min(p, key=fitfn)
        # replace the global best with the current iteration's best if it's better
        if global_best is None:
            global_best = iteration_best
        else:
            global_best = min(global_best, iteration_best, key=fitfn)
    assert global_best is not None, "global best should not be None"
    return global_best
