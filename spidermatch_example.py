from smm_dijkstra import ShapeMatchModelDijkstra
import gurobipy as gp
from gurobipy import GRB
import time
from utils.sm_utils import *
import scipy.sparse as sp
from utils.misc import robust_lossfun
from utils.vis_util import plot_match, get_cams_and_rotation

time_limit = 60 * 60
## Load data (change dataset accordingly when using files from different datasets)
dataset = "faust" # dt4d_inter, dt4d_intra, smal
filename1 = "datasets/FAUST_r/off/tr_reg_000.off"
filename2 = "datasets/FAUST_r/off/tr_reg_001.off"
shape_opts = {"num_faces": 100} # takes around 100s on M1 Mac Pro


## Load and downsample shapes and compute spidercurve on shape X
VX, FX, vx, fx, vx2VX, VY, FY, vy, fy, vy2VY = shape_loader(filename1, filename2, shape_opts)
ey, ex = get_spider_curve(vx, fx, vy, fy)

## Comptue Features and edge cost matrix
feature_opts = get_feature_opts(dataset)
feat_x, feat_y = get_features(VX, FX, VY, FY, feature_opts)
edge_costs = np.zeros((len(vx), len(vy)))
for i in range(0, len(vx)):
    diff = feat_y[vy2VY, :] - feat_x[vx2VX[i], :]
    edge_costs[i, :] = np.sum(to_numpy(robust_lossfun(torch.from_numpy(diff.astype('float64')),
                                                      alpha=torch.tensor(2, dtype=torch.float64),
                                                      scale=torch.tensor(0.3, dtype=torch.float64))), axis=1)

## ++++++++++++++++++++++++++++++++++++++++
## ++++++++ Solve with SpiderMatch ++++++++
## ++++++++++++++++++++++++++++++++++++++++
smm = ShapeMatchModelDijkstra(vy, ey, vx, ex, edge_costs.T, True, False, False)
E = smm.getCostVector()
RHS = smm.getRHS()
I, J, V = smm.getAVectors()
RHSleq = smm.getRHSleq()
Ileq, Jleq, Vleq = smm.getAleqVectors()

m = gp.Model("spidermatch")
x = m.addMVar(shape=E.shape[0], vtype=GRB.BINARY, name="x")
obj = E.transpose() @ x

A = sp.csr_matrix((V.flatten(), (I.flatten(), J.flatten())), shape=(RHS.shape[0], E.shape[0]))
m.addConstr(A @ x == RHS.flatten(), name="c")

Aleq = sp.csr_matrix((Vleq.flatten(), (Ileq.flatten(), Jleq.flatten())), shape=(RHSleq.shape[0], E.shape[0]))
m.addConstr(Aleq @ x <= RHSleq.flatten(), name="cleq")

m.setObjective(obj, GRB.MINIMIZE)

start_time = time.time()
m.setParam('TimeLimit', time_limit)
m.optimize()
end_time = time.time()
print(f"Optimisation took {end_time - start_time}s")

## Visualise result
result_vec = x.X
matching = smm.getProductSpace()[result_vec.astype('bool'), :]
point_map = matching[:, [0, 2]]
[cam, cams, rotationX, rotationY] = get_cams_and_rotation(dataset)
plot_match(vy, fy, vx, fx, point_map[:, [1, 0]], cam, "", offsetX=[1, 0, 0],
                            rotationShapeX=rotationX, rotationShapeY=rotationY)