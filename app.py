from pandas import read_excel
from numpy import sum, diag, eye, matmul, linalg
from pywebio.input import input_group, file_upload, select
from pywebio.output import use_scope, put_markdown
from pywebio import start_server
from sklearn.cluster import SpectralClustering
from communities.algorithms import louvain_method, girvan_newman, hierarchical_clustering


def lap_mat(sim_mat):
    d = sum(sim_mat, axis=1).tolist()
    z = [i ** (-0.5) for i in d]
    d_ = diag(z)
    an = len(sim_mat)
    mat_lap = eye(an) - matmul(matmul(d_, sim_mat), d_)
    return mat_lap


def main_gap(lap_matrix):
    mat, _ = linalg.eig(lap_matrix)
    mat = sorted(mat, reverse=True)
    g = [mat[i] - mat[i + 1] for i in range(len(mat) - 1)]
    for i in range(1, len(g) - 1):
        if g[i] > g[i - 1] and g[i] > g[i + 1]:
            return i + 1


def get_q(adj_m, c_, c_d):
    in_ = 0
    tot_ = 0
    in_list = []
    tot_list = []
    for i in c_d.keys():
        if i == c_:
            in_list.extend(c_d[i])
        tot_list.extend(c_d[i])
    for i in in_list:
        for z in in_list:
            in_ += adj_m[i][z]
    for i in in_list:
        for z in tot_list:
            tot_ += adj_m[i][z]
    return in_, tot_


def get_Q(adj_mat, c_set):
    m = 0
    adj_matrix_ = adj_mat - eye(len(adj_mat))
    for i in range(len(adj_matrix_)):
        for j in range(len(adj_matrix_[0])):
            m += adj_matrix_[i][j]
    m /= 2
    len_ = 0
    for c in c_set:
        len_ += len(c)

    c_list = [0 for _ in range(len_)]
    for s in range(len(c_set)):
        for i in list(c_set[s]):
            c_list[i] = s
    c_dict = {}
    for i in range(len(c_list)):
        c_dict.setdefault(c_list[i], [])
        c_dict[c_list[i]].append(i)
    Q = 0
    for i in c_dict.keys():
        n, t = get_q(adj_matrix_, i, c_dict)
        Q += n / (2 * m) - (t / (2 * m)) ** 2
    return Q


def louvain_divide(adj_mat):
    community, _ = louvain_method(adj_mat)
    Q = get_Q(adj_mat, community)
    return community, Q


def girvan_newman_divide(adj_mat):
    community, _ = girvan_newman(adj_mat)
    Q = get_Q(adj_mat, community)
    return community, Q


def hierarchical_clustering_divide(adj_mat):
    community = hierarchical_clustering(adj_mat, linkage="complete")
    Q = get_Q(adj_mat, community)
    return community, Q


def spectral_clustering_divide(adj_mat):
    sim_matrix = adj_mat + eye(len(adj_mat))
    lap_matrix = lap_mat(sim_matrix)
    k = main_gap(lap_matrix)
    sc = SpectralClustering(k, affinity='precomputed', n_init=10)
    sc.fit(adj_mat)
    community = sc.labels_.tolist()
    com_dict = {c: [i for i, val in enumerate(community) if val == c] for c in set(community)}
    comm = [set(val) for val in com_dict.values()]
    Q = get_Q(adj_mat, comm)
    return comm, Q


def max_q(adj_mat):
    method_list = [louvain_divide, girvan_newman_divide, hierarchical_clustering_divide, spectral_clustering_divide]
    com_dict = {i: list(method_list[i](adj_mat)) for i in range(len(method_list))}
    max_q_com = com_dict[0][0]
    max_q_ = com_dict[0][1]
    for i in com_dict.keys():
        if com_dict[i][1] > max_q_:
            max_q_ = com_dict[i][1]
            max_q_com = com_dict[i][0]
    return max_q_com, max_q_


method_dict = {
    "louvain": louvain_divide,
    "girvan_newman": girvan_newman_divide,
    "分层聚类": hierarchical_clustering_divide,
    "谱聚类": spectral_clustering_divide,
    "最优": max_q
}


def ask_info():
    with use_scope("ask", clear=True):
        data = input_group(
            "模块划分",
            [
                file_upload("相似度矩阵", name="file"),
                select("模块划分方法", name="method", options=list(method_dict.keys()), value="最优")
            ],
        )
        return data["file"], data["method"]


def put_com(community):
    with use_scope("divide", clear=True):
        put_markdown("## 模块划分结果")
        put_markdown(str(community))


def dsm():
    file, method = ask_info()
    dataframe = read_excel(file["content"], header=None)
    similar_matrix = dataframe.values
    adj_matrix = similar_matrix - eye(len(similar_matrix))
    com, _ = method_dict[method](adj_matrix)
    put_com(com)


if __name__ == "__main__":
    start_server(dsm, port=0, auto_open_webbrowser=True)
