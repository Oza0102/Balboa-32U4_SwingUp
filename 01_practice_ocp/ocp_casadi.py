import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# ===== パラメータ =====
tf = 1
dt = 0.01
N = int(round(tf / dt)) + 1

m = 1.0
g = 9.81
h = 1.0
w = 100

# True にすると、ペナルティではなく x1(tf)=h, x2(tf)=0 を厳密に満たす問題を解く
hard_constraint = False


def f(x, u):
    """状態方程式の右辺（CasADi の記号でも計算できる書き方）"""
    return ca.vertcat(x[1], u / m - g)


def main():
    opti = ca.Opti()                      # 最適化問題の入れ物

    # 決定変数：全時刻の状態 X と入力 U をまとめて未知数にする
    X = opti.variable(2, N)               # X[:, k] が時刻 k の [x1, x2]
    U = opti.variable(1, N - 1)           # U[k] が区間 k の入力

    # 制約1：初期条件
    opti.subject_to(X[:, 0] == ca.vertcat(0.0, 0.0))

    # 制約2：状態方程式（RK4 で1ステップ進めた値と一致すること）
    for k in range(N - 1):
        xk = X[:, k]
        uk = U[k]
        k1 = f(xk, uk)
        k2 = f(xk + dt / 2 * k1, uk)
        k3 = f(xk + dt / 2 * k2, uk)
        k4 = f(xk + dt * k3, uk)
        opti.subject_to(X[:, k + 1] == xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    # 評価関数
    J = 0.5 * ca.sumsqr(U) * dt
    if hard_constraint:
        opti.subject_to(X[:, N - 1] == ca.vertcat(h, 0.0))   # 終端条件を制約で課す
    else:
        J += 0.5 * w * ((X[0, N - 1] - h)**2 + X[1, N - 1]**2)
    opti.minimize(J)

    # ソルバー IPOPT で解く（print_level=0, sb="yes" で途中経過とバナーの表示を省略）
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})
    sol = opti.solve()

    x = sol.value(X).T                    # 形を (N, 2) にそろえる
    u = sol.value(U)                      # 長さ N-1
    print(f"J = {sol.value(J):.6f}   x1(tf) = {x[-1, 0]:.4f}   x2(tf) = {x[-1, 1]:.4f}")

    t = np.linspace(0, tf, N)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(t[:-1], u);  ax[0].set_title("u")
    ax[1].plot(t, x[:, 0]); ax[1].set_title("x1")
    ax[2].plot(t, x[:, 1]); ax[2].set_title("x2")
    for a in ax:
        a.set_xlabel("t [s]")
    fig.suptitle("CasADi")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
